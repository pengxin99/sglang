#include "utils.h"

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using fp16 = sycl::half;
using namespace sycl;

typedef sycl::half IT;
typedef sycl::half MT;

#define HEAD_INSIDE

template<uint32_t HEAD_DIM, uint32_t GROUP_SIZE>
inline void sdp_esimd_kernel_fp16I_fp16O(
    int64_t num_heads,
    int64_t num_heads_kv,
    uint8_t* query,
    uint8_t* key,
    uint8_t* value,
    void* alibi,
    void* attn_mask,
    uint8_t* output,
    float alpha,
    float beta,
    int64_t kv_len,
    sycl::queue& dpcpp_queue) {

    // assert((num_heads == 32 || num_heads == 28 || num_heads == 24) && hidden_dim == 128);

    const size_t GS = GROUP_SIZE;
    const size_t HD = HEAD_DIM;

    const void * mask = nullptr;
    // void * output = (void *)output_fp16;
    const size_t query_bsz_stride = 0;
    const size_t query_head_stride = HD;
    const size_t query_seq_stride = HD * num_heads;
    const size_t key_bsz_stride = 0;
#ifdef HEAD_INSIDE
    const size_t value_head_stride = HD;
    const size_t key_head_stride = HD;
#else
    const size_t key_head_stride = HD * kv_len;
    const size_t value_head_stride = HD * kv_len;
#endif
    const size_t value_bsz_stride = 0;
    const size_t mask_bsz_stride = 0;
    const size_t mask_head_stride = 0;
    const size_t mask_seq_stride = 0;
    const size_t output_bsz_stride = 0;
    const size_t output_head_stride = HD;
    const size_t output_seq_stride = HD * num_heads;
    const size_t bsz = 1; //batch_size;
    // const size_t num_heads = n_head;
    const size_t num_kv_heads = num_heads_kv;
    const size_t seq_len = 1;
    const size_t context_length = kv_len;

    constexpr int sub_gs = GS / 4;
    constexpr size_t softmax_offset = GS * HD * sizeof(float);
    constexpr size_t max_attn_offset = GS * HD * sizeof(float) + GS * sizeof(float);

    const size_t group_num = num_heads / num_kv_heads;
    const size_t sub_rows = context_length / GS;
    const size_t rem_rows = context_length % GS;
    const float attn_scale = 1 / std::sqrt((float)HD);


    sycl::range<3> global_size(1/*bsz*/, num_heads, seq_len * GS);
    sycl::range<3> local_size(1, 1, GS);

    // std::cout << "global range " << "(" << num_heads << ", " << seq_len * GS << ")" << std::endl;
    // std::cout << "local range " << "(" << GS << ")" << std::endl;

    dpcpp_queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(global_size, local_size),
            [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
                slm_init<GS * HD * sizeof(float) + GS * sizeof(float) * 2>();

                const size_t bsz_idx = item.get_global_id(0);
                const size_t head_idx = item.get_global_id(1);
                const size_t kv_head_idx = head_idx / group_num;
                const size_t seq_idx = item.get_group(2);
                const size_t tid = item.get_local_id(2);

                const IT * query_head = (const IT *)query + bsz_idx * query_bsz_stride
                                                          + head_idx * query_head_stride
                                                          + seq_idx * query_seq_stride;

                const IT * key_head = (const IT *)key + bsz_idx * key_bsz_stride
                                                      + kv_head_idx * key_head_stride;
                const IT * value_head = (const IT *)value + bsz_idx * value_bsz_stride
                                                          + kv_head_idx * value_head_stride;

                const MT * mask_head = (const MT *)mask + bsz_idx * mask_bsz_stride
                                                        + head_idx * mask_head_stride
                                                        + seq_idx * mask_seq_stride;

                IT * output_head = (IT *)output + bsz_idx * output_bsz_stride
                                                + head_idx * output_head_stride
                                                + seq_idx * output_seq_stride;

                simd<IT, HD> query_row = block_load<IT, HD>(query_head) * attn_scale;

                size_t start_row = sub_rows * tid + std::min(tid, rem_rows);
                size_t end_row = start_row + sub_rows + (tid < rem_rows);

                simd<float, HD> accs = 0;
                float softmax = 0;
                float max_attn = -sycl::detail::max_v<float>();
                for (size_t r = start_row; r < end_row; ++r) {
#ifdef HEAD_INSIDE
                    simd<IT, HD> key_row = block_load<IT, HD>(key_head + r * num_kv_heads * HD);
                    simd<IT, HD> value_row = block_load<IT, HD>(value_head + r * num_kv_heads * HD); // (r == 14) ? 34.0f : 0.0f;
#else
                    simd<IT, HD> key_row = block_load<IT, HD>(key_head + r * HD);
                    simd<IT, HD> value_row = block_load<IT, HD>(value_head + r * HD); // (r == 14) ? 34.0f : 0.0f;
#endif
                    float attn = sycl::ext::intel::esimd::detail::sum<float, IT, HD>(
                        query_row * key_row
                    );// + mask_head[r];

                    float new_max_attn = std::max(attn, max_attn);
                    float attn_exp_1 = sycl::ext::intel::esimd::exp(max_attn - new_max_attn);
                    float attn_exp_2 = sycl::ext::intel::esimd::exp(attn - new_max_attn);
                    accs = accs * attn_exp_1 + value_row * attn_exp_2;
                    softmax = softmax * attn_exp_1 + attn_exp_2;
                    max_attn = new_max_attn;
                }

                slm_block_store<float, 1>(max_attn_offset + tid * sizeof(float), max_attn);

                barrier();

                float max_attn_max = hmax<float, float, GS>(
                    slm_block_load<float, GS>(max_attn_offset)
                );

                if (max_attn < max_attn_max) {
                    float attn_exp = sycl::ext::intel::esimd::exp(max_attn - max_attn_max);
                    accs *= attn_exp;
                    softmax *= attn_exp;
                }

                slm_block_store(tid * HD * sizeof(float), accs);
                slm_block_store<float, 1>(softmax_offset + tid * sizeof(float), softmax);

                barrier();

                if (tid < 4) {
                    simd<float, HD> accs = 0;
                    #pragma unroll
                    for (int i = 0; i < sub_gs; ++i) {
                        accs += slm_block_load<float, HD>((tid * sub_gs + i) * HD * sizeof(float));
                    }
                    slm_block_store<float, HD>(tid * sub_gs * HD * sizeof(float), accs);
                }

                barrier();

                if (tid == 0) {
                    float softmax_sum = sycl::ext::intel::esimd::detail::sum<float, float, GS>(
                        slm_block_load<float, GS>(softmax_offset)
                    );

                    simd<float, HD> accs = 0;
                    #pragma unorll
                    for (int i = 0; i < 4; ++i) {
                        accs += slm_block_load<float, HD>(i * sub_gs * HD * sizeof(float));
                    }

                    simd<IT, HD> result = accs / softmax_sum;
                    block_store<IT, HD>(output_head, result);
                }
            });
      });


}