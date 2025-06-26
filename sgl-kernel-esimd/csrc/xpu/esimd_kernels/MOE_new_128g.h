#include "utils.h"
using namespace sycl::ext::intel::esimd;
using fp16 = sycl::half;
// using ST = at::ScalarType;

constexpr int QK2 = 128; // compatible
constexpr int SBS2 = 2; // compatible

constexpr int BLOCK_SIZE2 = QK2 / 2;

constexpr int SCALE_SIZE2 = sizeof(fp16);


ESIMD_INLINE auto load_qblocks2(const uint8_t * weight, const uint8_t * scale) {
    simd<uint8_t, BLOCK_SIZE2 * SBS2> ybytes = block_load<uint8_t, BLOCK_SIZE2 * SBS2>(weight);
    const simd<fp16, SBS2> scales = block_load<fp16, SBS2>((const fp16 *)scale);

    simd<fp16, QK2 * SBS2> yvs;
    #pragma unroll
    for (int i = 0; i < SBS2; ++i) {
        simd<uint8_t, QK2> uyv;
        // uyv.select<QK2 / 2, 1>(0) = ybytes.template select<QK2 / 2, 1>(i * QK2 / 2) & (uint8_t)0xF;
        // uyv.select<QK2 / 2, 1>(QK2 / 2) = ybytes.template select<QK2 / 2, 1>(i * QK2 / 2) >> (uint8_t)4;
        uyv.select<QK2 / 2, 2>(0) = ybytes.template select<QK2 / 2, 1>(i * QK2 / 2) & (uint8_t)0xF; // compatible
        uyv.select<QK2 / 2, 2>(1) = ybytes.template select<QK2 / 2, 1>(i * QK2 / 2) >> (uint8_t)4; // compatible
        yvs.template select<QK2, 1>(i * QK2) = (uyv.bit_cast_view<int8_t>() - (int8_t)8) * scales[i];
    }
    return yvs;
}

// C++ doesn't support function template partial specialization, so write a new version for SBS2=1
ESIMD_INLINE auto load_qblock2(const uint8_t * weight, const uint8_t * scale) {
    simd<uint8_t, BLOCK_SIZE2> ybytes = block_load<uint8_t, BLOCK_SIZE2>(weight);
    fp16 scales = *(const fp16 *)scale;

    simd<uint8_t, QK2> uyv;
    // uyv.select<QK2 / 2, 1>(0) = ybytes & (uint8_t)0xF;
    // uyv.select<QK2 / 2, 1>(QK2 / 2) = ybytes >> (uint8_t)4;
    uyv.select<QK2 / 2, 2>(0) = ybytes & (uint8_t)0xF; // compatible
    uyv.select<QK2 / 2, 2>(1) = ybytes >> (uint8_t)4; // compatible
    simd<fp16, QK2> yv = (uyv.bit_cast_view<int8_t>() - (int8_t)8) * scales;

    return yv;
}

template <typename IT, const int VS, const int GS, const int ES>
static void moe_forward_up_kernel_128g(
    const void* input_ptr,
    const int64_t* indexs,
    const uint64_t* gate_addrs,
    const uint64_t* up_addrs,
    const uint64_t* gate_scales_addrs,
    const uint64_t* up_scales_addrs,
    void* output_ptr,
    const int num_experts,
    const int state_size,
    const int output_size,
    // at::Device device
    sycl::queue& dpcpp_queue
) {
    static_assert(ES == 8 || ES == 16 || ES == 32);
    assert(output_size % VS == 0);

    const int nb = state_size / QK2;
    const int nsb = nb / SBS2;

    const IT* input = static_cast<const IT *>(input_ptr);
    IT* output = static_cast<IT *>(output_ptr);

    sycl::range<2> global_size(num_experts, output_size / VS * GS);
    sycl::range<2> local_size(1, GS);

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<2>(global_size, local_size),
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                slm_init<2 * GS * VS * sizeof(float)>();

                const int eid = item.get_global_id(0);
                const int tid = item.get_local_id(1);
                const int vid = item.get_group(1) * VS;

                // get correct expert
                const uint8_t * weight1 = (const uint8_t *)(gate_addrs[indexs[eid]]);
                const uint8_t * weight2 = (const uint8_t *)(up_addrs[indexs[eid]]);

                // const uint8_t* scales1 = weight1 + (int64_t)output_size * nb * BLOCK_SIZE2;
                // const uint8_t* scales2 = weight2 + (int64_t)output_size * nb * BLOCK_SIZE2;
                const uint8_t * scales1 = (const uint8_t *)(gate_scales_addrs[indexs[eid]]);
                const uint8_t * scales2 = (const uint8_t *)(up_scales_addrs[indexs[eid]]);

                const uint8_t * weight1_base = weight1 + nb * BLOCK_SIZE2 * vid;
                const uint8_t * scale1_base = scales1 + nb * SCALE_SIZE2 * vid;
                const uint8_t * weight2_base = weight2 + nb * BLOCK_SIZE2 * vid;
                const uint8_t * scale2_base = scales2 + nb * SCALE_SIZE2 * vid;

                simd<IT, VS * ES> accvs1{};
                simd<IT, VS * ES> accvs2{};

                for (int s = tid; s < nsb; s += GS) {
                    simd<IT, SBS2 * QK2> xvs = block_load<IT, SBS2 * QK2>(input + s * SBS2 * QK2);

                    #pragma unroll
                    for (int v = 0; v < VS; ++v) {
                        simd<fp16, SBS2 * QK2> yvs1 = load_qblocks2(
                            weight1_base + v * nb * BLOCK_SIZE2 + s * SBS2 * BLOCK_SIZE2,
                            scale1_base + v * nb * SCALE_SIZE2 + s * SBS2 * SCALE_SIZE2
                        );

                        #pragma unroll
                        for (int i = 0; i < SBS2 * QK2; i += ES) {
                            accvs1.template select<ES, 1>(v * ES) +=
                                xvs.template select<ES, 1>(i) *
                                yvs1.template select<ES, 1>(i);
                        }

                        simd<fp16, SBS2 * QK2> yvs2 = load_qblocks2(
                            weight2_base + v * nb * BLOCK_SIZE2 + s * SBS2 * BLOCK_SIZE2,
                            scale2_base + v * nb * SCALE_SIZE2 + s * SBS2 * SCALE_SIZE2
                        );

                        #pragma unroll
                        for (int i = 0; i < SBS2 * QK2; i += ES) {
                            accvs2.template select<ES, 1>(v * ES) +=
                                xvs.template select<ES, 1>(i) *
                                yvs2.template select<ES, 1>(i);
                        }
                    }
                }

                for (int b = nsb * SBS2 + tid; b < nb; b += GS) {
                    simd<IT, QK2> xv = block_load<IT, QK2>(input + b * QK2);

                    #pragma unroll
                    for (int v = 0; v < VS; ++v) {
                        simd<fp16, QK2> yv1 = load_qblock2(
                            weight1_base + v * nb * BLOCK_SIZE2 + b * BLOCK_SIZE2,
                            scale1_base + v * nb * SCALE_SIZE2 + b * SCALE_SIZE2
                        );

                        #pragma unroll
                        for (int i = 0; i < QK2; i += ES) {
                            accvs1.template select<ES, 1>(v * ES) +=
                                xv.template select<ES, 1>(i) *
                                yv1.template select<ES, 1>(i);
                        }

                        simd<fp16, QK2> yv2 = load_qblock2(
                            weight2_base + v * nb * BLOCK_SIZE2 + b * BLOCK_SIZE2,
                            scale2_base + v * nb * SCALE_SIZE2 + b * SCALE_SIZE2
                        );

                        #pragma unroll
                        for (int i = 0; i < QK2; i += ES) {
                            accvs2.template select<ES, 1>(v * ES) +=
                                xv.template select<ES, 1>(i) *
                                yv2.template select<ES, 1>(i);
                        }
                    }
                }

                simd<float, VS> accs1;
                simd<float, VS> accs2;
                #pragma unroll
                for(int v = 0; v < VS; ++v) {
                    accs1[v] = sycl::ext::intel::esimd::detail::sum<float, IT, ES>(
                        accvs1.template select<ES, 1>(v * ES)
                    );
                    accs2[v] = sycl::ext::intel::esimd::detail::sum<float, IT, ES>(
                        accvs2.template select<ES, 1>(v * ES)
                    );
                }

                slm_block_store<float, VS>(tid * VS * sizeof(float), accs1);
                slm_block_store<float, VS>(tid * VS * sizeof(float) + GS * VS * sizeof(float), accs2);

                barrier();

                if (tid == 0) {
                    #pragma unroll
                    for (int i = 1; i < GS; ++i) {
                        accs1 += slm_block_load<float, VS>(i * VS * sizeof(float));
                    }

                    #pragma unroll
                    for (int i = 1; i < GS; ++i) {
                        accs2 += slm_block_load<float, VS>(i * VS * sizeof(float) + GS * VS * sizeof(float));
                    }

                    simd<float, VS> result = accs1 / (1 + exp(-accs1)) * accs2;

                    block_store<IT, VS>(output + eid * output_size + vid, result);
                }
            }
        );
    };

    // utils::submit_kernel(cgf, device, "moe forward up kernel");
    dpcpp_queue.submit(cgf);
}
template <typename IT, const int VS, const int GS, const int ES>
static void moe_forward_down_kernel_128g(
    const void* input_ptr,
    const int64_t* indexs,
    const void* eweights,
    const uint64_t* down_addrs,
    const uint64_t* down_scales_addrs,
    void * output_ptr,
    const int num_experts,
    const int state_size,
    const int output_size,
    // at::Device device
    sycl::queue& dpcpp_queue
) {
    static_assert(ES == 8 || ES == 16 || ES == 32);
    assert(output_size % VS == 0);

    const int nb = state_size / QK2;
    const int nsb = nb / SBS2;

    sycl::range<2> global_size(num_experts, output_size / VS * GS);
    sycl::range<2> local_size(1, GS);

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<2>(global_size, local_size),
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                slm_init<GS * VS * sizeof(float)>();

                const int eid = item.get_global_id(0);
                const int tid = item.get_local_id(1);
                const int vid = item.get_group(1) * VS;

                const IT eweight = ((const IT*)eweights)[eid];
                const uint8_t* weight = (const uint8_t *)(down_addrs[indexs[eid]]);
                // const uint8_t* scales = weight + (int64_t)output_size * nb * BLOCK_SIZE2;
                const uint8_t* scales = (const uint8_t *)(down_scales_addrs[indexs[eid]]);
                const IT* input = static_cast<const IT *>(input_ptr) + eid * state_size;
                IT* output = static_cast<IT *>(output_ptr) + eid * output_size;

                const uint8_t * weight_base = weight + nb * BLOCK_SIZE2 * vid;
                const uint8_t * scale_base = scales + nb * SCALE_SIZE2 * vid;

                simd<IT, VS * ES> accvs{};

                for (int s = tid; s < nsb; s += GS) {
                    simd<IT, SBS2 * QK2> xvs = block_load<IT, SBS2 * QK2>(input + s * SBS2 * QK2);

                    #pragma unroll
                    for (int v = 0; v < VS; ++v) {
                        simd<fp16, SBS2 * QK2> yvs = load_qblocks2(
                            weight_base + v * nb * BLOCK_SIZE2 + s * SBS2 * BLOCK_SIZE2,
                            scale_base + v * nb * SCALE_SIZE2 + s * SBS2 * SCALE_SIZE2
                        );

                        #pragma unroll
                        for (int i = 0; i < SBS2 * QK2; i += ES) {
                            accvs.template select<ES, 1>(v * ES) +=
                                xvs.template select<ES, 1>(i) *
                                yvs.template select<ES, 1>(i);
                        }
                    }
                }

                for (int b = nsb * SBS2 + tid; b < nb; b += GS) {
                    simd<IT, QK2> xv = block_load<IT, QK2>(input + b * QK2);

                    #pragma unroll
                    for (int v = 0; v < VS; ++v) {
                        simd<fp16, QK2> yv = load_qblock2(
                            weight_base + v * nb * BLOCK_SIZE2 + b * BLOCK_SIZE2,
                            scale_base + v * nb * SCALE_SIZE2 + b * SCALE_SIZE2
                        );

                        #pragma unroll
                        for (int i = 0; i < QK2; i += ES) {
                            accvs.template select<ES, 1>(v * ES) +=
                                xv.template select<ES, 1>(i) *
                                yv.template select<ES, 1>(i);
                        }
                    }
                }

                simd<float, VS> accs;
                #pragma unroll
                for(int v = 0; v < VS; ++v) {
                    accs[v] = sycl::ext::intel::esimd::detail::sum<float, IT, ES>(
                        accvs.template select<ES, 1>(v * ES)
                    );
                }

                slm_block_store<float, VS>(tid * VS * sizeof(float), accs);

                barrier();

                if (tid == 0) {
                    #pragma unroll
                    for (int i = 1; i < GS; ++i) {
                        accs += slm_block_load<float, VS>(i * VS * sizeof(float));
                    }

                    block_store<IT, VS>(output + vid, accs * eweight);
                }
            }
        );
    };

    // utils::submit_kernel(cgf, device, "moe forward down kernel");
    dpcpp_queue.submit(cgf);
}