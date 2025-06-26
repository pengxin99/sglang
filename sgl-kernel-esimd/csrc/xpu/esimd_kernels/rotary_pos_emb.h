#include "utils.h"

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using fp16 = sycl::half;
using namespace sycl;

/**
 * 
 *  
    def rotate_half(x):
      """Rotates half the hidden dims of the input."""
      x1 = x[..., : x.shape[-1] // 2]
      x2 = x[..., x.shape[-1] // 2 :]
      return torch.cat((-x2, x1), dim=-1)  

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

    
*/
ESIMD_INLINE void rotary_pos_emb_llama(uint8_t* qState, uint8_t* kState, uint8_t* cos, uint8_t* sin, int num_heads_q, int hidden_dim_q, int num_heads_kv, int hidden_dim_kv, nd_item<2>& ndi) {
  int h = ndi.get_group(1); // [0, q_heads+k_heads)
  int v = ndi.get_group(0); // [0, input_len)


  simd<fp16, 128> input;
  simd<fp16, 128> input_rotate_half;
  simd<fp16, 128> output;
  simd<fp16, 128> cos_value;
  simd<fp16, 128> sin_value;
  unsigned int offsetCosSin = v * hidden_dim_q;

  cos_value.template bit_cast_view<fp16>().template select<128, 1>(0) =
    __ESIMD_ENS::lsc_block_load<
    fp16,
    128,
    __ESIMD_ENS::lsc_data_size::u16,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::uncached>((fp16*)cos + offsetCosSin);

  sin_value.template bit_cast_view<fp16>().template select<128, 1>(0) =
    __ESIMD_ENS::lsc_block_load<
    fp16,
    128,
    __ESIMD_ENS::lsc_data_size::u16,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::uncached>((fp16*)sin + offsetCosSin);

  if (h < num_heads_q) // q
  {
    unsigned int InOffset = h * hidden_dim_q + v * hidden_dim_q * num_heads_q;
    input.template bit_cast_view<fp16>().template select<128, 1>(0) =
      __ESIMD_ENS::lsc_block_load<
      fp16,
      128,
      __ESIMD_ENS::lsc_data_size::u16,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::uncached>((fp16*)qState + InOffset);

    input_rotate_half.select<64, 1>(64) = input.select<64, 1>(0);
    input_rotate_half.select<64, 1>(0) = input.select<64, 1>(64) * -1.0;
      
    #pragma unroll
      for (int k = 0; k < 8; k++) {
        output.select<16, 1>(16 * k) = input.select<16, 1>(16 * k) * cos_value.select<16, 1>(16 * k) 
                                      + input_rotate_half.select<16, 1>(16 * k) * sin_value.select<16, 1>(16 * k);
      }
  

    __ESIMD_ENS::lsc_block_store<
      fp16,
      128,
      __ESIMD_ENS::lsc_data_size::u8,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back
      >((fp16*)qState + InOffset, output.select<128, 1>(0));
  }
  else if (h < num_heads_q + num_heads_kv)  // k
  {
    unsigned int InOffset = (h - num_heads_q) * hidden_dim_kv + v * hidden_dim_kv * num_heads_kv;
    input.template bit_cast_view<fp16>().template select<128, 1>(0) =
      __ESIMD_ENS::lsc_block_load<
      fp16,
      128,
      __ESIMD_ENS::lsc_data_size::u16,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::uncached>((fp16*)kState + InOffset);


    input_rotate_half.select<64, 1>(64) = input.select<64, 1>(0);
    input_rotate_half.select<64, 1>(0) = input.select<64, 1>(64) * -1.0;
      
    #pragma unroll
      for (int k = 0; k < 8; k++) {
        output.select<16, 1>(16 * k) = input.select<16, 1>(16 * k) * cos_value.select<16, 1>(16 * k) 
                                      + input_rotate_half.select<16, 1>(16 * k) * sin_value.select<16, 1>(16 * k);
      }

    __ESIMD_ENS::lsc_block_store<
      fp16,
      128,
      __ESIMD_ENS::lsc_data_size::u8,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back
      >((fp16*)kState + InOffset, output.select<128, 1>(0));
  }
    


}



/*

def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    
    # x: [b, np, sq, hn]
    b, np, sq, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:, :sq]

    xshaped = x.reshape(b, np, sq, rot_dim // 2, 2)
    rope_cache = rope_cache.view(-1, 1, sq, xshaped.size(3), 2)

    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)

    return torch.cat((x_out2, x_pass), dim=-1)

    apply_rotary_pos_emb ================
    x torch.Size([1, 24, 1, 128])
    rope_cache torch.Size([1, 1, 64, 2])
    rope_cache2 torch.Size([1, 1, 64, 2])
    rope_cache view torch.Size([1, 1, 1, 64, 2])
    x_out2 not flat torch.Size([1, 24, 1, 64, 2])
    x_pass torch.Size([1, 24, 1, 0])
    xshaped torch.Size([1, 24, 1, 64, 2])
    x_out2 torch.Size([1, 24, 1, 128])
    apply_rotary_pos_emb ================
    x torch.Size([1, 6, 1, 128])
    rope_cache torch.Size([1, 1, 64, 2])
    rope_cache2 torch.Size([1, 1, 64, 2])
    rope_cache view torch.Size([1, 1, 1, 64, 2])
    x_out2 not flat torch.Size([1, 6, 1, 64, 2])
    x_pass torch.Size([1, 6, 1, 0])
    xshaped torch.Size([1, 6, 1, 64, 2])
    x_out2 torch.Size([1, 6, 1, 128])
    apply_rotary_pos_emb ================

 */

ESIMD_INLINE void rotary_pos_emb(uint8_t* qState, uint8_t* kState, uint8_t* rope_cacheState, int num_heads_q, int hidden_dim_q, int num_heads_kv, int hidden_dim_kv, nd_item<2>& ndi) {
  int h = ndi.get_group(0); // [0, mask_shape_0 // 16)

  simd<fp16, 128> input;
  simd<fp16, 128> input_rotate_half;
  simd<fp16, 128> rope_cache;

  rope_cache.template bit_cast_view<fp16>().template select<128, 1>(0) =
    __ESIMD_ENS::lsc_block_load<
    fp16,
    128,
    __ESIMD_ENS::lsc_data_size::u16,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::uncached>((fp16*)rope_cacheState);

  if (h < num_heads_q) // q
  {
    unsigned int InOffset = h * hidden_dim_q;
    input.template bit_cast_view<fp16>().template select<128, 1>(0) =
      __ESIMD_ENS::lsc_block_load<
      fp16,
      128,
      __ESIMD_ENS::lsc_data_size::u16,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::uncached>((fp16*)qState + InOffset);

    input_rotate_half.select<64, 2>(0) = input.select<64, 2>(0) * rope_cache.select<64, 2>(0) - input.select<64, 2>(1) * rope_cache.select<64, 2>(1);
    input_rotate_half.select<64, 2>(1) = input.select<64, 2>(1) * rope_cache.select<64, 2>(0) + input.select<64, 2>(0) * rope_cache.select<64, 2>(1);

    __ESIMD_ENS::lsc_block_store<
      fp16,
      128,
      __ESIMD_ENS::lsc_data_size::u8,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back
      >((fp16*)qState + InOffset, input_rotate_half.select<128, 1>(0));
  }
  else if (h < num_heads_q + num_heads_kv)  // k
  {
    unsigned int InOffset = (h - num_heads_q) * hidden_dim_kv;
    input.template bit_cast_view<fp16>().template select<128, 1>(0) =
      __ESIMD_ENS::lsc_block_load<
      fp16,
      128,
      __ESIMD_ENS::lsc_data_size::u16,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::uncached>((fp16*)kState + InOffset);

    input_rotate_half.select<64, 2>(0) = input.select<64, 2>(0) * rope_cache.select<64, 2>(0) - input.select<64, 2>(1) * rope_cache.select<64, 2>(1);
    input_rotate_half.select<64, 2>(1) = input.select<64, 2>(1) * rope_cache.select<64, 2>(0) + input.select<64, 2>(0) * rope_cache.select<64, 2>(1);

    __ESIMD_ENS::lsc_block_store<
      fp16,
      128,
      __ESIMD_ENS::lsc_data_size::u8,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back
      >((fp16*)kState + InOffset, input_rotate_half.select<128, 1>(0));
  }
    


}

