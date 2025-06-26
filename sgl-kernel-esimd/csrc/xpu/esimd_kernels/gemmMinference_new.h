#include "utils.h"

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using fp16 = sycl::half;
using namespace sycl;

using tf32 = sycl::ext::intel::experimental::esimd::tfloat32;

#define KVACC 8
#define SDP_THREAD 32

// KVACC fix to 2 for now, since mask read has limitaion
#if (KVACC != 2 && KVACC != 4 && KVACC != 8)
  jslkfjasklfj // compile failure
#endif

#define TYPE_XVE   fp16
// #define TYPE_XVE   float

#define DO_PREFETCH

template<uint32_t Q_HEAD, uint32_t KV_HEAD>
ESIMD_INLINE void SDP_xmx_safe_new_minfer(
  uint8_t* qState,
  uint8_t* kState,
  uint8_t* vState,
  uint8_t* mask,
  uint8_t* debugBuf,
  uint8_t* debugBuf2,
  uint8_t* upperMask,
  uint8_t* downMask,
  uint8_t* out,
  int kvSeqLen,
  int vCacheStride,
  nd_item<3>& ndi) {
  constexpr float matMulQuantCoeff = 0.08838834764831844f; // 1.0f / sqrt(128.0f);
  constexpr uint32_t QHEAD_DIV_KVHEAD = Q_HEAD / KV_HEAD;
  constexpr uint32_t kv_16x128_size = 16*16*8*sizeof(fp16);

  constexpr uint32_t pingSLMoffset = 0;
  constexpr uint32_t pongSLMoffset = (kv_16x128_size * KVACC + kv_16x128_size * KVACC);

  __ESIMD_NS::slm_init(2 * pongSLMoffset); 
  // KQ result; KQmax;  Qload

  // int localLinearId = ndi.get_local_linear_id(); // [0, 32)
  int localLinearId = ndi.get_local_id(2); // [0, 32)
  int h = ndi.get_group(0); // [0, 8)
  int v = ndi.get_group(2); // [0, batch)
  int head_shared = ndi.get_group(1); // [0, 4)
  // int last_v_idx = ndi.get_group_range(1) - 1;
  int hq = h * QHEAD_DIV_KVHEAD + head_shared;
  int h_kv = hq / QHEAD_DIV_KVHEAD;
  int kvSeqOutLoopCount = (kvSeqLen + 16*KVACC - 1) / (16*KVACC);

  int kvSeqRest = kvSeqLen - (kvSeqOutLoopCount - 1) * 16*KVACC;
  // kvSeqLen must be 16 aligned
  int kvGroup_mask_aligned = (kvSeqLen /16 + 15) / 16 * 16;  // Sparse mask must 16 aligned (in unit of 16x16)

  int q_idx_8aligned = (v * SDP_THREAD + localLinearId) * 8;
  bool skipQcal = false;
  if (q_idx_8aligned >= kvSeqLen)
  {
    skipQcal = true;
  }

  unsigned int offsetQ = q_idx_8aligned * Q_HEAD * 128 * sizeof(fp16) + hq * 128 * sizeof(fp16) ;
  unsigned int outputOffset = q_idx_8aligned * Q_HEAD * 128 + hq * 128;

  unsigned int offsetKbase = (h_kv * 128 + localLinearId * 8) * sizeof(fp16);   // read 16x8 K

  int localLinearId_minous_16 = localLinearId - 16;
  int idxVline = localLinearId_minous_16 % 2;
  int idxVCol = localLinearId_minous_16 / 2;
  bool is_first16 = localLinearId < 16;

  unsigned int offsetVBase = (h_kv * 128 * vCacheStride + idxVCol * 16 * vCacheStride + idxVline * 8) * sizeof(fp16);

  simd<uint32_t, 16> offsetK;
  simd<uint32_t, 16> offsetV;
  simd<uint32_t, 16> offsetMask;
#pragma unroll
  for (int k = 0; k < 16; k++) {
    offsetV[k] = k;
    offsetK[k] = k;
    offsetMask[k] = k;
  }
  offsetK = offsetK * 128 * KV_HEAD * sizeof(fp16) + offsetKbase;
  offsetV = offsetV * vCacheStride * sizeof(fp16) + offsetVBase;
  offsetMask = offsetMask * kvGroup_mask_aligned * sizeof(int8_t) 
    + hq * kvGroup_mask_aligned * kvSeqLen / 16 * sizeof(int8_t) 
    + v * SDP_THREAD * 8 / 16 * kvGroup_mask_aligned * sizeof(int8_t);  
  // v * SDP_THREAD * 8 is q_idx_8aligned start

  simd<fp16, 128 * 8> qq;
  simd<fp16, 128 * 8> qq_before_shuffle;
  simd<TYPE_XVE, 128 * 8> kvCacheOutFP32;
  simd<fp16, 128 * 8> output;

  simd<fp16, 16 * 8> kv_read;

  simd<fp16, 16 * 8 * 16> kk;    // 16*8 x 16
  simd<fp16, 8*16 * 8 * 2> vv;  // 8 x  16*8 x2
  simd<fp16, 16 * 8> kq_out;
  simd<TYPE_XVE, 16 * 8> softMax;
  simd<TYPE_XVE, 16 * 8> softMaxSumTemp = 0.0000000001;

  simd<fp16, 16 * 8 * 2> kq_mask_up;
  simd<fp16, 16 * 8 * 2> kq_mask_down;

  simd<int8_t, 16 * KVACC> sparse_mask{-1};
  bool read_flag[KVACC] = {false};

#pragma unroll
  for (int ll = 0; ll < 8; ll++)
  {
    qq_before_shuffle.template bit_cast_view<unsigned char>().template select<256, 1>(256*ll) =
          __ESIMD_ENS::lsc_block_load<
          unsigned char,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::uncached,
          __ESIMD_ENS::cache_hint::uncached>((unsigned char*)qState + offsetQ + ll * Q_HEAD * 128 * sizeof(fp16));
  }

#pragma unroll
  for (int ll = 0; ll < 8; ll++) { // 8 of 8x128
    #pragma unroll
    for (int kk = 0; kk < 8; kk++) {  // 128/16 = 8
      qq.select<16, 1>(kk*16*8 + ll*16) = qq_before_shuffle.select<16, 1>(ll*128 + kk*16);
    }
  }
  // qq = qq_before_shuffle;

#pragma unroll
  for (int ll = 0; ll < 2 * 8; ll++) {
    kq_mask_up.template bit_cast_view<unsigned char>().template select<32, 1>(ll * 32) =
      __ESIMD_ENS::lsc_block_load<
      unsigned char,
      32,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached>((unsigned char*)upperMask + (ll * 16) * sizeof(fp16));
  }

#pragma unroll
  for (int ll = 0; ll < 2 * 8; ll++) {
    kq_mask_down.template bit_cast_view<unsigned char>().template select<32, 1>(ll * 32) =
      __ESIMD_ENS::lsc_block_load<
      unsigned char,
      32,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached>((unsigned char*)downMask + (ll * 16) * sizeof(fp16));
  }

  // if (hq == 13)
  // {
  //   int offset = q_idx_8aligned * 128;

  //   for (int i = 0; i < 128*8; i++) { // 8 of 8x128
  //     ((fp16*)debugBuf)[offset + i] = qq[i];
  //   }
  //   // ((fp16*)debugBuf)[0] = qq[0];
  // }
  simd<int8_t, 16*KVACC> sparse_mask_sum = -1;
  // simd<int32_t, 4> sparse_mask_group = -1;
  simd<uint64_t, 2> sparse_mask_group = 1;

  simd<TYPE_XVE, 8*1> maxKq = -65504.0;//-65504.0;// -10000;
  simd<TYPE_XVE, 8*1> old_maxKq = -65504.0;//-65504.0;// -10000;
  simd<TYPE_XVE, 8*1> max_correction = -65504.0;//-65504.0;// -10000;
  kvCacheOutFP32 = 0;

  bool isPingSLM = true;
  uint32_t slmOffset = pingSLMoffset;
  for (int loopIdx = 0; loopIdx < kvSeqOutLoopCount; loopIdx++) {
    bool fully_skip = true;
    bool is_upper_right = ((v+1) * SDP_THREAD * 8 < (loopIdx + 1) * KVACC * 16);  // >= is not upper right include slash

    if (!is_upper_right)  // only not upper right need to check if fully_skip
    {

#if KVACC == 2
      sparse_mask.template bit_cast_view<uint16_t>().template select<16, 1>(0) =
      __ESIMD_ENS::lsc_gather<
      uint16_t,
      1,
      __ESIMD_ENS::lsc_data_size::u16,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached,
      16,
      uint32_t
      >((uint16_t*)mask, offsetMask + loopIdx * KVACC * sizeof(int8_t));
      // loopIdx * KVACC / 16 is kv idx
#elif KVACC == 4
      sparse_mask.template bit_cast_view<uint32_t>().template select<16, 1>(0) =
      __ESIMD_ENS::lsc_gather<
      uint32_t,
      1,
      __ESIMD_ENS::lsc_data_size::u32,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached,
      16,
      uint32_t
      >((uint32_t*)mask, offsetMask + loopIdx * KVACC * sizeof(int8_t));
      // loopIdx * KVACC / 16 is kv idx
#elif KVACC == 8
      sparse_mask.template bit_cast_view<uint64_t>().template select<16, 1>(0) =
      __ESIMD_ENS::lsc_gather<
      uint64_t,
      1,
      __ESIMD_ENS::lsc_data_size::u64,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached,
      16,
      uint64_t
      >((uint64_t*)mask, offsetMask + loopIdx * KVACC * sizeof(int8_t));
      // loopIdx * KVACC / 16 is kv idx
#endif

      sparse_mask_sum = sparse_mask + 1;
      #pragma unroll
      for (int kvac = 0; kvac < KVACC; kvac++) {
        int kv_idx_16 = loopIdx * KVACC + kvac;
        if (kv_idx_16 * 16 < kvSeqLen)
        {
          sparse_mask_group.template bit_cast_view<int8_t>().template select<16, 1>(0) = sparse_mask_sum.select<16, KVACC>(1*kvac);

          // if (sparse_mask_group[0] == 0 && sparse_mask_group[1] == 0 && sparse_mask_group[2] == 0 && sparse_mask_group[3] == 0)
          if (sparse_mask_group[0] == 0 && sparse_mask_group[1] == 0)
          {
            read_flag[kvac] = false;
          }
          else
          {
            read_flag[kvac] = true;
            fully_skip = false;
          }
        }
      }
    }

    if (!fully_skip)
    {

      if (isPingSLM)
      {
        slmOffset = pingSLMoffset;
      }
      else
      {
        slmOffset = pongSLMoffset;
      }
      isPingSLM = !isPingSLM;

      #pragma unroll
      for (int kvac = 0; kvac < KVACC; kvac++) {
        int kv_idx_16 = loopIdx * KVACC + kvac;
        if (read_flag[kvac] && kv_idx_16 * 16 < kvSeqLen)
        {
          int index_of_16x8_update = localLinearId;
          int offset_is_k_v = 0;
          if (is_first16)
          {
            kv_read.template bit_cast_view<uint32_t>().template select<64, 1>(0) =
            __ESIMD_ENS::lsc_gather<
            uint32_t,
            4,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            16,
            uint32_t
            >((uint32_t*)kState, offsetK + kv_idx_16 * 16 * 128 * KV_HEAD * sizeof(fp16));
          }
          else
          {
            kv_read.template bit_cast_view<uint32_t>().template select<64, 1>(0) =
            __ESIMD_ENS::lsc_gather<
            uint32_t,
            4,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            16,
            uint32_t
            >((uint32_t*)vState, offsetV + kv_idx_16 * 16 * sizeof(fp16));
            offset_is_k_v = kv_16x128_size * KVACC;
            index_of_16x8_update = localLinearId_minous_16;
          }

          slm_block_store<fp16, 128>(slmOffset + offset_is_k_v + kv_16x128_size * kvac + 16*8*index_of_16x8_update*sizeof(fp16), kv_read);
          
        }
      } // for (int kvac = 0; kvac < KVACC; kvac++)

      barrier();

      #pragma unroll
      for (int kvac = 0; kvac < KVACC; kvac++) {
        int kv_idx_16 = loopIdx * KVACC + kvac;
        int8_t cur_sparse_mask = sparse_mask[localLinearId/2*KVACC+kvac];
        if (cur_sparse_mask != -1 && kv_idx_16 * 16 < kvSeqLen && !skipQcal)
        {
          
#ifdef DO_PREFETCH
          // prefetch
          int kv_idx_16_pre = kv_idx_16 + KVACC;
          {
            int offset_is_k_v = 0;
            if (is_first16)
            {
              __ESIMD_ENS::lsc_prefetch<
              uint32_t,
              4,
              __ESIMD_ENS::lsc_data_size::u32,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached,
              16,
              uint32_t
              >((uint32_t*)kState, offsetK + kv_idx_16_pre * 16 * 128 * KV_HEAD * sizeof(fp16));
            }
            else
            {
              __ESIMD_ENS::lsc_prefetch<
              uint32_t,
              4,
              __ESIMD_ENS::lsc_data_size::u32,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached,
              16,
              uint32_t
              >((uint32_t*)vState, offsetV + kv_idx_16_pre * 16 * sizeof(fp16));
            }
          }
#endif

          // simd<fp16, 16 * 8 * 16> kk;    // 16*8 x 16
          // simd<fp16, 8*16 * 8 * 2> vv;  // 8 x  16*8 x2
          kk = slm_block_load<fp16, 16 * 8 * 16>(slmOffset + kv_16x128_size * kvac);

          // K*Q ------------------------
          
          simd<sycl::half, 8 * 16> cc_xmx{0};
          #pragma unroll
          for (int ww = 0; ww < 128/16; ww++) {

              simd<sycl::half, 8 * 16> bb_xmx{0};
              simd<sycl::half, 16 * 16> aa_xmx{0};

              bb_xmx= qq.select<8*16,1>(ww * 8*16);
              aa_xmx = kk.select<16*16, 1>(ww * 16*16);
              cc_xmx = xmx::dpas<8, 8, sycl::half, sycl::half, sycl::half, sycl::half>(cc_xmx, aa_xmx, bb_xmx);
          }
          kq_out.select<16*8,1>(0)=cc_xmx.select<16*8,1>(0);

          // soft Max ------------------------
          softMax = kq_out;
          softMax = softMax * matMulQuantCoeff;

          if (cur_sparse_mask == 2)
          {
            softMax = softMax + kq_mask_down.select<16*8, 1>((localLinearId%2)*16*8);
          }
          else if (cur_sparse_mask == 1)
          {
            softMax = softMax + kq_mask_up.select<16*8, 1>((localLinearId%2)*16*8);
          }
          // cur_sparse_mask == 0   no mask add needed.

          old_maxKq = maxKq;
          #pragma unroll
          for (int ll = 0; ll < 16; ll++) {
            maxKq = max<TYPE_XVE, 8, TYPE_XVE>(maxKq, softMax.select<8, 16>(ll));
          }
          maxKq = max<TYPE_XVE, 8, TYPE_XVE>(maxKq, old_maxKq);   // compare w/ old

          #pragma unroll
          for (int ll = 0; ll < 8; ll++) {
            softMax.select<16, 1>(ll *16) = softMax.select<16, 1>(ll *16) - maxKq[ll];
            softMax.select<16, 1>(ll *16) = pow<TYPE_XVE, 16, TYPE_XVE>(2.718f, softMax.select<16, 1>(ll *16));
          }

          // Correct max val
          if (kv_idx_16 >= 1)
          {
            max_correction = old_maxKq - maxKq;
            max_correction = pow<TYPE_XVE, 8, TYPE_XVE>(2.718f, max_correction);

            #pragma unroll
            for (int ll = 0; ll < 8; ll++) {    
              #pragma unroll
              for (int mm = 0; mm < 16; mm++) {
                kvCacheOutFP32.select<16, 1>(mm * 8*16 + ll * 16) = kvCacheOutFP32.select<16, 1>(mm * 8*16 + ll * 16) * max_correction[ll];
              }
              softMaxSumTemp.select<16, 1>(ll * 16) = softMaxSumTemp.select<16, 1>(ll * 16) * max_correction[ll];
            }
          }
          softMaxSumTemp.select<8*16, 1>(0) += softMax.select<8*16, 1>(0);
          
          vv = slm_block_load<fp16, 16 * 8 * 16>(slmOffset + kv_16x128_size * KVACC + kv_16x128_size * kvac);

          // if (hq == 0 && q_idx_8aligned == 0)
          // {
          //   vv.select<16*16, 1>(0) = 0;
          //   softMax.select<8*16,1>(0) = 0;
          // }
          
          // score*V ------------------------
          #pragma unroll
          for (int nc = 0; nc < 128/16; nc++) { // loop on non-common 128
              simd<TYPE_XVE, 8 * 16> cc_xmx{0};
              simd<sycl::half, 8 * 16> bb_xmx{0};
              simd<sycl::half, 16 * 16> aa_xmx{0};

              cc_xmx.select<16*8,1>(0) = kvCacheOutFP32.select<16*8,1>(nc*16*8);

              bb_xmx= softMax.select<8*16,1>(0);
              aa_xmx = vv.select<16*16, 1>(nc * 16*16);
              cc_xmx = xmx::dpas<8, 8, TYPE_XVE, TYPE_XVE, sycl::half, sycl::half>(cc_xmx, aa_xmx, bb_xmx);

              kvCacheOutFP32.select<16*8,1>(nc*16*8)=cc_xmx.select<16*8,1>(0);
          }

          // if (hq == 0 && q_idx_8aligned == 0)
          // {
          //     ((float*)debugBuf)[kv_idx_16] = maxKq[1];
          //     ((float*)debugBuf2)[kv_idx_16] = kvCacheOutFP32[3 + 16*1];
          // }

        } // if (kv_idx_16 * 16 < kvSeqLen)
      } // for (int kvac = 0; kvac < KVACC; kvac++)
    } // if (!fully_skip)
  } // for (int loopIdx = 0; loopIdx < kvSeqOutLoopCount; loopIdx++)

  if (skipQcal) return;

  #pragma unroll
  for (int qq = 0; qq < 8; qq++) {
    softMaxSumTemp.select<8, 1>(qq * 16) = softMaxSumTemp.select<8, 1>(qq * 16) + softMaxSumTemp.select<8, 1>(qq * 16 + 8);
    softMaxSumTemp.select<4, 1>(qq * 16) = softMaxSumTemp.select<4, 1>(qq * 16) + softMaxSumTemp.select<4, 1>(qq * 16 + 4);
    softMaxSumTemp.select<2, 1>(qq * 16) = softMaxSumTemp.select<2, 1>(qq * 16) + softMaxSumTemp.select<2, 1>(qq * 16 + 2);
    softMaxSumTemp[qq * 16] = softMaxSumTemp[qq * 16] + softMaxSumTemp[qq * 16 + 1];
    softMaxSumTemp[qq * 16] = 1.0f / softMaxSumTemp[qq * 16];

    #pragma unroll
    for (int ll = 0; ll < 16; ll++) {
      kvCacheOutFP32.select<16, 1>(ll * 8*16 + qq * 16) = kvCacheOutFP32.select<16, 1>(ll * 8*16 + qq * 16) * softMaxSumTemp[qq * 16];
    }
  }

#pragma unroll
  for (int ll = 0; ll < 8; ll++) { // 8 of 8x128
    #pragma unroll
    for (int kk = 0; kk < 8; kk++) {  // 128/16 = 8
      output.select<16, 1>(ll*128 + kk*16) = kvCacheOutFP32.select<16, 1>(kk*16*8 + ll*16);
    }
  }

#pragma unroll
  for (int ll = 0; ll < 8; ll++) { // 8 of 8x128
  __ESIMD_ENS::lsc_block_store<
    fp16,
    128,
    __ESIMD_ENS::lsc_data_size::default_size,
    __ESIMD_ENS::cache_hint::write_back,
    __ESIMD_ENS::cache_hint::write_back>((fp16*)out + outputOffset + ll * Q_HEAD * 128, output.select<128, 1>(ll*128));
  }
}
