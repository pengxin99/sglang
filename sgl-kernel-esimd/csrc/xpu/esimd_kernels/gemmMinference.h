#include "utils.h"

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using fp16 = sycl::half;
using namespace sycl;


template<uint32_t HEADS_Q, uint32_t HEADS_KV>
ESIMD_INLINE void gemmKQV_DenseIn(uint8_t* a, uint8_t* b, uint8_t* c, int outputRow, int hiddenDim, int tokenSize, nd_item<2>& ndi) {
  constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  constexpr uint32_t baseOffsetInc8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
  constexpr uint32_t baseOffsetInc4[4] = { 0, 1, 2, 3 };

  // Weight(a) is V,  input(b) is KQ_s
  // Assume V non-common is 128(outputRow)   KQ non common is (token len or input len)
  // Common dim is  REDUCE_K_DIM_DIV_32 * 32
  constexpr uint32_t loopCount = 8;  // 512 / 8T = 64  64 / 8 = 8
  constexpr uint32_t loopCountW = 2;  // 128 / 4T = 32  32 / 16 = 2
  constexpr uint32_t HEADS_Q_DIV_HEADS_KV = HEADS_Q/HEADS_KV;
  int hh = ndi.get_local_linear_id(); // [0, 64)
  int h = ndi.get_group(0); // [0, (row + 15) / 16)
  int hiddenDimFP16Size = hiddenDim * sizeof(fp16);
  uint32_t REDUCE_K_DIM_DIV_32 = hiddenDim >> 5; // hiddenDim / 32
  simd<fp16, 8 * 16 * 2 * 8> cc;
  simd<fp16, 8 * 32 * 8> bb;
  simd<fp16, 32 * 16 * 2> aa;
  simd<fp16, 32 * 16> aat;
  
  // group total num is groups_per_head * heads
  // 32 * ((m + 511) / 512) * ((n + 127) / 128)
  int groups_per_head = ndi.get_group_range(0) / HEADS_Q;   
  int h_perhead = h % groups_per_head;

  int head_q_Idx = h / groups_per_head;
  int head_kv_Idx = head_q_Idx / HEADS_Q_DIV_HEADS_KV;

  int aligned_tokenLength = (tokenSize + 7) / 8 * 8;

  // WG:
  // 8192 / 512 = 16    128 / 128 = 1 
  int tokenBlkCnt = (tokenSize + 511) / 512;
  int h_i = h_perhead % tokenBlkCnt;   // 16
  int h_w = h_perhead / tokenBlkCnt;   // 1

  int hi = h_i * 64/* 512/8 */ + (hh >> 2) * 8;   // 8t for input

  int hh_4 = (hh & 0x3);

  if (hi * 8 >= tokenSize) return;

  int hw222 = h_w * 8/* 128/16 */ + hh_4 * 2 + 0;  // 4t for weight

  // handle weight non-common dim not 256 aligned, assume 128 aligned
  // outputRow == 128

  cc = 0;

  uint32_t globalOffset = head_kv_Idx * hiddenDim * outputRow + hw222 * hiddenDim * 16;
  uint32_t baseOffsetA = globalOffset * sizeof(fp16);

  simd<uint32_t, 16> offset(baseOffsetInc16);

  offset = offset * hiddenDimFP16Size + baseOffsetA;

  uint32_t baseOffsetB = head_q_Idx * hiddenDim * aligned_tokenLength *sizeof(fp16) + hi * 8*32*sizeof(fp16)*REDUCE_K_DIM_DIV_32;

for (int ww = 0; ww < REDUCE_K_DIM_DIV_32 /*2048/32*/ ; ww++) 
{

  
{

  {
    aa.template bit_cast_view<uint32_t>().template select<128, 1>(0) =
      __ESIMD_ENS::lsc_gather<
      uint32_t,
      8,
      __ESIMD_ENS::lsc_data_size::u32,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached,
      16,
      uint32_t
      >((uint32_t*)a, offset + ww * 32 * sizeof(fp16));
    aa.template bit_cast_view<uint32_t>().template select<128, 1>(128) =
      __ESIMD_ENS::lsc_gather<
      uint32_t,
      8,
      __ESIMD_ENS::lsc_data_size::u32,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached,
      16,
      uint32_t
      >((uint32_t*)a, offset + ww * 32 * sizeof(fp16) + 16 * sizeof(fp16));

    aa.template bit_cast_view<uint32_t>().template select<128, 1>(256) =
      __ESIMD_ENS::lsc_gather<
      uint32_t,
      8,
      __ESIMD_ENS::lsc_data_size::u32,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached,
      16,
      uint32_t
      >((uint32_t*)a, offset + hiddenDimFP16Size * 16 + ww * 32 * sizeof(fp16));

    aa.template bit_cast_view<uint32_t>().template select<128, 1>(384) =
      __ESIMD_ENS::lsc_gather<
      uint32_t,
      8,
      __ESIMD_ENS::lsc_data_size::u32,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached,
      16,
      uint32_t
      >((uint32_t*)a, offset + hiddenDimFP16Size * 16 + ww * 32 * sizeof(fp16) + 16 * sizeof(fp16));

  }

} // www

// barrier();

#pragma unroll
for (int nn = 0; nn < loopCount; nn++)
{

{
    {
      //read 256 fp16 =8*8*4.
      bb.template bit_cast_view<uint8_t>().template select<256, 1>(/*www*8*32*2 +*/nn*8*32*2 + 0) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + baseOffsetB + nn * 256*sizeof(fp16)*REDUCE_K_DIM_DIV_32 + ww * 256 * sizeof(fp16));
      bb.template bit_cast_view<uint8_t>().template select<256, 1>(/*www*8*32*2 +*/nn*8*32*2 + 256) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + baseOffsetB + 256 + nn * 256*sizeof(fp16)*REDUCE_K_DIM_DIV_32 + ww * 256 * sizeof(fp16));
    }
} // www
} // loop i


#pragma unroll
for (int nn = 0; nn < loopCount; nn++)
{
  
#pragma unroll
for (int nw = 0; nw < loopCountW; nw++)
{
      aat.select<512, 1>(0) = aa.select<512, 1>(nw * 512);

{
    {
        
      simd<sycl::half, 8 * 16> bb_tmp{0};
      simd<sycl::half, 8 * 16> bb_xmx{0};
      simd<sycl::half, 16 * 16> aa_xmx{0};
      simd<sycl::half, 8 * 16> cc_xmx{0};

      cc_xmx.select<16*8,1>(0)=cc.select<16*8,1>((nw*8 + nn) * 16*8);
      //bb_xmx=bb_tmp.select<8*16,1>(0);
      bb_xmx=bb.select<8*16,1>(/*www*8*32*/nn*8*32 +0);
  #pragma unroll
      for(int t=0;t<16;t++)//t<N/2
      {
        aa_xmx.template select<16,1>(16 * t)=aat.template  select<16,1>(16*t);
      }
  #ifdef XMX_USED
      cc_xmx = xmx::dpas<8, 8, sycl::half, sycl::half, sycl::half, sycl::half>(cc_xmx, aa_xmx, bb_xmx);
  #endif

      bb_xmx=bb.select<8*16,1>(/*www*8*32*/nn*8*32 + 8*16);
  #pragma unroll
      for(int t=0;t<16;t++)//t<N/2
      {
        aa_xmx.template select<16,1>(16 * t)=aat.template  select<16,1>(16*t + 8*16*2*1);
      }
  #ifdef XMX_USED
      cc_xmx = xmx::dpas<8, 8, sycl::half, sycl::half, sycl::half, sycl::half>(cc_xmx, aa_xmx, bb_xmx);
  #endif

      cc.select<16*8,1>((nw*8 + nn) * 16*8)=cc_xmx.select<16*8,1>(0);

      }
    
} // www


} // loop w
} // loop i

} // loop common w

#pragma unroll
for (int nn = 0; nn < loopCount; nn++)
{
if (hi * 8 + nn * 8 + 7 < tokenSize)
{
#pragma unroll
for (int nw = 0; nw < loopCountW; nw++)
{
    int hw = h_w * 8/* 128/16 */ + hh_4 * 2 + nw;  // 4t for weight
    // handle weight non-common dim not 256 aligned, assume 128 aligned
    // outputRow == 128
    uint32_t offsetC = head_q_Idx * outputRow * tokenSize + hw * 16 + nn * 8 * outputRow + hi * 8 * outputRow;  // hi include 

  #pragma unroll
     for (int k = 0; k < 8; k++) {
      __ESIMD_ENS::lsc_block_store<
        fp16,
        16,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::write_back,
        __ESIMD_ENS::cache_hint::write_back>((fp16*)c + offsetC + k * outputRow, cc.select<16,1>((nw*8 + nn) * 16*8 + 16 * k));
     }
} // loop w
}
else
{
#pragma unroll
for (int nw = 0; nw < loopCountW; nw++)
{
    int hw = h_w * 8/* 128/16 */ + hh_4 * 2 + nw;  // 4t for weight
    // handle weight non-common dim not 256 aligned, assume 128 aligned
    // outputRow == 128
    uint32_t offsetC = head_q_Idx * outputRow * tokenSize + hw * 16 + nn * 8 * outputRow + hi * 8 * outputRow;  // hi include 

  #pragma unroll
     for (int k = 0; k < 8; k++) {
      if (hi * 8 + nn * 8 + k < tokenSize) {
      __ESIMD_ENS::lsc_block_store<
        fp16,
        16,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::write_back,
        __ESIMD_ENS::cache_hint::write_back>((fp16*)c + offsetC + k * outputRow, cc.select<16,1>((nw*8 + nn) * 16*8 + 16 * k));
      }
     }
} // loop w
}
} // loop i

}

template<uint32_t HEADS_Q>
ESIMD_INLINE void fp16ShuffleToFp16_xmx_no_k_split_ref_multiheads(uint8_t* a, uint8_t* b, uint32_t hiddenDim, uint32_t tokenLength, nd_item<2>& ndi) {
  constexpr uint32_t baseOffsetInc8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
  int h = ndi.get_group(0); // [0, n // 32)  and heads

  // group total num is groups_per_head * heads
  // 32 * ((m + 511) / 512) * ((n + 127) / 128)
  int aligned_tokenLength = (tokenLength + 7) / 8 * 8;

  int head_q_Idx = ndi.get_local_id(0);

  int v = ndi.get_group(1); // [0, aligned(k, 8) // 8)
  uint32_t baseOffset = head_q_Idx * hiddenDim * tokenLength * sizeof(fp16) + 
    (v * 8 * hiddenDim + h * 32) * sizeof(fp16);
  int offsetOut = head_q_Idx * hiddenDim * aligned_tokenLength + 
    (v * hiddenDim / 32 + h) * 8 * 32;

  simd<uint32_t, 8> offsetIn(baseOffsetInc8);
  simd<fp16, 256> fp16Input;
  simd<fp16, 256> fp16Output;
  offsetIn = offsetIn * hiddenDim * sizeof(fp16) + baseOffset;

  simd_mask<8> quantPred = 1;
  if (v*8 + 7 >= tokenLength)
  {
    quantPred = 0;
    int mask = tokenLength - v*8;
    for (int t = 0; t < mask; t++) {
      quantPred[t] = 1;
    }
  }

#pragma unroll
  for (int k = 0; k < 4 * 8; k++) {
    fp16Input.template bit_cast_view<fp16>().template select<8, 1>(8 * k) =
      __ESIMD_ENS::lsc_gather<
      fp16,
      1,
      __ESIMD_ENS::lsc_data_size::u16,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached,
      8,
      uint32_t
      >((fp16*)a, offsetIn, quantPred);
    offsetIn += sizeof(fp16);
  }

// 2x8x16
#pragma unroll
  for (int k = 0; k < 2; k++) {
#pragma unroll
    for (int kk = 0; kk < 8; kk++) {
        fp16Output.select<16, 1>(k*128 + kk*16) = fp16Input.select<16, 8>(k*128 + kk);
    }
  }

// // 8x32
// #pragma unroll
// for (int kk = 0; kk < 8; kk++) {
//     fp16Output.select<32, 1>(kk*32) = fp16Input.select<32, 8>(kk);
// }

#pragma unroll
  for (int k = 0; k < 2; k++) {
    __ESIMD_ENS::lsc_block_store<
      fp16,
      128,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((fp16*)b + offsetOut + 128 * k, fp16Output.select<128, 1>(128 * k));
  }
}

using tf32 = sycl::ext::intel::experimental::esimd::tfloat32;

template<uint32_t Q_HEAD, uint32_t KV_HEAD, uint32_t Q_THIN, int sparseFlag>
ESIMD_INLINE void SDP_Fusion128To2048KvlengthLoopFp32QFp16KvXveSimd16Slm_headQ_K_V_thin_xmx_safe_Temp(
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
  int q_reduce_left,
  nd_item<3>& ndi) {
  constexpr float matMulQuantCoeff = 0.08838834764831844f; // 1.0f / sqrt(128.0f);
  constexpr uint32_t kv_count_in_loop = 16;
  constexpr uint32_t qk_softmax_slm_size = kv_count_in_loop * 16 * 2 * 8 * sizeof(float);
  constexpr uint32_t qk_max_tmp_slm_size = kv_count_in_loop * 2 * 8 * sizeof(float);
  constexpr uint32_t q_slm_size = 128 * 16 * sizeof(fp16);
  constexpr uint32_t QHEAD_DIV_KVHEAD = Q_HEAD / KV_HEAD;
  constexpr uint32_t slmQLoadOffset = qk_softmax_slm_size + qk_max_tmp_slm_size;
  constexpr uint32_t slmKQMaxLoadOffset = qk_softmax_slm_size;
  constexpr uint32_t Q_THIN_DIV8 = (Q_THIN + 7) / 8;
  constexpr uint32_t Q_REST = Q_THIN - (Q_THIN_DIV8-1) * 8;
  //constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  __ESIMD_NS::slm_init((qk_softmax_slm_size + qk_max_tmp_slm_size + q_slm_size)); 
  // KQ result; KQmax;  Qload

  // int localLinearId = ndi.get_local_linear_id(); // [0, 32)
  int localLinearId = ndi.get_local_id(0); // [0, 32)
  //int localLinearId_origkq = localLinearId >> 1; // localLinearId / 1;
  // int hh = localLinearId & 0x3; // [0, 4)
  // int vv = localLinearId >> 2;  // [0, 8)
  int h = ndi.get_group(0); // [0, 8)
  int v = ndi.get_group(1); // [0, batch)
  int head_4 = ndi.get_group(2); // [0, 4)
  // int head_4 = ndi.get_local_id(2); // [0, 4)
  int last_v_idx = ndi.get_group_range(1) - 1;
  int hq = h * 4 + head_4;
  int h_kv = hq / QHEAD_DIV_KVHEAD;
  int kvSeqOutLoopCount = (kvSeqLen + 0xff) >> 8; // seqLen / 16*16
  int kvSeqRest = kvSeqLen - (kvSeqOutLoopCount - 1) * 16*16;
  int kvSeqLen32_aligned = (kvSeqLen + 31) / 32 * 32;
  int kvGroup_mask_aligned = (kvSeqLen/16 + 15) / 16 * 16;
  simd<fp16, 128> qqFp16In;
  simd<fp16, 8 * 16 * 2> qqFp16;
  simd<fp16, 8 * 16 > qqFp16Tmp;
  //simd<fp16, 128> qq;
  simd<fp16, 16 * 16 * 2> kk;
  // simd<fp16, 16 * 16 * 2> kk_1;
  simd<fp16, 16 * 16 * 2> vvState;
  simd<float, 16 * 8> kvCacheOut = 0;
  simd<float, 16 * 8> kvCacheOutFP32 = 0;
  simd<float, 16 * 8> softMaxSumTemp = 0.0000000001;
  simd<float, 32 * 16> softMaxCache;
  simd<float, 32 * 16> softMaxCacheFP32;
  simd<float, 16 * 8 * 2> softMax;
  float softMaxPadding = 0;
  simd<fp16, 16 * 8 * 2> output = 0;
  
  int localLinearId_Qchunck = localLinearId >> 3; // 32-> 16 / 8 = 2;
  int localLinearId_Vchunck = localLinearId & 0x7; // 32-> 16  % 8

  unsigned int outputOffset = v * 16 * Q_HEAD * 128 + hq * 128 + localLinearId_Vchunck * 16 + localLinearId_Qchunck * 8 * 128 * Q_HEAD;
  unsigned int outputVOffsetSlm = localLinearId * 128 * sizeof(fp16);
  unsigned int outputSoftmaxOffsetSlm = 0;
  unsigned int offsetQ = v * 16 * Q_HEAD * 128 * sizeof(fp16) + hq * 128 * sizeof(fp16);
  unsigned int offsetK_base = (h_kv * 128 + localLinearId * 128 * KV_HEAD * 16) * sizeof(fp16);
  unsigned int offsetVBase = (h_kv * vCacheStride * 128 + localLinearId_Vchunck * 16 * vCacheStride) * sizeof(fp16);
  simd<uint32_t, 16> offsetK;
  simd<uint32_t, 16> offsetV;
  for (int k = 0; k < 16; k++) {
    offsetV[k] = k;
    offsetK[k] = k;
  }
  offsetV = offsetV * vCacheStride * sizeof(fp16) + offsetVBase;
  offsetK = offsetK * 128 * KV_HEAD * sizeof(fp16) + offsetK_base;
  int kvSeqOffset = localLinearId * 16;
  
  if (v == last_v_idx && q_reduce_left != 0)
  {
    if (localLinearId < q_reduce_left)
    {
      int k = localLinearId;
      qqFp16In.template bit_cast_view<unsigned char>().template select<256, 1>(0) =
        __ESIMD_ENS::lsc_block_load<
        unsigned char,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::uncached,
        __ESIMD_ENS::cache_hint::uncached>((unsigned char*)qState + offsetQ + k * 128 * Q_HEAD * sizeof(fp16));

      qqFp16Tmp = qqFp16In;
        
      slm_block_store<fp16, 128>(slmQLoadOffset + k * 128 * sizeof(fp16), qqFp16Tmp);
    }
    else if (localLinearId < 16)
    {
      int k = localLinearId;
      qqFp16In = 0;
      qqFp16Tmp = qqFp16In;
        
      slm_block_store<fp16, 128>(slmQLoadOffset + k * 128 * sizeof(fp16), qqFp16Tmp);
    }
  }
  else
  {
    if (localLinearId < Q_THIN)
    {
      int k = localLinearId;
      qqFp16In.template bit_cast_view<unsigned char>().template select<256, 1>(0) =
        __ESIMD_ENS::lsc_block_load<
        unsigned char,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::uncached,
        __ESIMD_ENS::cache_hint::uncached>((unsigned char*)qState + offsetQ + k * 128 * Q_HEAD * sizeof(fp16));

      qqFp16Tmp = qqFp16In;
        
      slm_block_store<fp16, 128>(slmQLoadOffset + k * 128 * sizeof(fp16), qqFp16Tmp);
    }
    else if (localLinearId < 16)
    {
      int k = localLinearId;
      qqFp16In = 0;
      qqFp16Tmp = qqFp16In;
        
      slm_block_store<fp16, 128>(slmQLoadOffset + k * 128 * sizeof(fp16), qqFp16Tmp);
    }
  }


  // if (localLinearId == 0)
  // {
  //   qqFp32.select<16, 1>(0) = 0.001f;
      
  //   slm_block_store<float, 16>(slmQLoadOffset, qqFp32.select<16, 1>(0));
  // }

  barrier();

  
  simd<fp16, 16 * 8 * 2> kq_mask = -65504.0;
  simd<int8_t, 32> sparse_mask = -1;
  simd<int8_t, 32> sparse_mask_sum = -1;
  simd<int32_t, 4> sparse_mask_group = -1;

  simd<float, 8> lastkvCacheOutFP32_Max = -INFINITY;
  simd<float, 8> curkvCacheOutFP32_Max = -INFINITY;
  
  simd<float, 8> lastsoftMaxCacheFP32_Max1 = -INFINITY;
  simd<float, 8> lastsoftMaxCacheFP32_Max2 = -INFINITY;
  simd<float, 8> lastsoftMaxCacheFP32_MaxReal = -INFINITY;

  // for (int kk = 0; kk < 16; kk++) {
  //   qStateSimd[kk] = qState[0]
  // }
  // qStateSimd.template bit_cast_view<unsigned char>().template select<64, 1>(0) =
  //     __ESIMD_ENS::lsc_block_load<
  //     unsigned char,
  //     64,
  //     __ESIMD_ENS::lsc_data_size::default_size,
  //     __ESIMD_ENS::cache_hint::cached,
  //     __ESIMD_ENS::cache_hint::cached>((unsigned char*)qState);

  for (int loopIdx = 0; loopIdx < kvSeqOutLoopCount; loopIdx++) {

    bool not_upper_triangle = loopIdx * 256 < (v+1) * 16;
    // bool not_upper_triangle = true;
    bool is_skip_window = !not_upper_triangle;

    // if (sparseFlag == 999 && loopIdx > 0 && (loopIdx + 1) * 256 < v * 16 - 512)
    // {
    //   is_skip_window = true;
    // }
    if (!is_skip_window)
    {
      sparse_mask.template bit_cast_view<unsigned char>().template select<16, 1>(0) =
      __ESIMD_ENS::lsc_block_load<
      unsigned char,
      16,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached>((unsigned char*)mask + 
          (hq * kvGroup_mask_aligned * kvSeqLen / 16 + ( v * kvGroup_mask_aligned + loopIdx * 16) * sizeof(int8_t)));

      // sparse_mask_sum = sparse_mask;
      // sparse_mask_sum.select<8, 1>(0) += sparse_mask_sum.select<8, 1>(8);
      // sparse_mask_sum.select<4, 1>(0) += sparse_mask_sum.select<4, 1>(8);
      // sparse_mask_sum.select<2, 1>(0) += sparse_mask_sum.select<2, 1>(8);
      // sparse_mask_sum[0] += sparse_mask_sum[1];

      // if (sparse_mask_sum[0] == -16)
      // {
      //   is_skip_window = true;
      // }
      sparse_mask_sum = sparse_mask + 1;
      sparse_mask_group.template bit_cast_view<int8_t>().template select<16, 1>(0) = sparse_mask_sum.select<16, 1>(0);

      if (sparse_mask_group[0] == 0 && sparse_mask_group[1] == 0 && sparse_mask_group[2] == 0 && sparse_mask_group[3] == 0)
      {
        is_skip_window = true;
      }
    }
    if (!is_skip_window)
    {
// #pragma unroll
//       for (size_t inner_loop_of_thread = 0; inner_loop_of_thread < 16 / 16; inner_loop_of_thread++)
//       {

    constexpr int inner_loop_of_thread = 0;
    simd<float, 8*2> maxKq = -65504.0;//-65504.0;// -10000;

    if (kvSeqOffset < kvSeqLen) {
      output = 0;
      
    if (sparseFlag == 999)
    {
      // no heads

        // sparse_mask.template bit_cast_view<unsigned char>().template select<16, 1>(0) =
        //   __ESIMD_ENS::lsc_block_load<
        //   unsigned char,
        //   16,
        //   __ESIMD_ENS::lsc_data_size::default_size,
        //   __ESIMD_ENS::cache_hint::cached,
        //   __ESIMD_ENS::cache_hint::cached>((unsigned char*)mask + (( v * kvGroup_mask_aligned + loopIdx * 16) * sizeof(int8_t)));
// sparse_mask = 0;

      // multi-heads
        // sparse_mask.template bit_cast_view<unsigned char>().template select<16, 1>(0) =
        //   __ESIMD_ENS::lsc_block_load<
        //   unsigned char,
        //   16,
        //   __ESIMD_ENS::lsc_data_size::default_size,
        //   __ESIMD_ENS::cache_hint::cached,
        //   __ESIMD_ENS::cache_hint::cached>((unsigned char*)mask + 
        //       (hq * kvGroup_mask_aligned * kvSeqLen / 16 + ( v * kvGroup_mask_aligned + loopIdx * 16) * sizeof(int8_t)));

        if (sparse_mask[localLinearId] == 0) // full
        {
           kq_mask = 0;
        }
        else if (sparse_mask[localLinearId] == 1) // upper triangle
        {
  #pragma unroll
            for (int ll = 0; ll < 2 * 8; ll++) {
              kq_mask.template bit_cast_view<unsigned char>().template select<32, 1>(ll * 32) =
                __ESIMD_ENS::lsc_block_load<
                unsigned char,
                32,
                __ESIMD_ENS::lsc_data_size::default_size,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached>((unsigned char*)upperMask + (ll * 16) * sizeof(fp16));
            }

        }
        else if (sparse_mask[localLinearId] == 2) // down triangle
        {
 #pragma unroll
            for (int ll = 0; ll < 2 * 8; ll++) {
              kq_mask.template bit_cast_view<unsigned char>().template select<32, 1>(ll * 32) =
                __ESIMD_ENS::lsc_block_load<
                unsigned char,
                32,
                __ESIMD_ENS::lsc_data_size::default_size,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached>((unsigned char*)downMask + (ll * 16) * sizeof(fp16));
            }          
        }
    }
    else
    {
#pragma unroll
        for (int ll = 0; ll < 2 * 8; ll++) {
          kq_mask.template bit_cast_view<unsigned char>().template select<32, 1>(ll * 32) =
            __ESIMD_ENS::lsc_block_load<
            unsigned char,
            32,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((unsigned char*)mask + ((v * 16 + ll) * vCacheStride + kvSeqOffset) * sizeof(fp16));
        }  // vCacheStride is aligned kv_len in ipex

        if (loopIdx == kvSeqOutLoopCount - 1 && kvSeqOffset + 16 >= kvSeqLen)
        {
          int remain = kvSeqLen - kvSeqOffset;
          for (int ll = remain; ll < 16; ll++) {
            kq_mask.select<16, 16>(ll) = -65504.0;
          }
        }
    }

    if (sparse_mask[localLinearId] != -1 || sparseFlag != 999)
    {

#pragma unroll
    for (int ww = 0; ww < 128/32; ww++) {

      kk.template bit_cast_view<uint32_t>().template select<128, 1>(0) =
        __ESIMD_ENS::lsc_gather<
        uint32_t,
        8,
        __ESIMD_ENS::lsc_data_size::u32,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached,
        16,
        uint32_t
        >((uint32_t*)kState, offsetK + 128 * KV_HEAD * 16 * inner_loop_of_thread * sizeof(fp16) + ww * 32 * sizeof(fp16));

      kk.template bit_cast_view<uint32_t>().template select<128, 1>(128) =
        __ESIMD_ENS::lsc_gather<
        uint32_t,
        8,
        __ESIMD_ENS::lsc_data_size::u32,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached,
        16,
        uint32_t
        >((uint32_t*)kState, offsetK + 128 * KV_HEAD * 16 * inner_loop_of_thread * sizeof(fp16) + (ww * 32 + 16) * sizeof(fp16));
  
#pragma unroll
      for (int qq = 0; qq < Q_THIN_DIV8; qq++) {
#pragma unroll
        for (int ll = 0; ll < 8; ll++) {
          qqFp16.select<16, 1>(ll * 16) = 
            slm_block_load<fp16, 16>(slmQLoadOffset + qq * 128 * 8 * sizeof(fp16) + ll * 128 * sizeof(fp16) + ww * 32 * sizeof(fp16));
          // qqFp32.select<16, 1>(0) = slm_block_load<float, 16>(slmQLoadOffset);
          // qqFp16.select<16, 1>(ll * 16) = qqFp32.select<16, 1>(0);
            //slm_block_load<fp16, 16>(slmQLoadOffset);
        }
#pragma unroll
        for (int ll = 0; ll < 8; ll++) {
          qqFp16.select<16, 1>(ll * 16 + 128) = 
            slm_block_load<fp16, 16>(slmQLoadOffset + qq * 128 * 8 * sizeof(fp16) + ll * 128 * sizeof(fp16) + (ww * 32 + 16) * sizeof(fp16));
          // qqFp32.select<16, 1>(0) = slm_block_load<float, 16>(slmQLoadOffset);
          // qqFp16.select<16, 1>(ll * 16 + 128) = qqFp32.select<16, 1>(0);
            //slm_block_load<fp16, 16>(slmQLoadOffset);
        }        

        //qqFp16 = 0.001f;
        {  
                simd<sycl::half, 8 * 16> bb_xmx{0};
                simd<sycl::half, 16 * 16> aa_xmx{0};
                simd<sycl::half, 8 * 16> cc_xmx{0};

                cc_xmx.select<16*8,1>(0)=output.select<16*8,1>(qq*16*8);
                bb_xmx= qqFp16.select<8*16,1>(0);
                aa_xmx = kk.select<256, 1>(0);

            //#ifdef XMX_USED
                cc_xmx = xmx::dpas<8, 8, sycl::half, sycl::half, sycl::half, sycl::half>(cc_xmx, aa_xmx, bb_xmx);
            //#endif

                bb_xmx= qqFp16.select<8*16,1>(8*16);
                aa_xmx = kk.select<256, 1>(256);
            //#ifdef XMX_USED
                cc_xmx = xmx::dpas<8, 8, sycl::half, sycl::half, sycl::half, sycl::half>(cc_xmx, aa_xmx, bb_xmx);
            //#endif

                output.select<16*8,1>(qq*16*8)=cc_xmx.select<16*8,1>(0);
        }

      }
      
    } // ww for k

      softMax = output;
      softMax = softMax * matMulQuantCoeff;
       softMax = softMax + kq_mask;


    #pragma unroll
        for (int ll = 0; ll < 16; ll++) {
          maxKq = max<float, 16, float>(maxKq, softMax.select<16, 16>(ll)); //1.0f; for dbg  ((float)loopIdx) * 2.0f;
        }
    //   maxKq[0] = softMax[0]; 
    // maxKq = 90.1875;
      
    #pragma unroll
        for (int ll = 0; ll < 8*2; ll++) {
          softMax.select<16, 1>(ll *16) = softMax.select<16, 1>(ll *16) - maxKq[ll];

          softMax.select<16, 1>(ll *16) = pow<float, 16, float>(2.718f, softMax.select<16, 1>(ll *16));
        }

      } 
      else // sparse cal skip =============================================
      {
        softMax = 0;
      }

    } else {
      softMax = 0;
      // if (localLinearId == 0 || localLinearId == 1 || localLinearId == 2)
      // maxKq = matMulQuantCoeff; 
    }

    // if (localLinearId == 0 || localLinearId == 1 || localLinearId == 2)
    // {
    // maxKq[0] = matMulQuantCoeff;
    // }
    // maxKq[1] = matMulQuantCoeff;
    // maxKq[2] = matMulQuantCoeff; 
    // maxKq[3] = matMulQuantCoeff; 
    // maxKq[4] = matMulQuantCoeff; 
    // maxKq[5] = matMulQuantCoeff; 
    // maxKq[6] = matMulQuantCoeff; 
    // maxKq[7] = matMulQuantCoeff; 
    // maxKq[8] = matMulQuantCoeff;
    // maxKq[9] = matMulQuantCoeff; 
    // maxKq[10] = matMulQuantCoeff; 
    // maxKq[11] = matMulQuantCoeff; 
    // maxKq[12] = matMulQuantCoeff; 
    // maxKq[13] = matMulQuantCoeff; 
    // maxKq[14] = matMulQuantCoeff; 
    //maxKq[15] = matMulQuantCoeff; 

    // for (int iii = 0; iii < 16; iii++)
    // {
    //   int head = (h * 4 + head_4);
    //   int token_offset = v * 16;
    //   int offset = head * 32544 * 256 / 16  * 4 + (token_offset + iii) * 256 / 16 * 4 + 256 / 16 * loopIdx + localLinearId * 16 / 16;
    //   ((float*)(debugBuf))[offset] = maxKq[iii];
    // }


    slm_block_store<float, 128>(outputSoftmaxOffsetSlm + (localLinearId) * 16 * 2 * 8 * sizeof(float), softMax.select<128, 1>(0));
    slm_block_store<float, 128>(outputSoftmaxOffsetSlm + (localLinearId) * 16 * 2 * 8 * sizeof(float) + 128 * sizeof(float), softMax.select<128, 1>(128));
    
    // slm_block_store<float, 8>(slmKQMaxLoadOffset + localLinearId * 2 * 8 * sizeof(float), maxKq.select<8, 1>(0));
    // slm_block_store<float, 8>(slmKQMaxLoadOffset + localLinearId * 2 * 8 * sizeof(float) + 8 * sizeof(float), maxKq.select<8, 1>(8));
    slm_block_store<float, 16>(slmKQMaxLoadOffset + (localLinearId) * 2 * 8 * sizeof(float), maxKq);
      // } // inner_loop_of_thread
// if (localLinearId == 0 || localLinearId == 1 || localLinearId == 2) {
//         __ESIMD_ENS::lsc_block_store<
//           float,
//           32,
//           __ESIMD_ENS::lsc_data_size::default_size,
//           __ESIMD_ENS::cache_hint::write_back,
//           __ESIMD_ENS::cache_hint::write_back>((float*)out + 
//           outputSoftmaxOffsetSlm + localLinearId * 16 * 2 * 8, softMax.select<32, 1>(0));
//                   __ESIMD_ENS::lsc_block_store<
//           float,
//           32,
//           __ESIMD_ENS::lsc_data_size::default_size,
//           __ESIMD_ENS::cache_hint::write_back,
//           __ESIMD_ENS::cache_hint::write_back>((float*)out + 
//           outputSoftmaxOffsetSlm + localLinearId * 16 * 2 * 8 + 32, softMax.select<32, 1>(32));
//                   __ESIMD_ENS::lsc_block_store<
//           float,
//           32,
//           __ESIMD_ENS::lsc_data_size::default_size,
//           __ESIMD_ENS::cache_hint::write_back,
//           __ESIMD_ENS::cache_hint::write_back>((float*)out + 
//           outputSoftmaxOffsetSlm + localLinearId * 16 * 2 * 8 + 64, softMax.select<32, 1>(64));
//                   __ESIMD_ENS::lsc_block_store<
//           float,
//           32,
//           __ESIMD_ENS::lsc_data_size::default_size,
//           __ESIMD_ENS::cache_hint::write_back,
//           __ESIMD_ENS::cache_hint::write_back>((float*)out + 
//           outputSoftmaxOffsetSlm + localLinearId * 16 * 2 * 8 + 96, softMax.select<32, 1>(96));

//         __ESIMD_ENS::lsc_block_store<
//           float,
//           32,
//           __ESIMD_ENS::lsc_data_size::default_size,
//           __ESIMD_ENS::cache_hint::write_back,
//           __ESIMD_ENS::cache_hint::write_back>((float*)out + 
//           outputSoftmaxOffsetSlm + localLinearId * 16 * 2 * 8 + 128, softMax.select<32, 1>(128));

//           __ESIMD_ENS::lsc_block_store<
//           float,
//           32,
//           __ESIMD_ENS::lsc_data_size::default_size,
//           __ESIMD_ENS::cache_hint::write_back,
//           __ESIMD_ENS::cache_hint::write_back>((float*)out + 
//           outputSoftmaxOffsetSlm + localLinearId * 16 * 2 * 8 + 128 + 32, softMax.select<32, 1>(128+32));

//           __ESIMD_ENS::lsc_block_store<
//           float,
//           32,
//           __ESIMD_ENS::lsc_data_size::default_size,
//           __ESIMD_ENS::cache_hint::write_back,
//           __ESIMD_ENS::cache_hint::write_back>((float*)out + 
//           outputSoftmaxOffsetSlm + localLinearId * 16 * 2 * 8 + 128 + 64, softMax.select<32, 1>(128+64));

//           __ESIMD_ENS::lsc_block_store<
//           float,
//           32,
//           __ESIMD_ENS::lsc_data_size::default_size,
//           __ESIMD_ENS::cache_hint::write_back,
//           __ESIMD_ENS::cache_hint::write_back>((float*)out + 
//           outputSoftmaxOffsetSlm + localLinearId * 16 * 2 * 8 + 128 + 96, softMax.select<32, 1>(128+96));


//         __ESIMD_ENS::lsc_block_store<
//           float,
//           16,
//           __ESIMD_ENS::lsc_data_size::default_size,
//           __ESIMD_ENS::cache_hint::write_back,
//           __ESIMD_ENS::cache_hint::write_back>((float*)out + 768 + localLinearId * 2 * 8, maxKq);

//         __ESIMD_ENS::lsc_block_store<
//           float,
//           32,
//           __ESIMD_ENS::lsc_data_size::default_size,
//           __ESIMD_ENS::cache_hint::write_back,
//           __ESIMD_ENS::cache_hint::write_back>((float*)out + 1024 + localLinearId * 2 * 8 * 16, kq_mask.select<32, 1>(0));
//                 __ESIMD_ENS::lsc_block_store<
//           float,
//           32,
//           __ESIMD_ENS::lsc_data_size::default_size,
//           __ESIMD_ENS::cache_hint::write_back,
//           __ESIMD_ENS::cache_hint::write_back>((float*)out + 1024 + localLinearId * 2 * 8 * 16 + 32, kq_mask.select<32, 1>(32));
//                   __ESIMD_ENS::lsc_block_store<
//           float,
//           32,
//           __ESIMD_ENS::lsc_data_size::default_size,
//           __ESIMD_ENS::cache_hint::write_back,
//           __ESIMD_ENS::cache_hint::write_back>((float*)out + 1024 + localLinearId * 2 * 8 * 16 + 64, kq_mask.select<32, 1>(64));
//                   __ESIMD_ENS::lsc_block_store<
//           float,
//           32,
//           __ESIMD_ENS::lsc_data_size::default_size,
//           __ESIMD_ENS::cache_hint::write_back,
//           __ESIMD_ENS::cache_hint::write_back>((float*)out + 1024 + localLinearId * 2 * 8 * 16 + 96, kq_mask.select<32, 1>(96));
//                   __ESIMD_ENS::lsc_block_store<
//           float,
//           32,
//           __ESIMD_ENS::lsc_data_size::default_size,
//           __ESIMD_ENS::cache_hint::write_back,
//           __ESIMD_ENS::cache_hint::write_back>((float*)out + 1024 + localLinearId * 2 * 8 * 16 + 128, kq_mask.select<32, 1>(128));
//                   __ESIMD_ENS::lsc_block_store<
//           float,
//           32,
//           __ESIMD_ENS::lsc_data_size::default_size,
//           __ESIMD_ENS::cache_hint::write_back,
//           __ESIMD_ENS::cache_hint::write_back>((float*)out + 1024 + localLinearId * 2 * 8 * 16 + 160, kq_mask.select<32, 1>(160));
//                   __ESIMD_ENS::lsc_block_store<
//           float,
//           32,
//           __ESIMD_ENS::lsc_data_size::default_size,
//           __ESIMD_ENS::cache_hint::write_back,
//           __ESIMD_ENS::cache_hint::write_back>((float*)out + 1024 + localLinearId * 2 * 8 * 16 + 192, kq_mask.select<32, 1>(192));
//                   __ESIMD_ENS::lsc_block_store<
//           float,
//           32,
//           __ESIMD_ENS::lsc_data_size::default_size,
//           __ESIMD_ENS::cache_hint::write_back,
//           __ESIMD_ENS::cache_hint::write_back>((float*)out + 1024 + localLinearId * 2 * 8 * 16 + 224, kq_mask.select<32, 1>(224));
// }



    barrier();

    // 8 thread for v16x8=128 and 2 thraed for kq8x2=16
    if (localLinearId < Q_THIN_DIV8 * 8)
    {

      for (int ww = 0; ww < 16*16/32; ww++) {  // 16 x (2x16)
        // bool need_to_skip = ww * 32 > v * 16;
        // bool need_to_skip = ww * 32 + 1 > v * 16;
        if ((sparse_mask[2*ww] != -1 || sparse_mask[2*ww + 1] != -1) || sparseFlag != 999)
        {

        if (loopIdx < kvSeqOutLoopCount - 1 || ww*32 < kvSeqRest) {
        
        vvState.template bit_cast_view<uint32_t>().template select<128, 1>(0) =
            __ESIMD_ENS::lsc_gather<
            uint32_t,
            8,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            16,
            uint32_t
            >((uint32_t*)vState, offsetV + ww * 32 * sizeof(fp16));
        vvState.template bit_cast_view<uint32_t>().template select<128, 1>(128) =
            __ESIMD_ENS::lsc_gather<
            uint32_t,
            8,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            16,
            uint32_t
            >((uint32_t*)vState, offsetV + (ww * 32 + 16) * sizeof(fp16));

        // if (ww*32 + 16 >= kvSeqRest)
        // {
        //   simd<fp16, 16 * 16 * 2> vvState2;
        //   //vvState.select<256, 1>(256) = 0;
        //   vvState2 = vvState;
        //   vvState = 0;
        //   //vvState.select<16, 2>(0) = vvState2.select<16, 2>(0);
        // }


        int slmSoftMaxLoadOffset = outputSoftmaxOffsetSlm;
        softMaxCacheFP32.select<128, 1>(0) = slm_block_load<float, 128>(outputSoftmaxOffsetSlm + ww * 2 * 16 * 2 * 8 * sizeof(float) + 128 * localLinearId_Qchunck * sizeof(float));
        if (loopIdx < kvSeqOutLoopCount - 1 || ww*32 + 16 < kvSeqRest)
        {
          softMaxCacheFP32.select<128, 1>(128) = slm_block_load<float, 128>(outputSoftmaxOffsetSlm + (ww * 2 + 1) * 16 * 2 * 8 * sizeof(float) + 128 * localLinearId_Qchunck * sizeof(float));
        }
        else
        {
          softMaxCacheFP32.select<128, 1>(128) = 0;
        }

        lastsoftMaxCacheFP32_Max1.select<8, 1>(0) = slm_block_load<float, 8>(slmKQMaxLoadOffset + ww * 2 * 2 * 8 * sizeof(float) + 8 * localLinearId_Qchunck * sizeof(float));
        lastsoftMaxCacheFP32_Max2.select<8, 1>(0) = slm_block_load<float, 8>(slmKQMaxLoadOffset + (ww * 2 + 1) * 2 * 8 * sizeof(float) + 8 * localLinearId_Qchunck * sizeof(float));

        if (loopIdx == kvSeqOutLoopCount - 1 && ww*32 + 16 >= kvSeqRest)
        {
          int remain = kvSeqRest - ww*32;
          for (int ll = remain; ll < 16; ll++) {
              softMaxCacheFP32.select<8, 16>(ll) = 0;
            }
          for (int ll = 0; ll < 16; ll++) {
              softMaxCacheFP32.select<8, 16>(ll + 128) = 0;
            }
        }
        else if (loopIdx == kvSeqOutLoopCount - 1 && ww*32 + 32 >= kvSeqRest)
        {
          int remain = kvSeqRest - ww*32 - 16;
          for (int ll = remain; ll < 16; ll++) {
              softMaxCacheFP32.select<8, 16>(ll + 128) = 0;
            }
        }

        // dbg
        //  softMaxCacheFP32 = 1.0f;
        //  lastsoftMaxCacheFP32_Max1 = 1.0f;
        //  lastsoftMaxCacheFP32_Max2 = 1.0f;
        //lastkvCacheOutFP32_Max = 1.0f;
        //lastsoftMaxCacheFP32_MaxReal = 1.0f;
        // lastsoftMaxCacheFP32_Max1 = ((float)loopIdx) * 2.0f;
        // lastsoftMaxCacheFP32_Max2 = ((float)loopIdx) * 2.0f;
        
        // handle softMaxCacheFP32 max -----------------------------------------
        lastsoftMaxCacheFP32_MaxReal = max<float, 8, float>(lastsoftMaxCacheFP32_Max1, lastsoftMaxCacheFP32_Max2);

        if (loopIdx == 0 && ww == 0)
        {
          lastkvCacheOutFP32_Max = lastsoftMaxCacheFP32_MaxReal;
        }

        lastsoftMaxCacheFP32_MaxReal = max<float, 8, float>(lastkvCacheOutFP32_Max, lastsoftMaxCacheFP32_MaxReal);

        // lastsoftMaxCacheFP32_MaxReal = 91.0f;
    // if (loopIdx != 0 || ww != 0)
    {
        lastsoftMaxCacheFP32_Max1 = lastsoftMaxCacheFP32_Max1 - lastsoftMaxCacheFP32_MaxReal;
        lastsoftMaxCacheFP32_Max2 = lastsoftMaxCacheFP32_Max2 - lastsoftMaxCacheFP32_MaxReal;

        lastsoftMaxCacheFP32_Max1 = pow<float, 8, float>(2.718f, lastsoftMaxCacheFP32_Max1);
        lastsoftMaxCacheFP32_Max2 = pow<float, 8, float>(2.718f, lastsoftMaxCacheFP32_Max2);

        // softMaxCacheFP32.select<128, 1>(0) * e^oldmax * e^-newmax
        // softMaxCacheFP32.select<128, 1>(128) * e^oldmax * e^-newmax


    #pragma unroll
        for (int ll = 0; ll < 8; ll++) {
          softMaxCacheFP32.select<16, 1>(ll * 16) = softMaxCacheFP32.select<16, 1>(ll * 16) * lastsoftMaxCacheFP32_Max1[ll];
        }
    #pragma unroll
        for (int ll = 0; ll < 8; ll++) {
          softMaxCacheFP32.select<16, 1>(128 + ll * 16) = softMaxCacheFP32.select<16, 1>(128 + ll * 16)  * lastsoftMaxCacheFP32_Max2[ll];
        }
    }

         softMaxCache = softMaxCacheFP32;

        // handle kvCacheOutFP32 max and softMaxSumTemp -----------------------------------------

        // if (loopIdx != 0 || ww != 0)
        {
          curkvCacheOutFP32_Max = lastkvCacheOutFP32_Max - lastsoftMaxCacheFP32_MaxReal;
          curkvCacheOutFP32_Max = pow<float, 8, float>(2.718f, curkvCacheOutFP32_Max);

          kvCacheOutFP32 = kvCacheOut;

          // kvCacheOutFP32 * e^oldmax * e^-newmax
      #pragma unroll
          for (int ll = 0; ll < 8; ll++) {
            kvCacheOutFP32.select<16, 1>(ll * 16) = kvCacheOutFP32.select<16, 1>(ll * 16) * curkvCacheOutFP32_Max[ll];
            softMaxSumTemp.select<16, 1>(ll * 16) = softMaxSumTemp.select<16, 1>(ll * 16) * curkvCacheOutFP32_Max[ll];
          }

          kvCacheOut = kvCacheOutFP32;
        }

       lastkvCacheOutFP32_Max = lastsoftMaxCacheFP32_MaxReal;

      //  int head = (h * 4 + head_4);
      //  int token_offset = v * 16;
      //  if (head == 0 && token_offset == 0*16 && localLinearId_Qchunck == 0)
      //  {
      //    for (int iii = 0; iii < 8; iii++)
      //    {
           
      //      int offset = (iii) * 32544 / 32 + loopIdx * 256 / 32 + ww;
      //      ((float*)(debugBuf2))[offset] = lastsoftMaxCacheFP32_MaxReal[iii];
      //    }
      //  }

       bool kqv_xmx_fp16 = true;
      //  if (v < 2)
      //  {
      //     kqv_xmx_fp16 = false;
      //  }

      // bool kqv_xmx_fp16 = false;
       
       if (kqv_xmx_fp16)
        {
                // simd<float, 8> softMaxCache_MaxValue{1};
                // simd<float, 32 * 16> softMaxCacheNormal;
                // #pragma unroll
                //   for (int ll = 0; ll < 16; ll++) {
                //     softMaxCache_MaxValue = max<float, 8, float>(softMaxCache_MaxValue, softMaxCache.select<8, 16>(ll)); 
                //   }
                // #pragma unroll
                //   for (int ll = 0; ll < 16; ll++) {
                //     softMaxCache_MaxValue = max<float, 8, float>(softMaxCache_MaxValue, softMaxCache.select<8, 16>(ll + 8*16)); 
                //   }
                // #pragma unroll
                //   for (int ll = 0; ll < 16; ll++) {
                //     softMaxCacheNormal.select<8, 16>(ll) = softMaxCache.select<8, 16>(ll) / softMaxCache_MaxValue; 
                //   }
                // #pragma unroll
                //   for (int ll = 0; ll < 16; ll++) {
                //     softMaxCacheNormal.select<8, 16>(ll + 8*16) = softMaxCache.select<8, 16>(ll + 8*16) / softMaxCache_MaxValue;
                //   }

                  simd<sycl::half, 8 * 16> bb_xmx{0};
                  simd<sycl::half, 16 * 16> aa_xmx{0};
                  simd<float, 8 * 16> cc_xmx{0};

                  // cc_xmx.select<16*8,1>(0)=kvCacheOut.select<16*8,1>(0);
                  // bb_xmx= softMaxCacheNormal.select<8*16,1>(0);
                  bb_xmx= softMaxCache.select<8*16,1>(0);
                  aa_xmx = vvState.select<256, 1>(0);

              //#ifdef XMX_USED
                  cc_xmx = xmx::dpas<8, 8, float, float, sycl::half, sycl::half>(cc_xmx, aa_xmx, bb_xmx);
              //#endif

                  // bb_xmx= softMaxCacheNormal.select<8*16,1>(8*16);
                  bb_xmx= softMaxCache.select<8*16,1>(8*16);
                  aa_xmx = vvState.select<256, 1>(256);
              //#ifdef XMX_USED
                  cc_xmx = xmx::dpas<8, 8, float, float, sycl::half, sycl::half>(cc_xmx, aa_xmx, bb_xmx);
              //#endif

              // #pragma unroll
              //     for (int ll = 0; ll < 16; ll++) {
              //       cc_xmx.select<8, 16>(ll) = cc_xmx.select<8, 16>(ll) * softMaxCache_MaxValue;
              //     }

                  kvCacheOut.select<16*8,1>(0) += cc_xmx.select<16*8,1>(0);
          }

          else
          {  
                  simd<tf32, 8 * 8> bb_xmx{0};
                  simd<tf32, 16 * 8> aa_xmx{0};
                  simd<float, 8 * 16> cc_xmx{0};
                  
                  // softMaxCache = 1;
                  // vvState = 1;

        #pragma unroll
                  for (int ll = 0; ll < 8; ll++) {  // loop on softMaxCache non-common dim
                    bb_xmx.select<8, 1>(8 * ll) = softMaxCache.select<8, 1>(ll*16);
                  }
        #pragma unroll
                  for (int ll = 0; ll < 8; ll++) {
                    simd<fp16, 32> shuffleTemp;
                    shuffleTemp = vvState.select<32, 1>(32 * ll);
                    vvState.select<16, 1>(32 * ll) = shuffleTemp.select<16, 2>(0);
                    vvState.select<16, 1>(32 * ll + 16) = shuffleTemp.select<16, 2>(1);
                  }
        #pragma unroll
                  for (int ll = 0; ll < 8; ll++) {  // loop on vvState common dim
                    aa_xmx.select<16, 1>(16 * ll) = vvState.select<16, 1>(ll*16);
                  }
                  cc_xmx = xmx::dpas<8, 8, float, float, tf32, tf32>(cc_xmx, aa_xmx, bb_xmx);

        // ====================


        #pragma unroll
                  for (int ll = 0; ll < 8; ll++) {  // loop on softMaxCache non-common dim
                    bb_xmx.select<8, 1>(8 * ll) = softMaxCache.select<8, 1>(ll*16 + 8);
                  }
        #pragma unroll
                  for (int ll = 0; ll < 8; ll++) {  // loop on vvState common dim
                    aa_xmx.select<16, 1>(16 * ll) = vvState.select<16, 1>((8 + ll)*16);
                  }
                  cc_xmx = xmx::dpas<8, 8, float, float, tf32, tf32>(cc_xmx, aa_xmx, bb_xmx);

        // ====================


        #pragma unroll
                  for (int ll = 0; ll < 8; ll++) {  // loop on softMaxCache non-common dim
                    bb_xmx.select<8, 1>(8 * ll) = softMaxCache.select<8, 1>(8*16 + ll*16);
                  }
        #pragma unroll
                  for (int ll = 0; ll < 8; ll++) {
                    simd<fp16, 32> shuffleTemp;
                    shuffleTemp = vvState.select<32, 1>(256 + 32 * ll);
                    vvState.select<16, 1>(256 + 32 * ll) = shuffleTemp.select<16, 2>(0);
                    vvState.select<16, 1>(256 + 32 * ll + 16) = shuffleTemp.select<16, 2>(1);
                  }
        #pragma unroll
                  for (int ll = 0; ll < 8; ll++) {  // loop on vvState common dim
                    aa_xmx.select<16, 1>(16 * ll) = vvState.select<16, 1>(256 + ll*16);
                  }
                  cc_xmx = xmx::dpas<8, 8, float, float, tf32, tf32>(cc_xmx, aa_xmx, bb_xmx);

        // ====================


        #pragma unroll
                  for (int ll = 0; ll < 8; ll++) {  // loop on softMaxCache non-common dim
                    bb_xmx.select<8, 1>(8 * ll) = softMaxCache.select<8, 1>(8*16 + ll*16 + 8);
                  }
        #pragma unroll
                  for (int ll = 0; ll < 8; ll++) {  // loop on vvState common dim
                    aa_xmx.select<16, 1>(16 * ll) = vvState.select<16, 1>(256 + (8 + ll)*16);
                  }
                  cc_xmx = xmx::dpas<8, 8, float, float, tf32, tf32>(cc_xmx, aa_xmx, bb_xmx);
        // ====================

                  kvCacheOut.select<16*8,1>(0) += cc_xmx.select<16*8,1>(0);
          }

          softMaxSumTemp.select<128, 1>(0) += softMaxCache.select<128, 1>(0);
          softMaxSumTemp.select<128, 1>(0) += softMaxCache.select<128, 1>(128);
          
        }
      
        } // end sparse check ===================================
      
      } // ww for end
    }


    barrier();
    }

    offsetK += 16 * 16 * 128 * KV_HEAD * sizeof(fp16);
    offsetV += 16 * 16 * sizeof(fp16);
    kvSeqOffset += 16 * 16;
//   simd<float, 16 * 8> INFFFF = -INFINITY;

// INFFFF = INFFFF + 20;

// for (int ll = 0; ll < 8; ll++){
//   kvCacheOutFP32.select<16, 1>(ll *16) = pow<float, 16, float>(2.718f, INFFFF.select<16, 1>(ll *16));
// }
  }

  if (localLinearId < Q_THIN_DIV8 * 8)
  {
    kvCacheOutFP32 = kvCacheOut;

    if (v == last_v_idx && q_reduce_left != 0)
    {
      if (localLinearId_Qchunck < Q_THIN_DIV8 - 1)
      {
        int loopqCnt = q_reduce_left <= 8 ? q_reduce_left : 8;
        for (int qq = 0; qq < loopqCnt; qq++) {
          softMaxSumTemp.select<8, 1>(qq * 16) = softMaxSumTemp.select<8, 1>(qq * 16) + softMaxSumTemp.select<8, 1>(qq * 16 + 8);
          softMaxSumTemp.select<4, 1>(qq * 16) = softMaxSumTemp.select<4, 1>(qq * 16) + softMaxSumTemp.select<4, 1>(qq * 16 + 4);
          softMaxSumTemp.select<2, 1>(qq * 16) = softMaxSumTemp.select<2, 1>(qq * 16) + softMaxSumTemp.select<2, 1>(qq * 16 + 2);
          softMaxSumTemp[qq * 16] = softMaxSumTemp[qq * 16] + softMaxSumTemp[qq * 16 + 1];
          softMaxSumTemp[qq * 16] = 1.0f / softMaxSumTemp[qq * 16];
          kvCacheOutFP32.select<16, 1>(qq * 16) = kvCacheOutFP32.select<16, 1>(qq * 16) * softMaxSumTemp[qq * 16];
          __ESIMD_ENS::lsc_block_store<
            fp16,
            16,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::write_back,
            __ESIMD_ENS::cache_hint::write_back>((fp16*)out + outputOffset + qq * 128 * Q_HEAD, kvCacheOutFP32.select<16, 1>(qq * 16));
        }
      }
      else
      {
        int loopqCnt = q_reduce_left > 8 ? q_reduce_left-8 : 0;
        for (int qq = 0; qq < loopqCnt; qq++) {
          softMaxSumTemp.select<8, 1>(qq * 16) = softMaxSumTemp.select<8, 1>(qq * 16) + softMaxSumTemp.select<8, 1>(qq * 16 + 8);
          softMaxSumTemp.select<4, 1>(qq * 16) = softMaxSumTemp.select<4, 1>(qq * 16) + softMaxSumTemp.select<4, 1>(qq * 16 + 4);
          softMaxSumTemp.select<2, 1>(qq * 16) = softMaxSumTemp.select<2, 1>(qq * 16) + softMaxSumTemp.select<2, 1>(qq * 16 + 2);
          softMaxSumTemp[qq * 16] = softMaxSumTemp[qq * 16] + softMaxSumTemp[qq * 16 + 1];
          softMaxSumTemp[qq * 16] = 1.0f / softMaxSumTemp[qq * 16];
          kvCacheOutFP32.select<16, 1>(qq * 16) = kvCacheOutFP32.select<16, 1>(qq * 16) * softMaxSumTemp[qq * 16];
          __ESIMD_ENS::lsc_block_store<
            fp16,
            16,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::write_back,
            __ESIMD_ENS::cache_hint::write_back>((fp16*)out + outputOffset + qq * 128 * Q_HEAD, kvCacheOutFP32.select<16, 1>(qq * 16));
        }
      }
    }
    else
    {
      if (localLinearId_Qchunck < Q_THIN_DIV8 - 1)
      {
    #pragma unroll
        for (int qq = 0; qq < 8; qq++) {
          softMaxSumTemp.select<8, 1>(qq * 16) = softMaxSumTemp.select<8, 1>(qq * 16) + softMaxSumTemp.select<8, 1>(qq * 16 + 8);
          softMaxSumTemp.select<4, 1>(qq * 16) = softMaxSumTemp.select<4, 1>(qq * 16) + softMaxSumTemp.select<4, 1>(qq * 16 + 4);
          softMaxSumTemp.select<2, 1>(qq * 16) = softMaxSumTemp.select<2, 1>(qq * 16) + softMaxSumTemp.select<2, 1>(qq * 16 + 2);
          softMaxSumTemp[qq * 16] = softMaxSumTemp[qq * 16] + softMaxSumTemp[qq * 16 + 1];
          softMaxSumTemp[qq * 16] = 1.0f / softMaxSumTemp[qq * 16];
          kvCacheOutFP32.select<16, 1>(qq * 16) = kvCacheOutFP32.select<16, 1>(qq * 16) * softMaxSumTemp[qq * 16];
          __ESIMD_ENS::lsc_block_store<
            fp16,
            16,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::write_back,
            __ESIMD_ENS::cache_hint::write_back>((fp16*)out + outputOffset + qq * 128 * Q_HEAD, kvCacheOutFP32.select<16, 1>(qq * 16));
        }
      }
      else
      {
    #pragma unroll
        for (int qq = 0; qq < Q_REST; qq++) {
          softMaxSumTemp.select<8, 1>(qq * 16) = softMaxSumTemp.select<8, 1>(qq * 16) + softMaxSumTemp.select<8, 1>(qq * 16 + 8);
          softMaxSumTemp.select<4, 1>(qq * 16) = softMaxSumTemp.select<4, 1>(qq * 16) + softMaxSumTemp.select<4, 1>(qq * 16 + 4);
          softMaxSumTemp.select<2, 1>(qq * 16) = softMaxSumTemp.select<2, 1>(qq * 16) + softMaxSumTemp.select<2, 1>(qq * 16 + 2);
          softMaxSumTemp[qq * 16] = softMaxSumTemp[qq * 16] + softMaxSumTemp[qq * 16 + 1];
          softMaxSumTemp[qq * 16] = 1.0f / softMaxSumTemp[qq * 16];
          kvCacheOutFP32.select<16, 1>(qq * 16) = kvCacheOutFP32.select<16, 1>(qq * 16) * softMaxSumTemp[qq * 16];
          __ESIMD_ENS::lsc_block_store<
            fp16,
            16,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::write_back,
            __ESIMD_ENS::cache_hint::write_back>((fp16*)out + outputOffset + qq * 128 * Q_HEAD, kvCacheOutFP32.select<16, 1>(qq * 16));
        }
      }
    }
  }
}
