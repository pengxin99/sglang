#include "utils.h"

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using fp16 = sycl::half;
using namespace sycl;

ESIMD_INLINE void gemmReduce2048WeightsQ40InputFp16_ipex(
    uint8_t* a,
    uint8_t* b,
    uint8_t* c,
    uint8_t* d,
    int hiddenDim,
    int tokenSize,
    int reduceIdx,
    int lastReduce,
    nd_item<2>& ndi) {
  /**
  GEMMReduce2048:
    Divide groups by the non-common dimension of weight, Divide local_threads by
  the common dimension(2048). Total groups: (non_dimension_weight +
  pixelPerGroup - 1) / pixelPerGroup Total local threads: 64 PixelPerThread:
  common dimension / 64 = 32 InputsPerLoop equans to 8, loop tokenSize /
  InputsPerLoop time in each group to handle all tokens of inputs. In each
  thread, will handle weights(pixelPerGroup, PixelPerThread) and
  inputs(tokenSize, PixelPerThread), got results(tokenSize, pixelPerGroup)
  */
  constexpr uint32_t baseOffsetInc16[16] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  constexpr uint32_t baseOffsetInc8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  constexpr uint32_t baseOffsetInc4[4] = {0, 1, 2, 3};

  // threads_num * InputsPerLoop * pixelPerGroup if for result of each thread
  // InputsPerLoop * 4 * pixelPerGroup if for the reduce results of every 16
  // threads
  __ESIMD_NS::slm_init(64 * 8 * 16 * sizeof(fp16) + 16 * 32 * sizeof(fp16));
  int hh = ndi.get_local_linear_id(); // [0, 64)
  int h = ndi.get_group(0); // [0, (row + 15) / 16)
  int v = reduceIdx; // [0, (hiddenDim + 2047) / 2048)
  int outputRow = ndi.get_group_range(0) *
      16; // non_dimension_weight, group_id * PixelPerGroup
  int hiddenDimInt4Size = hiddenDim >> 1; // hiddenDim * sizeof(int4)
  int hiddenDimDequantSize = hiddenDim >> 5; // hiddenDim / 32

  // globalOffset = reduce_idx * 2048 + groud_id * hiddenDim * PixelPerGroup +
  // local_thread_id * PixelPerThread
  uint32_t globalOffset = v * 2048 + h * hiddenDim * 16 + hh * 32;
  uint32_t baseOffsetA = globalOffset >> 1;
  // baseOffsetQuant = globalOffset / 32 * sizeof(fp16)
  uint32_t baseOffsetQuant =
      /*hiddenDimInt4Size * outputRow +*/ (globalOffset >> 4);
  uint32_t baseOffsetB =
      (v * 2048 + hh * 32) * sizeof(fp16); // thread_id * PixelPerThread
  uint32_t offsetC = h * 16 +
      hh * outputRow; // output shape is (tokenSize, non_dimension_weight)
  simd<fp16, 8 * 32> bb; // temp fp16 container
  simd<fp16, 8 * 32>
      bb_fp16; // input_fp16 container, PixelPerThread * InputsPerLoop
  simd<fp16, 32 * 16> aa; // weight container, PixelPerThread * PixelPerGroup
  simd<fp16, 8 * 16> cc; // container for qunat_scale/output_temp
  simd<fp16, 8 * 16> cc_fp16; // output container, InputsPerLoop * PixelPerGroup
  simd<uint32_t, 16> offset(baseOffsetInc16);
  simd<uint32_t, 16> offsetQuant(baseOffsetInc16);
  uint32_t loopCount =
      (tokenSize + 7) >> 3; // loop count for input, tokenSize / InputsPerLoop

  offsetQuant =
      offsetQuant * hiddenDimDequantSize * sizeof(fp16) + baseOffsetQuant;
  // load quant_scale(16, 1), will work with weights(16 * 32) during dequan
  // process
  {
    cc.template bit_cast_view<fp16>().template select<16, 1>(16 * 0) =
        __ESIMD_ENS::lsc_gather<
            fp16,
            1,
            __ESIMD_ENS::lsc_data_size::u16,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::uncached,
            16,
            uint32_t>((fp16*)d, offsetQuant);

    offsetQuant += 32 * sizeof(fp16);
  }

  cc.select<16, 1>(16) =
      cc.template bit_cast_view<fp16>().template select<16, 1>(0);

  offset = offset * hiddenDimInt4Size + baseOffsetA;

  {
    // load weights, shape (4, 16, 8), weight is int4, uint32_t buffer will
    // include 8 weights
    bb.template bit_cast_view<uint32_t>().template select<64, 1>(64 * 0) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            4,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::uncached,
            __ESIMD_ENS::cache_hint::uncached,
            16,
            uint32_t>((uint32_t*)a, offset);

    // shuffle weights to (4, 8, 16), weight is int4, so select stride is 4 for
    // 8 weights
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      simd<uint8_t, 16> bitShiftTemp =
          bb.template bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 0);

      simd<uint8_t, 16> temp = bitShiftTemp & 0xf;
      aa.select<16, 1>((4 * 2 * kk + 0) * 16) = temp;

      temp = bitShiftTemp >> 4;
      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 1 * 16) = temp;

      // ==========================================================================================
      bitShiftTemp =
          bb.template bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 1);

      temp = bitShiftTemp & 0xf;
      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 2 * 16) = temp;

      temp = bitShiftTemp >> 4;
      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 3 * 16) = temp;

      // ==========================================================================================
      bitShiftTemp =
          bb.template bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 2);

      temp = bitShiftTemp & 0xf;
      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 4 * 16) = temp;

      temp = bitShiftTemp >> 4;
      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 5 * 16) = temp;

      // ==========================================================================================
      bitShiftTemp =
          bb.template bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 3);

      temp = bitShiftTemp & 0xf;
      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 6 * 16) = temp;

      temp = bitShiftTemp >> 4;
      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 7 * 16) = temp;
    }
    offset += 128 * sizeof(uint32_t);
  }

#pragma unroll
  for (size_t k = 0; k < 32; k++) {
    aa.select<16, 1>(16 * k) -= 8.0f;
  }
  // dequant, weight(4, 8, 16) * quant_scale(1, 16).
  // PixelPerGroup as continuous dimension to leverage simd16
#pragma unroll
  for (int k = 0; k < 32; k++) {
    aa.select<16, 1>(16 * k) = aa.select<16, 1>(16 * k) * cc.select<16, 1>(16);
  }

  for (int nn = 0; nn < loopCount; nn++) {
    cc = 0;
    // loop for 8 inputs. load PixelPerThread * 8 inputs
#pragma unroll
    for (int k = 0; k < 8; k++) {
      bb_fp16.template bit_cast_view<uint8_t>().template select<64, 1>(64 * k) =
          __ESIMD_ENS::lsc_block_load<
              uint8_t,
              64,
              __ESIMD_ENS::lsc_data_size::default_size,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + baseOffsetB);

      // get results for each loop in local thread
      // (32, 16) * (1, 32) -> (1, 16), PixelPerGroup as continuous dimension to
      // leverage simd16
#pragma unroll
      for (int kkk = 0; kkk < 32; kkk++) {
        cc.select<16, 1>(16 * k) +=
            aa.select<16, 1>(kkk * 16) * bb_fp16[k * 32 + kkk];
      }
      // move baseOffsetB to next input token
      baseOffsetB += hiddenDim * sizeof(fp16);
    }

    // store results in SLM
    // SLMOffset: thread_id * PixelPerGroup + input_token_id * PixelPerGroup *
    // thread_num
#pragma unroll
    for (int k = 0; k < 8; k++) {
      slm_block_store<fp16, 16>(
          (hh * 16 + k * 16 * 64) * sizeof(fp16), cc.select<16, 1>(16 * k));
    }

    barrier();

    // after all thread complete, start first step to handle the results in diff
    // thread load results(8, 64, 16) from SLM, use 32 threads to sum results
    // for diff thread results(8, 64, 16) -> results(32, 16, 16) -> results(32,
    // 16) then store in SLM, offset: threads_num * InputsPerLoop *
    // PixelPerGroup + thread_id * PixelPerGroup
    if (hh < 32) {
      uint32_t slmOffset = hh * 16 * 16 * sizeof(fp16);
#pragma unroll
      for (int k = 0; k < 4; k++) {
        bb.select<64, 1>(64 * k) =
            slm_block_load<fp16, 64>(slmOffset + k * 64 * sizeof(fp16));
      }

#pragma unroll
      for (int k = 1; k < 8; k++) {
        bb.select<32, 1>(0) += bb.select<32, 1>(32 * k);
      }
      bb.select<16, 1>(0) += bb.select<16, 1>(16);
      slm_block_store<fp16, 16>(
          8 * 16 * 64 * sizeof(fp16) + hh * 16 * sizeof(fp16),
          bb.select<16, 1>(0));
    }

    barrier();
    // after first step complete, start second step to handle the results below
    // results(32, 16) -> results(8, 4, 16) -> results(8, 16)
    // load results(32, 16) from SLM, use 8 threads to sum results for diff
    // PixelPerGroup handle 4 * 16 results for each thread
    if (hh < 8) {
      if (8 * nn + hh < tokenSize) {
        // load result for before gemmReduce2048
        if (v != 0) {
          cc_fp16.select<16, 1>(0) = __ESIMD_ENS::lsc_block_load<
              fp16,
              16,
              __ESIMD_ENS::lsc_data_size::default_size,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached>((fp16*)c + offsetC);
        } else {
          cc_fp16.select<16, 1>(0) = 0;
        }
        // thread_id * 4 * PixelPerGroup + PixelPerGroup * threads_num *
        // InputsPerLoop
        uint32_t slmOffset =
            hh * 16 * 4 * sizeof(fp16) + 16 * 64 * 8 * sizeof(fp16);
        bb.template bit_cast_view<fp16>().template select<64, 1>(0) =
            slm_block_load<fp16, 64>(slmOffset);

#pragma unroll
        for (int k = 0; k < 4; k++) {
          cc_fp16.select<16, 1>(0) += bb.select<16, 1>(16 * k);
        }

        // store the final results int output buffer
        __ESIMD_ENS::lsc_block_store<
            fp16,
            16,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::write_back,
            __ESIMD_ENS::cache_hint::write_back>(
            (fp16*)c + offsetC, cc_fp16.select<16, 1>(0));

        offsetC += 8 * outputRow;
      }
    }
  }
}

ESIMD_INLINE void gemmReduce768WeightsQ40InputFp32_ipex(
    uint8_t* a,
    uint8_t* b,
    uint8_t* c,
    uint8_t* d,
    int hiddenDim,
    int tokenSize,
    int reduceIdx,
    int lastReduce,
    nd_item<2>& ndi) {
  constexpr uint32_t baseOffsetInc16[16] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  constexpr uint32_t baseOffsetInc8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  constexpr uint32_t baseOffsetInc4[4] = {0, 1, 2, 3};
  __ESIMD_NS::slm_init(24 * 8 * 16 * sizeof(float) + 16 * 32 * sizeof(float));
  int hh = ndi.get_local_linear_id(); // [0, 64)
  int h = ndi.get_group(0); // [0, (row + 15) / 16)
  int v = reduceIdx; // [0, (row + 15) / 16)
  int outputRow = ndi.get_group_range(0) * 16;
  int hiddenDimInt4Size = hiddenDim >> 1;
  int hiddenDimDequantSize = hiddenDim >> 5;
  uint32_t globalOffset = v * 2048 + h * hiddenDim * 16 + hh * 32;
  uint32_t baseOffsetA = globalOffset >> 1;
  uint32_t baseOffsetQuant =
      /*hiddenDimInt4Size * outputRow +*/ (globalOffset >> 4);
  uint32_t baseOffsetB = (v * 2048 + hh * 32) * sizeof(fp16);
  uint32_t offsetC = h * 16 + hh * outputRow;
  simd<float, 8 * 32> bb;
  simd<fp16, 8 * 32> bb_fp16;
  simd<float, 32 * 16> aa;
  simd<float, 8 * 16> cc;
  simd<fp16, 8 * 16> cc_fp16;
  simd<uint32_t, 16> offset(baseOffsetInc16);
  simd<uint32_t, 16> offsetQuant(baseOffsetInc16);
  uint32_t loopCount = (tokenSize + 7) >> 3;

  offsetQuant =
      offsetQuant * hiddenDimDequantSize * sizeof(fp16) + baseOffsetQuant;

  {
    cc.template bit_cast_view<fp16>().template select<16, 1>(16 * 0) =
        __ESIMD_ENS::lsc_gather<
            fp16,
            1,
            __ESIMD_ENS::lsc_data_size::u16,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::uncached,
            16,
            uint32_t>((fp16*)d, offsetQuant);

    offsetQuant += 12 * sizeof(fp16);
  }

  cc.select<16, 1>(16) =
      cc.template bit_cast_view<fp16>().template select<16, 1>(0);
  offset = offset * hiddenDimInt4Size + baseOffsetA;

  {
    bb.template bit_cast_view<uint32_t>().template select<64, 1>(64 * 0) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            4,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::uncached,
            __ESIMD_ENS::cache_hint::uncached,
            16,
            uint32_t>((uint32_t*)a, offset);

#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      simd<uint8_t, 16> bitShiftTemp =
          bb.template bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 0);

      simd<uint8_t, 16> temp = bitShiftTemp & 0xf;
      aa.select<16, 1>((4 * 2 * kk + 0) * 16) = temp;

      temp = bitShiftTemp >> 4;
      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 1 * 16) = temp;

      // ==========================================================================================
      bitShiftTemp =
          bb.template bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 1);

      temp = bitShiftTemp & 0xf;
      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 2 * 16) = temp;

      temp = bitShiftTemp >> 4;
      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 3 * 16) = temp;

      // ==========================================================================================
      bitShiftTemp =
          bb.template bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 2);

      temp = bitShiftTemp & 0xf;
      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 4 * 16) = temp;

      temp = bitShiftTemp >> 4;
      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 5 * 16) = temp;

      // ==========================================================================================
      bitShiftTemp =
          bb.template bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 3);

      temp = bitShiftTemp & 0xf;
      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 6 * 16) = temp;

      temp = bitShiftTemp >> 4;
      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 7 * 16) = temp;
    }
    offset += 48 * sizeof(uint32_t);
  }

#pragma unroll
  for (size_t k = 0; k < 32; k++) {
    aa.select<16, 1>(16 * k) -= 8.0f;
  }
#pragma unroll
  for (int k = 0; k < 32; k++) {
    aa.select<16, 1>(16 * k) = aa.select<16, 1>(16 * k) * cc.select<16, 1>(16);
  }

  for (int nn = 0; nn < loopCount; nn++) {
    cc = 0;
#pragma unroll
    for (int k = 0; k < 8; k++) {
      bb_fp16.template bit_cast_view<uint8_t>().template select<64, 1>(64 * k) =
          __ESIMD_ENS::lsc_block_load<
              uint8_t,
              64,
              __ESIMD_ENS::lsc_data_size::default_size,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + baseOffsetB);

#pragma unroll
      for (int kkk = 0; kkk < 32; kkk++) {
        cc.select<16, 1>(16 * k) +=
            aa.select<16, 1>(kkk * 16) * bb_fp16[k * 32 + kkk];
      }

      baseOffsetB += hiddenDim * sizeof(fp16);
    }

    barrier();

#pragma unroll
    for (int k = 0; k < 8; k++) {
      slm_block_store<float, 16>(
          (hh * 16 + k * 16 * 24) * sizeof(float), cc.select<16, 1>(16 * k));
    }

    barrier();

    if (hh < 8) {
      if (8 * nn + hh < tokenSize) {
        uint32_t slmOffset = hh * 16 * 24 * sizeof(float);
        if (v != 0) {
          cc_fp16.select<16, 1>(0) = __ESIMD_ENS::lsc_block_load<
              fp16,
              16,
              __ESIMD_ENS::lsc_data_size::default_size,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached>((fp16*)c + offsetC);
        } else {
          cc_fp16.select<16, 1>(0) = 0;
        }
#pragma unroll
        for (int k = 0; k < 4; k++) {
          bb.template bit_cast_view<float>().template select<64, 1>(64 * k) =
              slm_block_load<float, 64>(slmOffset);
          slmOffset += 64 * sizeof(float);
        }

#pragma unroll
        for (int k = 0; k < 16; k++) {
          cc_fp16.select<16, 1>(0) += bb.select<16, 1>(16 * k);
        }

#pragma unroll
        for (int k = 0; k < 2; k++) {
          bb.template bit_cast_view<float>().template select<64, 1>(64 * k) =
              slm_block_load<float, 64>(slmOffset);
          slmOffset += 64 * sizeof(float);
        }

#pragma unroll
        for (int k = 0; k < 8; k++) {
          cc_fp16.select<16, 1>(0) += bb.select<16, 1>(16 * k);
        }

        __ESIMD_ENS::lsc_block_store<
            fp16,
            16,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::write_back,
            __ESIMD_ENS::cache_hint::write_back>(
            (fp16*)c + offsetC, cc_fp16.select<16, 1>(0));

        offsetC += 8 * outputRow;
      }
    }
  }
}

ESIMD_INLINE void gemmReduce2048WeightsQ40InputFp32_ipex(
    uint8_t* a,
    uint8_t* b,
    uint8_t* c,
    uint8_t* d,
    int hiddenDim,
    int tokenSize,
    int reduceIdx,
    int lastReduce,
    nd_item<2>& ndi) {
  constexpr uint32_t baseOffsetInc16[16] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  constexpr uint32_t baseOffsetInc8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  constexpr uint32_t baseOffsetInc4[4] = {0, 1, 2, 3};
  __ESIMD_NS::slm_init(64 * 8 * 16 * sizeof(float) + 16 * 32 * sizeof(float));
  int hh = ndi.get_local_linear_id(); // [0, 64)
  int h = ndi.get_group(0); // [0, (row + 15) / 16)
  int v = reduceIdx; // [0, (row + 15) / 16)
  int outputRow = ndi.get_group_range(0) * 16;
  int hiddenDimInt4Size = hiddenDim >> 1;
  int hiddenDimDequantSize = hiddenDim >> 5;
  uint32_t globalOffset = v * 2048 + h * hiddenDim * 16 + hh * 32;
  uint32_t baseOffsetA = globalOffset >> 1;
  uint32_t baseOffsetQuant =
      /*hiddenDimInt4Size * outputRow +*/ (globalOffset >> 4);
  uint32_t baseOffsetB = (v * 2048 + hh * 32) * sizeof(fp16);
  uint32_t offsetC = h * 16 + hh * outputRow;
  simd<float, 8 * 32> bb;
  simd<fp16, 8 * 32> bb_fp16;
  simd<float, 32 * 16> aa;
  simd<float, 8 * 16> cc;
  simd<fp16, 8 * 16> cc_fp16;
  simd<uint32_t, 16> offset(baseOffsetInc16);
  simd<uint32_t, 16> offsetQuant(baseOffsetInc16);
  uint32_t loopCount = (tokenSize + 7) >> 3;

  offsetQuant =
      offsetQuant * hiddenDimDequantSize * sizeof(fp16) + baseOffsetQuant;

  {
    cc.template bit_cast_view<fp16>().template select<16, 1>(16 * 0) =
        __ESIMD_ENS::lsc_gather<
            fp16,
            1,
            __ESIMD_ENS::lsc_data_size::u16,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::uncached,
            16,
            uint32_t>((fp16*)d, offsetQuant);

    offsetQuant += 32 * sizeof(fp16);
  }

  cc.select<16, 1>(16) =
      cc.template bit_cast_view<fp16>().template select<16, 1>(0);

  offset = offset * hiddenDimInt4Size + baseOffsetA;

  {
    bb.template bit_cast_view<uint32_t>().template select<64, 1>(64 * 0) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            4,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::uncached,
            __ESIMD_ENS::cache_hint::uncached,
            16,
            uint32_t>((uint32_t*)a, offset);

#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      simd<int8_t, 16> bitShiftTemp =
          bb.template bit_cast_view<int8_t>().select<16, 4>(64 * kk + 0);

      simd<int8_t, 16> temp = bitShiftTemp & 0xf;
      temp.select<16, 1>(0) = temp.select<16, 1>(0) << 4;
      temp.select<16, 1>(0) = temp.select<16, 1>(0) >> 4;

      aa.select<16, 1>((4 * 2 * kk + 0) * 16) = temp;

      temp = bitShiftTemp >> 4;

      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 1 * 16) = temp;

      // ==========================================================================================
      bitShiftTemp =
          bb.template bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 1);

      temp = bitShiftTemp & 0xf;
      temp.select<16, 1>(0) = temp.select<16, 1>(0) << 4;
      temp.select<16, 1>(0) = temp.select<16, 1>(0) >> 4;

      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 2 * 16) = temp;

      temp = bitShiftTemp >> 4;

      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 3 * 16) = temp;

      // ==========================================================================================
      bitShiftTemp =
          bb.template bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 2);

      temp = bitShiftTemp & 0xf;
      temp.select<16, 1>(0) = temp.select<16, 1>(0) << 4;
      temp.select<16, 1>(0) = temp.select<16, 1>(0) >> 4;

      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 4 * 16) = temp;

      temp = bitShiftTemp >> 4;

      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 5 * 16) = temp;

      // ==========================================================================================
      bitShiftTemp =
          bb.template bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 3);

      temp = bitShiftTemp & 0xf;
      temp.select<16, 1>(0) = temp.select<16, 1>(0) << 4;
      temp.select<16, 1>(0) = temp.select<16, 1>(0) >> 4;

      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 6 * 16) = temp;

      temp = bitShiftTemp >> 4;

      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 7 * 16) = temp;
    }
    offset += 128 * sizeof(uint32_t);
  }

  // aa = aa - 8.0f;
#pragma unroll
  for (int k = 0; k < 32; k++) {
    aa.select<16, 1>(16 * k) = aa.select<16, 1>(16 * k) * cc.select<16, 1>(16);
  }

  for (int nn = 0; nn < loopCount; nn++) {
    cc = 0;
#pragma unroll
    for (int k = 0; k < 8; k++) {
      bb_fp16.template bit_cast_view<uint8_t>().template select<64, 1>(64 * k) =
          __ESIMD_ENS::lsc_block_load<
              uint8_t,
              64,
              __ESIMD_ENS::lsc_data_size::default_size,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + baseOffsetB);

#pragma unroll
      for (int kkk = 0; kkk < 32; kkk++) {
        cc.select<16, 1>(16 * k) +=
            aa.select<16, 1>(kkk * 16) * bb_fp16[k * 32 + kkk];
      }

      baseOffsetB += hiddenDim * sizeof(fp16);
    }

#pragma unroll
    for (int k = 0; k < 8; k++) {
      slm_block_store<float, 16>(
          (hh * 16 + k * 16 * 64) * sizeof(float), cc.select<16, 1>(16 * k));
    }

    barrier();

    if (hh < 32) {
      uint32_t slmOffset = hh * 16 * 16 * sizeof(float);
#pragma unroll
      for (int k = 0; k < 4; k++) {
        bb.select<64, 1>(64 * k) =
            slm_block_load<float, 64>(slmOffset + k * 64 * sizeof(float));
      }

#pragma unroll
      for (int k = 1; k < 8; k++) {
        bb.select<32, 1>(0) += bb.select<32, 1>(32 * k);
      }
      bb.select<16, 1>(0) += bb.select<16, 1>(16);
      slm_block_store<float, 16>(
          8 * 16 * 64 * sizeof(float) + hh * 16 * sizeof(float),
          bb.select<16, 1>(0));
    }

    barrier();

    if (hh < 8) {
      if (8 * nn + hh < tokenSize) {
        if (v != 0) {
          cc_fp16.select<16, 1>(0) = __ESIMD_ENS::lsc_block_load<
              fp16,
              16,
              __ESIMD_ENS::lsc_data_size::default_size,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached>((fp16*)c + offsetC);
        } else {
          cc_fp16.select<16, 1>(0) = 0;
        }
        uint32_t slmOffset =
            hh * 16 * 4 * sizeof(float) + 16 * 64 * 8 * sizeof(float);
        bb.template bit_cast_view<float>().template select<64, 1>(0) =
            slm_block_load<float, 64>(slmOffset);

#pragma unroll
        for (int k = 0; k < 4; k++) {
          cc_fp16.select<16, 1>(0) += bb.select<16, 1>(16 * k);
        }

        __ESIMD_ENS::lsc_block_store<
            fp16,
            16,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::write_back,
            __ESIMD_ENS::cache_hint::write_back>(
            (fp16*)c + offsetC, cc_fp16.select<16, 1>(0));

        offsetC += 8 * outputRow;
      }
    }
  }
}
