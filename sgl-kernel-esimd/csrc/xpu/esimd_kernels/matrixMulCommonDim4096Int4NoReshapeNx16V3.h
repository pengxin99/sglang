#include "utils.h"

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using fp16 = sycl::half;
using namespace sycl;

template <uint32_t pixelPerGroupShift>
ESIMD_INLINE void matrixMulCommonDim4096Int4NoReshapeNx16V3_ipex2(
    uint8_t* a,
    uint8_t* b,
    uint8_t* c,
    uint8_t* d,
    nd_item<1>& ndi) {
  /**
  GEMV: Divide groups by the non-common dimension, Divide local_threads by the
  common dimension. Total groups: (n + pixelPerGroup - 1) / pixelPerGroup Total
  local threads: 16 Pixel per thread: common dimension / 16 = 256 So calculate
  the result for 256 inputs and (256 * pixelPerGroup) weights in one local
  thread.
  */
  constexpr uint32_t pixelPerGroup = 1 << pixelPerGroupShift;
  constexpr uint32_t quantPerGroup =
      4096 / 32 * pixelPerGroup; // 32 quant weight perf quant scale
  constexpr uint32_t baseOffsetInc16[16] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  constexpr uint32_t baseOffsetInc8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  __ESIMD_NS::slm_init(
      16 * 16 *
      sizeof(float)); // slm size: local_threads * piexelPerGroup * output size
  int hh = ndi.get_local_id(0); // [0, 16)
  int h = ndi.get_group(0); // [0, 256)
  // int rowSize = ndi.get_group_range(0) * pixelPerGroup;

  // A is the weight, offsetABase is the offset in each local thread,
  // equals to group_id * pixelPerGroup * common_dim + thread_id * data per
  // lsc_gather
  int offsetABase = (h * pixelPerGroup * 4096 + hh * 8 * 8) >> 1;
  int offsetQuanBase = /*rowSize * 2048 +*/ h * quantPerGroup * sizeof(fp16) +
      hh * 2 * sizeof(fp16);
  int offsetB = hh * 64 * sizeof(fp16); // thread_id * data per lsc_block_load
  int outputOffset = pixelPerGroup * h;
  simd<int8_t, 128> aaa; // weight container, Pixel_per_thread * sizeof(int4)
  simd<fp16, 16> quant; // quant_scale container, quan_scale num per thread
  simd<float, 8> fp32Quant; // quant_scale_fp32 container
  simd<float, 256> bb; // input_fp32 container Pixel_per_thread * sizeof(float)
  simd<fp16, 256> bb_fp16; // input_fp16 container
  simd<float, 16 * 16>
      aa; // weight_shuffle container, Pixel_per_thread * sizeof(float)
  simd<float, 16> cc(0.0f); // output container, pixelPerGroup * sizeof(float)
  simd<uint32_t, 8> offsetA(baseOffsetInc8);
  simd<uint32_t, 8> offsetQuan(baseOffsetInc8);
  simd_mask<8> quantPred = 1;
  quantPred[4] = 0;
  quantPred[5] = 0;
  quantPred[6] = 0;
  quantPred[7] = 0;
  offsetA = offsetA * sizeof(uint32_t) +
      offsetABase; // load weight with 8*1 lsc_gather, the offset equals to
                   // offsetABase + offset_lsc_gather each line offset in the
                   // lsc_gather is sizeof(uint32_t)
  offsetQuan = offsetQuan * 32 * sizeof(fp16) +
      offsetQuanBase; // 2 quant_scales were take from each 32 qunat_scales

  // total 256 inputs were taken from 4096 common dimension
  // and 64 inputs were taken from each 1024 inputs
#pragma unroll
  for (int k = 0; k < 4; k++) {
    bb_fp16.template bit_cast_view<unsigned char>().template select<128, 1>(
        128 * k) =
        __ESIMD_ENS::lsc_block_load<
            uint8_t,
            128,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);

    offsetB += 1024 * sizeof(fp16);
  }

  // convert fp16 inputs to fp32 inputs
  bb.select<256, 1>(0) = bb_fp16.select<256, 1>(0);

  // total 256 weights were taken from 4096 common dimension,
  // and 64 weights were taken from each 1024 weights,
  // and 2 quant_scale were take from each 32 qunat_scales.
  for (int n = 0; n < pixelPerGroup; n++) {
    cc = 0.0f;
    quant.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)d, offsetQuan, quantPred);

    aaa.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)a, offsetA);
    offsetA += 512;

    aaa.template bit_cast_view<uint32_t>().template select<8, 1>(8) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)a, offsetA);
    offsetA += 512; // 2048 - 16 * sizeof(uint32_t)

    aaa.template bit_cast_view<uint32_t>().template select<8, 1>(8 * 2) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)a, offsetA);
    offsetA += 512; // 2048 - 16 * sizeof(uint32_t)

    aaa.template bit_cast_view<uint32_t>().template select<8, 1>(8 * 3) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)a, offsetA);
    offsetA += 512; // 2048 - 16 * sizeof(uint32_t)

    // shuffle weights
#pragma unroll
    for (int k = 0; k < 8; k++) {
      simd<uint8_t, 16> aaaa = aaa.select<16, 1>(16 * k) &
          0xf; // handle the low 4 bits for the first 16 weights
      aa.select<16, 2>(32 * k) = aaaa.select<16, 1>(0);

      aaaa.select<16, 1>(0) = aaa.select<16, 1>(
          16 * k); // handle the hight 4 bits for the first 16 weights
      aaaa.select<16, 1>(0) = aaaa.select<16, 1>(0) >> 4;
      aa.select<16, 2>(32 * k + 1) = aaaa.select<16, 1>(0);
    }

#pragma unroll
    for (size_t k = 0; k < 16; k++) {
      aa.select<16, 1>(16 * k) -= 8.0f;
    }
    fp32Quant = quant.select<8, 1>(0);

    // dequant 256(Pixel_per_thread) weights
#pragma unroll
    for (int k = 0; k < 8; k++) {
      aa.select<32, 1>(32 * k) = fp32Quant[k] * aa.select<32, 1>(32 * k);
    }

    // calculate 256(Pixel_per_thread) weights * inputs with simd16
#pragma unroll
    for (int k = 0; k < 16; k++) {
      cc += aa.select<16, 1>(16 * k) * bb.select<16, 1>(16 * k);
    }

    // add the 256(Pixel_per_thread) mat results with Fold-sum algorithm
    cc.select<8, 1>(0) += cc.select<8, 1>(8);
    cc.select<4, 1>(0) += cc.select<4, 1>(4);
    cc.select<2, 1>(0) += cc.select<2, 1>(2);
    simd<float, 1> slmAccumulationTemp = cc[0] + cc[1];

    // store the accumulation result into SLM.
    // SLM equals to an 2-D array with shape [local_threads, pixelPerGroup]
    // so slmAccumulationOffset is local_thread_id * pixelPerGroup + pixel_id
    uint32_t slmAccumulationOffset = (hh * pixelPerGroup + n) * sizeof(float);
    slm_block_store<float, 1>(slmAccumulationOffset, slmAccumulationTemp);
    offsetQuan += 128 * sizeof(fp16);
  }
  barrier();

  // after all local threads complete. use the first thread to Aggregate the
  // results of all threads
  if (hh == 0) {
    if constexpr (pixelPerGroupShift == 4) {
      // for pixelPerGroup 16, load all the slmAccumulationTemp from SLM
      // calculate the final pixelPerGroup results with simd16.
#pragma unroll
      for (int k = 0; k < 4; k++) {
        bb.select<64, 1>(64 * k) =
            slm_block_load<float, 64>(64 * k * sizeof(float));
      }
#pragma unroll
      for (int k = 1; k < 16; k++) {
        bb.select<16, 1>(0) += bb.select<16, 1>(16 * k);
      }

    } else if constexpr (pixelPerGroupShift == 3) {
#pragma unroll
      for (int k = 0; k < 2; k++) {
        bb.select<64, 1>(64 * k) =
            slm_block_load<float, 64>(64 * k * sizeof(float));
      }
#pragma unroll
      for (int k = 1; k < 8; k++) {
        bb.select<16, 1>(0) += bb.select<16, 1>(16 * k);
      }
      bb.select<8, 1>(0) += bb.select<8, 1>(8);
    } else if constexpr (pixelPerGroupShift == 2) {
      bb.select<64, 1>(0) = slm_block_load<float, 64>(0);
#pragma unroll
      for (int k = 1; k < 4; k++) {
        bb.select<16, 1>(0) += bb.select<16, 1>(16 * k);
      }
      bb.select<8, 1>(0) += bb.select<8, 1>(8);
      bb.select<4, 1>(0) += bb.select<4, 1>(4);
    } else if constexpr (pixelPerGroupShift == 1) {
      bb.select<32, 1>(0) = slm_block_load<float, 32>(0);
      bb.select<16, 1>(0) += bb.select<16, 1>(16 * 1);
      bb.select<8, 1>(0) += bb.select<8, 1>(8);
      bb.select<4, 1>(0) += bb.select<4, 1>(4);
      bb.select<2, 1>(0) += bb.select<2, 1>(2);
    } else if constexpr (pixelPerGroupShift == 0) {
      bb.select<16, 1>(0) = slm_block_load<float, 16>(0);
      bb.select<8, 1>(0) += bb.select<8, 1>(8);
      bb.select<4, 1>(0) += bb.select<4, 1>(4);
      bb.select<2, 1>(0) += bb.select<2, 1>(2);
      bb.select<1, 1>(0) += bb.select<1, 1>(1);
    }

    bb_fp16.select<pixelPerGroup, 1>(0) = bb.select<pixelPerGroup, 1>(0);

    __ESIMD_ENS::lsc_block_store<
        fp16,
        pixelPerGroup,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::write_back,
        __ESIMD_ENS::cache_hint::write_back>(
        (fp16*)c + outputOffset, bb_fp16.select<pixelPerGroup, 1>(0));
  }
}

template <uint32_t pixelPerGroupShift>
ESIMD_INLINE void matrixMulCommonDim3072Int4NoReshapeNx16V3_ipex2(
    uint8_t* a,
    uint8_t* b,
    uint8_t* c,
    uint8_t* d,
    nd_item<1>& ndi) {
  /**
  GEMV: Divide groups by the non-common dimension, Divide local_threads by the
  common dimension. Total groups: (n + pixelPerGroup - 1) / pixelPerGroup Total
  local threads: 16 Pixel per thread: common dimension / 16 = 256 So calculate
  the result for 256 inputs and (256 * pixelPerGroup) weights in one local
  thread.
  */
  constexpr uint32_t pixelPerGroup = 1 << pixelPerGroupShift;
  constexpr uint32_t quantPerGroup =
      3072 / 32 * pixelPerGroup; // 32 quant weight perf quant scale
  constexpr uint32_t baseOffsetInc16[16] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  constexpr uint32_t baseOffsetInc8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  __ESIMD_NS::slm_init(
      16 * 16 *
      sizeof(float)); // slm size: local_threads * piexelPerGroup * output size
  int hh = ndi.get_local_id(0); // [0, 16)
  int h = ndi.get_group(0); // [0, 256)
  // int rowSize = ndi.get_group_range(0) * pixelPerGroup;

  // A is the weight, offsetABase is the offset in each local thread,
  // equals to group_id * pixelPerGroup * common_dim + thread_id * data per
  // lsc_gather
  int offsetABase = (h * pixelPerGroup * 3072 + hh * 8 * 8) >> 1;
  int offsetQuanBase = /*rowSize * 2048 +*/ h * quantPerGroup * sizeof(fp16) +
      hh * 2 * sizeof(fp16);
  int offsetB = hh * 64 * sizeof(fp16); // thread_id * data per lsc_block_load
  int outputOffset = pixelPerGroup * h;
  simd<int8_t, 128> aaa; // weight container, Pixel_per_thread * sizeof(int4)
  simd<fp16, 16> quant; // quant_scale container, quan_scale num per thread
  simd<float, 8> fp32Quant; // quant_scale_fp32 container
  simd<float, 256> bb; // input_fp32 container Pixel_per_thread * sizeof(float)
  simd<fp16, 256> bb_fp16; // input_fp16 container
  simd<float, 16 * 16>
      aa; // weight_shuffle container, Pixel_per_thread * sizeof(float)
  simd<float, 16> cc(0.0f); // output container, pixelPerGroup * sizeof(float)
  simd<uint32_t, 8> offsetA(baseOffsetInc8);
  simd<uint32_t, 8> offsetQuan(baseOffsetInc8);
  simd_mask<8> quantPred = 1;
  quantPred[3] = 0;
  quantPred[4] = 0;
  quantPred[5] = 0;
  quantPred[6] = 0;
  quantPred[7] = 0;
  offsetA = offsetA * sizeof(uint32_t) +
      offsetABase; // load weight with 8*1 lsc_gather, the offset equals to
                   // offsetABase + offset_lsc_gather each line offset in the
                   // lsc_gather is sizeof(uint32_t)
  offsetQuan = offsetQuan * 32 * sizeof(fp16) +
      offsetQuanBase; // 2 quant_scales were take from each 32 qunat_scales

  // total 256 inputs were taken from 4096 common dimension
  // and 64 inputs were taken from each 1024 inputs
#pragma unroll
  for (int k = 0; k < 3; k++) {
    bb_fp16.template bit_cast_view<unsigned char>().template select<128, 1>(
        128 * k) =
        __ESIMD_ENS::lsc_block_load<
            uint8_t,
            128,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);

    offsetB += 1024 * sizeof(fp16);
  }

  // convert fp16 inputs to fp32 inputs
  bb.select<256, 1>(0) = bb_fp16.select<256, 1>(0);

  // total 256 weights were taken from 4096 common dimension,
  // and 64 weights were taken from each 1024 weights,
  // and 2 quant_scale were take from each 32 qunat_scales.
  for (int n = 0; n < pixelPerGroup; n++) {
    cc = 0.0f;
    quant.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)d, offsetQuan, quantPred);

    aaa.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)a, offsetA);
    offsetA += 512;

    aaa.template bit_cast_view<uint32_t>().template select<8, 1>(8) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)a, offsetA);
    offsetA += 512; // 2048 - 16 * sizeof(uint32_t)

    aaa.template bit_cast_view<uint32_t>().template select<8, 1>(8 * 2) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)a, offsetA);
    offsetA += 512; // 2048 - 16 * sizeof(uint32_t)

    // aaa.template bit_cast_view<uint32_t>().template select<8, 1>(8 * 3) =
    //     __ESIMD_ENS::lsc_gather<
    //         uint32_t,
    //         1,
    //         __ESIMD_ENS::lsc_data_size::u32,
    //         __ESIMD_ENS::cache_hint::cached,
    //         __ESIMD_ENS::cache_hint::cached,
    //         8,
    //         uint32_t>((uint32_t*)a, offsetA);
    // offsetA += 512; // 2048 - 16 * sizeof(uint32_t)

    // shuffle weights
#pragma unroll
    for (int k = 0; k < 6; k++) {
      simd<uint8_t, 16> aaaa = aaa.select<16, 1>(16 * k) &
          0xf; // handle the low 4 bits for the first 16 weights
      aa.select<16, 2>(32 * k) = aaaa.select<16, 1>(0);

      aaaa.select<16, 1>(0) = aaa.select<16, 1>(
          16 * k); // handle the hight 4 bits for the first 16 weights
      aaaa.select<16, 1>(0) = aaaa.select<16, 1>(0) >> 4;
      aa.select<16, 2>(32 * k + 1) = aaaa.select<16, 1>(0);
    }

#pragma unroll
    for (size_t k = 0; k < 12; k++) {
      aa.select<16, 1>(16 * k) -= 8.0f;
    }
    fp32Quant = quant.select<8, 1>(0);

    // dequant 256(Pixel_per_thread) weights
#pragma unroll
    for (int k = 0; k < 6; k++) {
      aa.select<32, 1>(32 * k) = fp32Quant[k] * aa.select<32, 1>(32 * k);
    }

    // calculate 256(Pixel_per_thread) weights * inputs with simd16
#pragma unroll
    for (int k = 0; k < 12; k++) {
      cc += aa.select<16, 1>(16 * k) * bb.select<16, 1>(16 * k);
    }

    // add the 256(Pixel_per_thread) mat results with Fold-sum algorithm
    cc.select<8, 1>(0) += cc.select<8, 1>(8);
    cc.select<4, 1>(0) += cc.select<4, 1>(4);
    cc.select<2, 1>(0) += cc.select<2, 1>(2);
    simd<float, 1> slmAccumulationTemp = cc[0] + cc[1];

    // store the accumulation result into SLM.
    // SLM equals to an 2-D array with shape [local_threads, pixelPerGroup]
    // so slmAccumulationOffset is local_thread_id * pixelPerGroup + pixel_id
    uint32_t slmAccumulationOffset = (hh * pixelPerGroup + n) * sizeof(float);
    slm_block_store<float, 1>(slmAccumulationOffset, slmAccumulationTemp);
    offsetQuan += 3072 / 32 * sizeof(fp16);
  }
  barrier();

  // after all local threads complete. use the first thread to Aggregate the
  // results of all threads
  if (hh == 0) {
    if constexpr (pixelPerGroupShift == 4) {
      // for pixelPerGroup 16, load all the slmAccumulationTemp from SLM
      // calculate the final pixelPerGroup results with simd16.
#pragma unroll
      for (int k = 0; k < 4; k++) {
        bb.select<64, 1>(64 * k) =
            slm_block_load<float, 64>(64 * k * sizeof(float));
      }
#pragma unroll
      for (int k = 1; k < 16; k++) {
        bb.select<16, 1>(0) += bb.select<16, 1>(16 * k);
      }

    } else if constexpr (pixelPerGroupShift == 3) {
#pragma unroll
      for (int k = 0; k < 2; k++) {
        bb.select<64, 1>(64 * k) =
            slm_block_load<float, 64>(64 * k * sizeof(float));
      }
#pragma unroll
      for (int k = 1; k < 8; k++) {
        bb.select<16, 1>(0) += bb.select<16, 1>(16 * k);
      }
      bb.select<8, 1>(0) += bb.select<8, 1>(8);
    } else if constexpr (pixelPerGroupShift == 2) {
      bb.select<64, 1>(0) = slm_block_load<float, 64>(0);
#pragma unroll
      for (int k = 1; k < 4; k++) {
        bb.select<16, 1>(0) += bb.select<16, 1>(16 * k);
      }
      bb.select<8, 1>(0) += bb.select<8, 1>(8);
      bb.select<4, 1>(0) += bb.select<4, 1>(4);
    } else if constexpr (pixelPerGroupShift == 1) {
      bb.select<32, 1>(0) = slm_block_load<float, 32>(0);
      bb.select<16, 1>(0) += bb.select<16, 1>(16 * 1);
      bb.select<8, 1>(0) += bb.select<8, 1>(8);
      bb.select<4, 1>(0) += bb.select<4, 1>(4);
      bb.select<2, 1>(0) += bb.select<2, 1>(2);
    } else if constexpr (pixelPerGroupShift == 0) {
      bb.select<16, 1>(0) = slm_block_load<float, 16>(0);
      bb.select<8, 1>(0) += bb.select<8, 1>(8);
      bb.select<4, 1>(0) += bb.select<4, 1>(4);
      bb.select<2, 1>(0) += bb.select<2, 1>(2);
      bb.select<1, 1>(0) += bb.select<1, 1>(1);
    }

    bb_fp16.select<pixelPerGroup, 1>(0) = bb.select<pixelPerGroup, 1>(0);

    __ESIMD_ENS::lsc_block_store<
        fp16,
        pixelPerGroup,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::write_back,
        __ESIMD_ENS::cache_hint::write_back>(
        (fp16*)c + outputOffset, bb_fp16.select<pixelPerGroup, 1>(0));
  }
}


template <uint32_t pixelPerGroupShift>
ESIMD_INLINE void matrixMulCommonDim11008Int4NoReshapeNx16V2_ipex(
    uint8_t* a,
    uint8_t* b,
    uint8_t* c,
    uint8_t* d,
    nd_item<1>& ndi) {
  /**
  GEMV for common dimension 11008:
        Divide groups by the non-common dimension, Divide local_threads by the
  common dimension. Total groups: (n + pixelPerGroup - 1) / pixelPerGroup Total
  local threads: 16 Pixel per thread: Not evenly in each thread. handle 704
  elements per thread in the first 12 threads, handle 640 elements per thread
  for other 4 threads
  */
  constexpr uint32_t pixelPerGroup = 1 << pixelPerGroupShift;
  constexpr uint32_t quantPerGroup =
      11008 / 32 * pixelPerGroup; // 32 quant weight perf quant scale
  constexpr uint32_t sumThreads = pixelPerGroup /
      16; // 16 result in no-dimension will put in one Group in SLM
  constexpr uint32_t baseOffsetInc16[16] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  constexpr uint32_t baseOffsetInc8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  __ESIMD_NS::slm_init(
      16 * 128 * sizeof(float)); // local_thread_num * pixelPerGroup?
  int hh = ndi.get_local_id(0); // [0, 16)
  int h = ndi.get_group(0); // [0, 64)
  // int rowSize = ndi.get_group_range(0) * pixelPerGroup;

  // A is weight, offsetABase is the offset in each local thread,
  // equals to group_id * pixelPerGroup * common_dim + thread_id * data per
  // lsc_gather offsetQuanBase is the offset of quant_Scale, group_id *
  // pixelPerGroup + thread_id * quant_scale per lsc_gather
  int offsetABase = (h * pixelPerGroup * 11008 + hh * 8 * 8) >> 1;
  int offsetQuanBase = /*rowSize * 5504 +*/ h * quantPerGroup * sizeof(fp16) +
      hh * 2 * sizeof(fp16);
  int offsetB =
      hh * 64 * sizeof(fp16); // B is input, handle 64 inputs per local_threads
  int outputOffset = pixelPerGroup * h;
  simd<int8_t, 64> aaa; // temp weight container in each local_thread
  simd<fp16, 32> quant; // quant_scale container, quan_scale num per thread
  simd<fp16, 704>
      bb; // input_fp16 container max_Pixel_per_thread * sizeof(float)
  simd<fp16, 256> bbb; // input_fp16 container for reduce step
  simd<float, 8 * 16>
      aa; // weight_shuffle container, Pixel_per_thread * sizeof(float)
  simd<float, 16> cc(0.0f); // output container, pixelPerGroup * sizeof(float)
  simd<uint32_t, 8> offsetA(baseOffsetInc8);
  simd<uint32_t, 8> offsetQuan(baseOffsetInc8);
  offsetA = offsetA * sizeof(uint32_t) +
      offsetABase; // load 64 weights continuously per 1024 weights
  offsetQuan = offsetQuan * 32 * sizeof(fp16) +
      offsetQuanBase; // load 2 quan_scale per 1024 weights

  // Load input
  // for the first 12 local threads, total 704 inputs were taken from 11008
  // common dimension and 64 inputs were taken from each 1024 inputs
#pragma unroll
  for (int k = 0; k < 10; k++) {
    bb.template bit_cast_view<unsigned char>().template select<128, 1>(
        128 * k) =
        __ESIMD_ENS::lsc_block_load<
            uint8_t,
            128,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);

    offsetB += 1024 * sizeof(fp16);
  }

  if (hh < 12) {
    bb.template bit_cast_view<unsigned char>().template select<128, 1>(
        128 * 10) =
        __ESIMD_ENS::lsc_block_load<
            uint8_t,
            128,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);
  } else {
    // for last 4 threads, only 640 inputs were taken, so here is 0
    bb.template bit_cast_view<unsigned char>().template select<128, 1>(
        128 * 10) = 0;
  }

  // Load weight/quant_scale
  // for the first 12 local threads, total 704 weights were taken from 11008
  // common dimension and 64 weights were taken from each 1024 inputs
  for (int n = 0; n < pixelPerGroup; n++) {
    cc = 0.0f;
    offsetQuan = baseOffsetInc8;
    // 32 quant_scale for 1024 weights, 344 quant_scale for 11008 weights
    offsetQuan = offsetQuan * 32 * sizeof(fp16) + offsetQuanBase +
        n * 344 * sizeof(fp16);
    // load 8 * 2 quant_scale for the first 8 * 128 weights
    quant.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)d, offsetQuan);

    // offset of 1024 quant_scale(fp16) * 8
    offsetQuan += 32 * sizeof(fp16) * 8;

    // load 8 * 2 quant_scale for the sencod 8 * 128 weights
    quant.template bit_cast_view<uint32_t>().template select<8, 1>(8) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)d, offsetQuan);

    // load 8 * 8 weights continuously one time, need 12 times load for the
    // first 12 threads, 10 times for other threads for diff PixelPerGroup,
    // offset need to add 11008 * sizeof(int4)
    offsetA = baseOffsetInc8;
    offsetA = offsetA * sizeof(uint32_t) + offsetABase + n * 5504;

#pragma unroll
    for (int k = 0; k < 5; k++) {
      simd<float, 4> fp32Q = quant.select<4, 1>(4 * k);
      aaa.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
          __ESIMD_ENS::lsc_gather<
              uint32_t,
              1,
              __ESIMD_ENS::lsc_data_size::u32,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached,
              8,
              uint32_t>((uint32_t*)a, offsetA);
      offsetA += 512; // offset of 1024 weight(int4)

      aaa.template bit_cast_view<uint32_t>().template select<8, 1>(8) =
          __ESIMD_ENS::lsc_gather<
              uint32_t,
              1,
              __ESIMD_ENS::lsc_data_size::u32,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached,
              8,
              uint32_t>((uint32_t*)a, offsetA);
      offsetA += 512; // offset of 1024 weight(int4)

      // shuffle weights
#pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        simd<uint8_t, 16> aaaa = aaa.select<16, 1>(16 * kk) &
            0xf; // handle the low 4 bits for the first 16 weights
        aa.select<16, 2>(32 * kk) = aaaa.select<16, 1>(0);

        aaaa.select<16, 1>(0) = aaa.select<16, 1>(
            16 * kk); // handle the hight 4 bits for the first 16 weights
        aaaa.select<16, 1>(0) = aaaa.select<16, 1>(0) >> 4;
        aa.select<16, 2>(32 * kk + 1) = aaaa.select<16, 1>(0);
      }

#pragma unroll
      for (size_t k = 0; k < 8; k++) {
        aa.select<16, 1>(16 * k) -= 8.0f;
      }
      // dequant 128 weights in each thread, 640 weights for 5 loops
#pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        aa.select<32, 1>(32 * kk) = fp32Q[kk] * aa.select<32, 1>(32 * kk);
      }

      // calculate 128 inputs/weights in each thread with simd16, 640
      // inputs/weights for 5 loops
#pragma unroll
      for (int kk = 0; kk < 8; kk++) {
        cc += aa.select<16, 1>(16 * kk) * bb.select<16, 1>(16 * kk + 128 * k);
      }
    }

    // for the first 12 local_threas, need to handle 64 more elements
    if (hh < 12) {
      simd<float, 2> fp32Q = quant.select<2, 1>(4 * 5);
      aaa.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
          __ESIMD_ENS::lsc_gather<
              uint32_t,
              1,
              __ESIMD_ENS::lsc_data_size::u32,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached,
              8,
              uint32_t>((uint32_t*)a, offsetA);
      offsetA += 512; // offset of 1024 weight(int4)

#pragma unroll
      for (int kk = 0; kk < 2; kk++) {
        simd<uint8_t, 16> aaaa = aaa.select<16, 1>(16 * kk) & 0xf;
        aa.select<16, 2>(32 * kk) = aaaa.select<16, 1>(0);

        aaaa.select<16, 1>(0) = aaa.select<16, 1>(16 * kk);
        aaaa.select<16, 1>(0) = aaaa.select<16, 1>(0) >> 4;
        aa.select<16, 2>(32 * kk + 1) = aaaa.select<16, 1>(0);
      }
#pragma unroll
      for (size_t k = 0; k < 4; k++) {
        aa.select<16, 1>(16 * k) -= 8.0f;
      }

#pragma unroll
      for (int kk = 0; kk < 2; kk++) {
        aa.select<32, 1>(32 * kk) = fp32Q[kk] * aa.select<32, 1>(32 * kk);
      }

#pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        cc += aa.select<16, 1>(16 * kk) * bb.select<16, 1>(16 * kk + 128 * 5);
      }
    }

    // add the 704 or 640(Pixel_per_thread) mat results with Fold-sum algorithm
    cc.select<8, 1>(0) += cc.select<8, 1>(8);
    cc.select<4, 1>(0) += cc.select<4, 1>(4);
    cc.select<2, 1>(0) += cc.select<2, 1>(2);
    simd<float, 1> slmAccumulationTemp = cc[0] + cc[1];

    // pixelPerGroup is 64, 16 pixel as on Group in SLM, total 4 Groups in SLM,
    // to leverage simd16 n >> 4 to get slmGroup id n & 0xf to get the index in
    // slmGroup
    uint32_t slmGroup = n >> 4; // [0, 4)
    uint32_t slmInnerGroupOffset = n & 0xf; // [0, 16)
    // SLM can look as 3-D array with shape (slmGroup_num, local_thread_size,
    // slm_group_size) slm offset for each result is slmGroup_id *
    // loacl_thread_id * slmGroup_size + local_thread_id * slmGroup_size +
    // slmInnerGroupOffset
    uint32_t slmAccumulationOffset =
        (slmGroup * 16 * 16 + hh * 16 + slmInnerGroupOffset) * sizeof(float);
    // slm_scalar_store(slmAccumulationOffset, slmAccumulationTemp);
    slm_block_store<float, 1>(slmAccumulationOffset, slmAccumulationTemp);
  }
  barrier();
  // each sumThread will get 16 results in the non-dimension
  if (hh < sumThreads) {
    // local_thread_id * local_thread_num * slm_group_size
    // need to load 16 * 16 slmAccumulationTemp each local_thread
    uint32_t slmSumPhase1LoadOffset = hh * 16 * 16 * sizeof(float);
#pragma unroll
    for (int k = 0; k < 4; k++) {
      bbb.select<64, 1>(64 * k) = slm_block_load<float, 64>(
          slmSumPhase1LoadOffset + 64 * k * sizeof(float));
    }

    // got 16 final results with Fold-sum algorithm
#pragma unroll
    for (int k = 1; k < 16; k++) {
      bbb.select<16, 1>(0) = bbb.select<16, 1>(0) + bbb.select<16, 1>(16 * k);
    }

    bb.select<16, 1>(0) = bbb.select<16, 1>(0);

    __ESIMD_ENS::lsc_block_store<
        fp16,
        16,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::write_back,
        __ESIMD_ENS::cache_hint::write_back>(
        (fp16*)c + outputOffset + hh * 16, bb.select<16, 1>(0));
  }
}

template <uint32_t pixelPerGroupShift>
ESIMD_INLINE void matrixMulCommonDim14336Int4NoReshapeNx16V2_ipex(
    uint8_t* a,
    uint8_t* b,
    uint8_t* c,
    uint8_t* d,
    nd_item<1>& ndi) {
  /**
  GEMV for common dimension 14336:
        Divide groups by the non-common dimension, Divide local_threads by the
  common dimension. Total groups: (n + pixelPerGroup - 1) / pixelPerGroup Total
  local threads: 16 Pixel per thread: handle 896 elements per thread, store
  128/896 inputs into SLM for each thread to avoid GRF spill
  */
  constexpr uint32_t pixelPerGroup = 1 << pixelPerGroupShift;
  constexpr uint32_t quantPerGroup =
      14336 / 32 * pixelPerGroup; // 32 quant weight perf quant scale
  constexpr uint32_t sumThreads = pixelPerGroup /
      16; // 16 result in no-dimension will put in one Group in SLM
  constexpr uint32_t baseOffsetInc16[16] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  constexpr uint32_t baseOffsetInc8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  constexpr uint32_t offsetSLM = 16 * 128 * sizeof(float);
  __ESIMD_NS::slm_init(16 * 128 * sizeof(float) + 16 * 2 * 64 * sizeof(fp16));
  int hh = ndi.get_local_id(0); // [0, 16)
  int h = ndi.get_group(0); // [0, 256)
  // int rowSize = ndi.get_group_range(0) * pixelPerGroup;

  // A is weight, offsetABase is the offset in each local thread,
  // equals to group_id * pixelPerGroup * common_dim + thread_id * data per
  // lsc_gather offsetQuanBase is the offset of quant_Scale, group_id *
  // pixelPerGroup + thread_id * quant_scale per lsc_gather
  int offsetABase = (h * pixelPerGroup * 14336 + hh * 8 * 8) >> 1;
  int offsetQuanBase = /*rowSize * 7168 +*/ h * quantPerGroup * sizeof(fp16) +
      hh * 2 * sizeof(fp16);
  int offsetB =
      hh * 64 * sizeof(fp16); // B is input, handle 64 inputs per local_thread
  int outputOffset = pixelPerGroup * h;
  int offsetSLMThread = hh * 2 * 64 *
      sizeof(fp16); // 128 inputs store into the SLM per local_thread
  simd<int8_t, 64> aaa; // temp weight container in each local_thread
  simd<fp16, 32> quant; // quant_scale container, quan_scale num per thread
  // simd<fp16, 896> bb;                                            //
  // input_fp16 container max_Pixel_per_thread * sizeof(float)
  simd<fp16, 768> bb; // actually input_fp16 container to avoid grf spill
  simd<fp16, 64> bbb; // temp input container in each local_thread
  simd<float, 8 * 16> aa; // weight container, handle 128 weight
  simd<float, 16> cc(0.0f); // output container, pixelPerGroup * sizeof(float)
  simd<uint32_t, 8> offsetA(baseOffsetInc8);
  simd<uint32_t, 8> offsetQuan(baseOffsetInc8);
  offsetA = offsetA * sizeof(uint32_t) +
      offsetABase; // load 64 weights continuously per 1024 weights
  offsetQuan = offsetQuan * 32 * sizeof(fp16) +
      offsetQuanBase; // load 2 quan_scale per 1024 weights

  // Load input
  // total 768 inputs were taken from 14336 common dimension, then store in  GRF
  // and 64 inputs were taken from each 1024 inputs
#pragma unroll
  for (int k = 0; k < 12; k++) {
    bb.template bit_cast_view<unsigned char>().template select<128, 1>(
        128 * k) =
        __ESIMD_ENS::lsc_block_load<
            uint8_t,
            128,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);

    offsetB += 1024 * sizeof(fp16);
  }

  // total 128 inputs were taken from 14336 common dimension, then store in  SLM
  // and 64 inputs were taken from each 1024 inputs
#pragma unroll
  for (int k = 0; k < 2; k++) {
    bbb.template bit_cast_view<unsigned char>().template select<128, 1>(0) =
        __ESIMD_ENS::lsc_block_load<
            uint8_t,
            128,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);
    slm_block_store<fp16, 64>(
        offsetSLM + offsetSLMThread + k * 64 * sizeof(fp16),
        bbb.select<64, 1>(0));

    offsetB += 1024 * sizeof(fp16);
  }

  // Load weight/quant_scale
  // for the first 12 local threads, total 704 weights were taken from 11008
  // common dimension and 64 weights were taken from each 1024 inputs
  for (int n = 0; n < pixelPerGroup; n++) {
    cc = 0.0f;
    offsetQuan = baseOffsetInc8;
    // 32 quant_scale for 1024 weights, 448 quant_scale for 11008 weights
    offsetQuan = offsetQuan * 32 * sizeof(fp16) + offsetQuanBase +
        n * 448 * sizeof(fp16);
    // load 8 * 2 quant_scale for the first 8 * 128 weights
    quant.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)d, offsetQuan);

    // offset of 1024 quant_scale(fp16) * 8
    offsetQuan += 32 * sizeof(fp16) * 8;

    // load 8 * 2 quant_scale for the sencod 8 * 128 weights
    quant.template bit_cast_view<uint32_t>().template select<8, 1>(8) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)d, offsetQuan);

    // load 8 * 8 weights continuously one time, need 14 times load all 896
    // weights for diff PixelPerGroup, offset need to add 14336 * sizeof(int4)
    offsetA = baseOffsetInc8;
    offsetA = offsetA * sizeof(uint32_t) + offsetABase + n * 7168;

#pragma unroll
    for (int k = 0; k < 7; k++) {
      simd<float, 4> fp32Q = quant.select<4, 1>(4 * k);
      aaa.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
          __ESIMD_ENS::lsc_gather<
              uint32_t,
              1,
              __ESIMD_ENS::lsc_data_size::u32,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached,
              8,
              uint32_t>((uint32_t*)a, offsetA);
      offsetA += 512; // offset of 1024 weight(int4)

      aaa.template bit_cast_view<uint32_t>().template select<8, 1>(8) =
          __ESIMD_ENS::lsc_gather<
              uint32_t,
              1,
              __ESIMD_ENS::lsc_data_size::u32,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached,
              8,
              uint32_t>((uint32_t*)a, offsetA);
      offsetA += 512; // offset of 1024 weight(int4)

      // shuffle weights
#pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        simd<uint8_t, 16> aaaa = aaa.select<16, 1>(16 * kk) & 0xf;
        aa.select<16, 2>(32 * kk) = aaaa.select<16, 1>(0);

        aaaa.select<16, 1>(0) = aaa.select<16, 1>(16 * kk);
        aaaa.select<16, 1>(0) = aaaa.select<16, 1>(0) >> 4;
        aa.select<16, 2>(32 * kk + 1) = aaaa.select<16, 1>(0);
      }

#pragma unroll
      for (size_t k = 0; k < 8; k++) {
        aa.select<16, 1>(16 * k) -= 8.0f;
      }
      // dequant 128 weights in each loop
#pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        aa.select<32, 1>(32 * kk) = fp32Q[kk] * aa.select<32, 1>(32 * kk);
      }

      // calculate 128 inputs/weights in each thread with simd16, 896
      // inputs/weights for 7 loops the inputs store in GRF in the first 6 loops
      // the inputs store in SLM in the 7th loop
      if (k < 6) {
#pragma unroll
        for (int kk = 0; kk < 8; kk++) {
          cc += aa.select<16, 1>(16 * kk) * bb.select<16, 1>(16 * kk + 128 * k);
        }
      } else {
#pragma unroll
        for (int kk = 0; kk < 8; kk++) {
          bbb.select<16, 1>(0) = slm_block_load<fp16, 16>(
              offsetSLM + offsetSLMThread + 16 * kk * sizeof(fp16));
          cc += aa.select<16, 1>(16 * kk) * bbb.select<16, 1>(0);
        }
      }
    }

    // add the 896(Pixel_per_thread) mat results with Fold-sum algorithm
    cc.select<8, 1>(0) += cc.select<8, 1>(8);
    cc.select<4, 1>(0) += cc.select<4, 1>(4);
    cc.select<2, 1>(0) += cc.select<2, 1>(2);
    simd<float, 1> slmAccumulationTemp = cc[0] + cc[1];

    // pixelPerGroup is 16, 16 pixel as on Group in SLM, total 1 Groups in SLM,
    // to leverage simd16 n >> 4 to get slmGroup id n & 0xf to get the index in
    // slmGroup
    uint32_t slmGroup = n >> 4; // [0, 1)
    uint32_t slmInnerGroupOffset = n & 0xf; // [0, 16)
    uint32_t slmAccumulationOffset =
        (slmGroup * 16 * 16 + hh * 16 + slmInnerGroupOffset) * sizeof(float);
    // slm_scalar_store(slmAccumulationOffset, slmAccumulationTemp);
    slm_block_store<float, 1>(slmAccumulationOffset, slmAccumulationTemp);
  }
  barrier();
  // each sumThread will get 16 results in the non-dimension
  if (hh < sumThreads) {
    // local_thread_id * local_thread_num * slm_group_size
    // need to load 16 * 16 slmAccumulationTemp each local_thread
    uint32_t slmSumPhase1LoadOffset = hh * 16 * 16 * sizeof(float);
#pragma unroll
    for (int k = 0; k < 4; k++) {
      bb.select<64, 1>(64 * k) = slm_block_load<float, 64>(
          slmSumPhase1LoadOffset + 64 * k * sizeof(float));
    }

    // got 16 final results with Fold-sum algorithm
#pragma unroll
    for (int k = 1; k < 16; k++) {
      bb.select<16, 1>(0) = bb.select<16, 1>(0) + bb.select<16, 1>(16 * k);
    }

    // bb.select<16, 1>(0) = bb_f32.select<16, 1>(0);

    __ESIMD_ENS::lsc_block_store<
        fp16,
        16,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::write_back,
        __ESIMD_ENS::cache_hint::write_back>(
        (fp16*)c + outputOffset + hh * 16, bb.select<16, 1>(0));
  }
}

template <uint32_t pixelPerGroupShift>
ESIMD_INLINE void matrixMulCommonDim4096Int4NoReshapeNx16V3_ipex2_fused(
    uint8_t* a,
    uint8_t* b,
    uint8_t* out1,
    uint8_t* out2,
    uint8_t* out3,
    uint8_t* d,
    nd_item<1>& ndi) {
  /**
  GEMV: Divide groups by the non-common dimension, Divide local_threads by the
  common dimension. Total groups: (n + pixelPerGroup - 1) / pixelPerGroup Total
  local threads: 16 Pixel per thread: common dimension / 16 = 256 So calculate
  the result for 256 inputs and (256 * pixelPerGroup) weights in one local
  thread.
  */
  constexpr uint32_t pixelPerGroup = 1 << pixelPerGroupShift;
  constexpr uint32_t quantPerGroup =
      4096 / 32 * pixelPerGroup; // 32 quant weight perf quant scale
  constexpr uint32_t baseOffsetInc16[16] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  constexpr uint32_t baseOffsetInc8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  __ESIMD_NS::slm_init(
      16 * 16 *
      sizeof(float)); // slm size: local_threads * piexelPerGroup * output size
  int hh = ndi.get_local_id(0); // [0, 16)
  int h = ndi.get_group(0); // [0, 256)
  // int rowSize = ndi.get_group_range(0) * pixelPerGroup;

  // A is the weight, offsetABase is the offset in each local thread,
  // equals to group_id * pixelPerGroup * common_dim + thread_id * data per
  // lsc_gather
  int offsetABase = (h * pixelPerGroup * 4096 + hh * 8 * 8) >> 1;
  int offsetQuanBase = /*rowSize * 2048 +*/ h * quantPerGroup * sizeof(fp16) +
      hh * 2 * sizeof(fp16);
  int offsetB = hh * 64 * sizeof(fp16); // thread_id * data per lsc_block_load
  int outputOffset = pixelPerGroup * h;
  simd<int8_t, 128> aaa; // weight container, Pixel_per_thread * sizeof(int4)
  simd<fp16, 16> quant; // quant_scale container, quan_scale per thread
  simd<float, 8> fp32Quant; // quant_scale_fp32 container
  simd<float, 256> bb; // input_fp32 container Pixel_per_thread * sizeof(float)
  simd<fp16, 256> bb_fp16; // input_fp16 container
  simd<float, 16 * 16>
      aa; // weight_shuffle container, Pixel_per_thread * sizeof(float)
  simd<float, 16> cc(0.0f); // output container, pixelPerGroup * sizeof(float)
  simd<uint32_t, 8> offsetA(baseOffsetInc8);
  simd<uint32_t, 8> offsetQuan(baseOffsetInc8);
  simd_mask<8> quantPred = 1;
  quantPred[4] = 0;
  quantPred[5] = 0;
  quantPred[6] = 0;
  quantPred[7] = 0;
  offsetA = offsetA * sizeof(uint32_t) +
      offsetABase; // load weight with 8*1 lsc_gather, the offset equals to
                   // offsetABase + offset_lsc_gather each line offset in the
                   // lsc_gather is sizeof(uint32_t)
  offsetQuan = offsetQuan * 32 * sizeof(fp16) +
      offsetQuanBase; // 2 quant_scales were take from each 32 qunat_scales

  // total 256 inputs were taken from 4096 common dimension
  // and 64 inputs were taken from each 1024 inputs
#pragma unroll
  for (int k = 0; k < 4; k++) {
    bb_fp16.template bit_cast_view<unsigned char>().template select<128, 1>(
        128 * k) =
        __ESIMD_ENS::lsc_block_load<
            uint8_t,
            128,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);

    offsetB += 1024 * sizeof(fp16);
  }

  // convert fp16 inputs to fp32 inputs
  bb.select<256, 1>(0) = bb_fp16.select<256, 1>(0);

  // total 256 weights were taken from 4096 common dimension,
  // and 64 weights were taken from each 1024 weights,
  // and 2 quant_scale were take from each 32 qunat_scales.
  for (int n = 0; n < pixelPerGroup; n++) {
    cc = 0.0f;
    quant.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)d, offsetQuan, quantPred);

    aaa.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)a, offsetA);
    offsetA += 512;

    aaa.template bit_cast_view<uint32_t>().template select<8, 1>(8) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)a, offsetA);
    offsetA += 512; // 2048 - 16 * sizeof(uint32_t)

    aaa.template bit_cast_view<uint32_t>().template select<8, 1>(8 * 2) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)a, offsetA);
    offsetA += 512; // 2048 - 16 * sizeof(uint32_t)

    aaa.template bit_cast_view<uint32_t>().template select<8, 1>(8 * 3) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)a, offsetA);
    offsetA += 512; // 2048 - 16 * sizeof(uint32_t)

    // shuffle weights
#pragma unroll
    for (int k = 0; k < 8; k++) {
      simd<uint8_t, 16> aaaa = aaa.select<16, 1>(16 * k) &
          0xf; // handle the low 4 bits for the first 16 weights
      aa.select<16, 2>(32 * k) = aaaa.select<16, 1>(0);

      aaaa.select<16, 1>(0) = aaa.select<16, 1>(
          16 * k); // handle the hight 4 bits for the first 16 weights
      aaaa.select<16, 1>(0) = aaaa.select<16, 1>(0) >> 4;
      aa.select<16, 2>(32 * k + 1) = aaaa.select<16, 1>(0);
    }

#pragma unroll
    for (size_t k = 0; k < 16; k++) {
      aa.select<16, 1>(16 * k) -= 8.0f;
    }
    fp32Quant = quant.select<8, 1>(0);

    // dequant 256(Pixel_per_thread) weights
#pragma unroll
    for (int k = 0; k < 8; k++) {
      aa.select<32, 1>(32 * k) = fp32Quant[k] * aa.select<32, 1>(32 * k);
    }

    // calculate 256(Pixel_per_thread) weights * inputs with simd16
#pragma unroll
    for (int k = 0; k < 16; k++) {
      cc += aa.select<16, 1>(16 * k) * bb.select<16, 1>(16 * k);
    }

    // add the 256(Pixel_per_thread) mat results with Fold-sum algorithm
    cc.select<8, 1>(0) += cc.select<8, 1>(8);
    cc.select<4, 1>(0) += cc.select<4, 1>(4);
    cc.select<2, 1>(0) += cc.select<2, 1>(2);
    simd<float, 1> slmAccumulationTemp = cc[0] + cc[1];

    // store the accumulation result into SLM.
    // SLM equals to an 2-D array with shape [local_threads, pixelPerGroup]
    // so slmAccumulationOffset is local_thread_id * pixelPerGroup + pixel_id
    uint32_t slmAccumulationOffset = (hh * pixelPerGroup + n) * sizeof(float);
    slm_block_store<float, 1>(slmAccumulationOffset, slmAccumulationTemp);
    offsetQuan += 128 * sizeof(fp16);
  }
  barrier();

  // after all local threads complete. use the first thread to Aggregate the
  // results of all threads
  if (hh == 0) {
    if constexpr (pixelPerGroupShift == 4) {
      // for pixelPerGroup 16, load all the slmAccumulationTemp from SLM
      // calculate the final pixelPerGroup results with simd16.
#pragma unroll
      for (int k = 0; k < 4; k++) {
        bb.select<64, 1>(64 * k) =
            slm_block_load<float, 64>(64 * k * sizeof(float));
      }
#pragma unroll
      for (int k = 1; k < 16; k++) {
        bb.select<16, 1>(0) += bb.select<16, 1>(16 * k);
      }

    } else if constexpr (pixelPerGroupShift == 3) {
#pragma unroll
      for (int k = 0; k < 2; k++) {
        bb.select<64, 1>(64 * k) =
            slm_block_load<float, 64>(64 * k * sizeof(float));
      }
#pragma unroll
      for (int k = 1; k < 8; k++) {
        bb.select<16, 1>(0) += bb.select<16, 1>(16 * k);
      }
      bb.select<8, 1>(0) += bb.select<8, 1>(8);
    } else if constexpr (pixelPerGroupShift == 2) {
      bb.select<64, 1>(0) = slm_block_load<float, 64>(0);
#pragma unroll
      for (int k = 1; k < 4; k++) {
        bb.select<16, 1>(0) += bb.select<16, 1>(16 * k);
      }
      bb.select<8, 1>(0) += bb.select<8, 1>(8);
      bb.select<4, 1>(0) += bb.select<4, 1>(4);
    } else if constexpr (pixelPerGroupShift == 1) {
      bb.select<32, 1>(0) = slm_block_load<float, 32>(0);
      bb.select<16, 1>(0) += bb.select<16, 1>(16 * 1);
      bb.select<8, 1>(0) += bb.select<8, 1>(8);
      bb.select<4, 1>(0) += bb.select<4, 1>(4);
      bb.select<2, 1>(0) += bb.select<2, 1>(2);
    } else if constexpr (pixelPerGroupShift == 0) {
      bb.select<16, 1>(0) = slm_block_load<float, 16>(0);
      bb.select<8, 1>(0) += bb.select<8, 1>(8);
      bb.select<4, 1>(0) += bb.select<4, 1>(4);
      bb.select<2, 1>(0) += bb.select<2, 1>(2);
      bb.select<1, 1>(0) += bb.select<1, 1>(1);
    }

    bb_fp16.select<pixelPerGroup, 1>(0) = bb.select<pixelPerGroup, 1>(0);

    // the non common dimension of Q is 4096
    if (pixelPerGroup * h < 4096) {
      __ESIMD_ENS::lsc_block_store<
          fp16,
          pixelPerGroup,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::write_back,
          __ESIMD_ENS::cache_hint::write_back>(
          (fp16*)out1 + outputOffset, bb_fp16.select<pixelPerGroup, 1>(0));
    }
    // the non common dimension of K is 1024
    else if (pixelPerGroup * h < 5120) {
      __ESIMD_ENS::lsc_block_store<
          fp16,
          pixelPerGroup,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::write_back,
          __ESIMD_ENS::cache_hint::write_back>(
          (fp16*)out2 + outputOffset - 4096, // K offset should minus all Q data
          bb_fp16.select<pixelPerGroup, 1>(0));
    }
    // the non common dimension of V is 1024
    else if (pixelPerGroup * h < 6144) {
      __ESIMD_ENS::lsc_block_store<
          fp16,
          pixelPerGroup,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::write_back,
          __ESIMD_ENS::cache_hint::write_back>(
          (fp16*)out3 + outputOffset - 4096 -
              1024, // V offset should minus all Q/K data
          bb_fp16.select<pixelPerGroup, 1>(0));
    }
  }
}
