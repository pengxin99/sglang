#include "utils.h"

template <uint32_t pixelPerGroupShift>
ESIMD_INLINE void matrixMulCommonDim14336Int4NoReshapeNx16V2NoSLM(
    uint8_t* a,
    uint8_t* b,
    uint8_t* c,
    uint8_t* d,
    nd_item<1>& ndi) {
  constexpr uint32_t pixelPerGroup = 1 << pixelPerGroupShift;
  constexpr uint32_t quantPerGroup = 14336 / 32 * pixelPerGroup;
  constexpr uint32_t sumThreads = pixelPerGroup / 16;
  constexpr uint32_t baseOffsetInc16[16] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  constexpr uint32_t baseOffsetInc8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  constexpr uint32_t offsetSLM = 16 * 128 * sizeof(float);
  __ESIMD_NS::slm_init(16 * 128 * sizeof(float));
  int hh = ndi.get_local_id(0); // [0, 64)
  int h = ndi.get_group(0); // [0, 256)
  int rowSize = ndi.get_group_range(0) * pixelPerGroup;
  int offsetABase = (h * pixelPerGroup * 14336 + hh * 8 * 8) >> 1;
  int offsetQuanBase =
      /*rowSize * 7168 +*/ h * quantPerGroup * sizeof(fp16) +
      hh * 2 * sizeof(fp16);
  int offsetBBase = hh * 64 * sizeof(fp16);
  int outputOffset = pixelPerGroup * h;
  int offsetSLMThread = hh * 2 * 64 * sizeof(float);
  simd<unsigned char, 64> aaa;
  simd<fp16, 32> quant;
  simd<fp16, 448> bb;
  simd<float, 256> bb_fp32;
  simd<float, 4 * 16> aa;
  simd<float, 16> cc(0.0f);
  simd<fp16, 16> cc_fp16(0.0f);
  simd<uint32_t, 8> offsetA(baseOffsetInc8);
  simd<uint32_t, 8> offsetQuan(baseOffsetInc8);
  offsetA = offsetA * sizeof(uint32_t) + offsetABase;
  offsetQuan = offsetQuan * 32 * sizeof(fp16) + offsetQuanBase;

#pragma unroll
  for (int ll = 0; ll < 2; ll++) {
    int offsetB = offsetBBase + ll * 1024 * sizeof(fp16);
#pragma unroll
    for (int k = 0; k < 7; k++) {
      bb.template bit_cast_view<unsigned char>().template select<128, 1>(
          128 * k) =
          __ESIMD_ENS::lsc_block_load<
              uint8_t,
              128,
              __ESIMD_ENS::lsc_data_size::default_size,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);

      offsetB += 2 * 1024 * sizeof(fp16);
    }

    for (int n = 0; n < pixelPerGroup; n++) {
      cc = 0.0f;
      offsetQuan = baseOffsetInc8;
      // offsetQuan = offsetQuan * 32 * sizeof(fp16) + offsetQuanBase + n * 448
      // * sizeof(fp16);
      offsetQuan = offsetQuan * 32 * 2 * sizeof(fp16) + offsetQuanBase +
          n * 448 * sizeof(fp16);

      // offsetQuan += ll * 32 * sizeof(fp16) * 8;
      offsetQuan += ll * 32 * sizeof(fp16);

      quant.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
          __ESIMD_ENS::lsc_gather<
              uint32_t,
              1,
              __ESIMD_ENS::lsc_data_size::u32,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached,
              8,
              uint32_t>((uint32_t*)d, offsetQuan);

      offsetA = baseOffsetInc8;
      offsetA = offsetA * sizeof(uint32_t) + offsetABase + n * 7168;
      offsetA += ll * 512;

#pragma unroll
      for (int k = 0; k < 7; k++) {
        simd<float, 2> fp32Q = quant.select<2, 1>(2 * k);
        aaa.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
            __ESIMD_ENS::lsc_gather<
                uint32_t,
                1,
                __ESIMD_ENS::lsc_data_size::u32,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached,
                8,
                uint32_t>((uint32_t*)a, offsetA);
        offsetA += 1024; // 2 * 512

#pragma unroll
        for (int kk = 0; kk < 2; kk++) {
          //   aa.select<16, 1>(32 * kk) = aaa.select<16, 1>(16 * kk) & 0xf;
          //   aa.select<16, 1>(32 * kk + 16) = aaa.select<16, 1>(16 * kk) >> 4;
          aa.select<16, 2>(32 * kk) = aaa.select<16, 1>(16 * kk) & 0xf;
          aa.select<16, 2>(32 * kk + 1) = aaa.select<16, 1>(16 * kk) >> 4;
        }

        aa = aa - 8.0f;
#pragma unroll
        for (int kk = 0; kk < 2; kk++) {
          aa.select<32, 1>(32 * kk) = fp32Q[kk] * aa.select<32, 1>(32 * kk);
        }

#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          // cc += aa.select<16, 1>(16 * kk) * bb.select<16, 1>(16 * kk + 128 *
          // k);
          cc += aa.select<16, 1>(16 * kk) * bb.select<16, 1>(16 * kk + 64 * k);
        }
      }

      cc.select<8, 1>(0) += cc.select<8, 1>(8);
      cc.select<4, 1>(0) += cc.select<4, 1>(4);
      cc.select<2, 1>(0) += cc.select<2, 1>(2);
      simd<float, 1> slmAccumulationTemp = cc[0] + cc[1];
      uint32_t slmGroup = n >> 4;
      uint32_t slmInnerGroupOffset = n & 0xf;
      uint32_t slmAccumulationOffset =
          (slmGroup * 16 * 16 + hh * 16 + slmInnerGroupOffset) * sizeof(float);

      if (ll != 0) {
        simd<float, 1> dd = slm_block_load<float, 1>(slmAccumulationOffset);
        slmAccumulationTemp += dd[0];
      }
      // slm_scalar_store(slmAccumulationOffset, slmAccumulationTemp);
      slm_block_store<float, 1>(slmAccumulationOffset, slmAccumulationTemp);
    }
  }
  barrier();
  if (hh < sumThreads) {
    uint32_t slmSumPhase1LoadOffset = hh * 16 * 16 * sizeof(float);
#pragma unroll
    for (int k = 0; k < 4; k++) {
      bb_fp32.select<64, 1>(64 * k) = slm_block_load<float, 64>(
          slmSumPhase1LoadOffset + 64 * k * sizeof(float));
    }

#pragma unroll
    for (int k = 1; k < 16; k++) {
      bb_fp32.select<16, 1>(0) =
          bb_fp32.select<16, 1>(0) + bb_fp32.select<16, 1>(16 * k);
    }

    cc_fp16.select<16, 1>(0) = bb_fp32.select<16, 1>(0);
    __ESIMD_ENS::lsc_block_store<
        fp16,
        16,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::write_back,
        __ESIMD_ENS::cache_hint::write_back>(
        (fp16*)c + outputOffset + hh * 16, cc_fp16.select<16, 1>(0));
  }
}