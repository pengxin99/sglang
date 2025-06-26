#include "utils.h"

#define GROUP 32
template<uint32_t K_DIM, uint32_t pixelPerGroupShift>
ESIMD_INLINE void GEMV_Int4Weight_FP16InOutNx16Temp_largeGRF_block_ppg8_8T_32g_768(uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* d, nd_item<1>& ndi) {
  constexpr uint32_t pixelPerGroup = 1 << pixelPerGroupShift;
  constexpr uint32_t quantPerGroup = K_DIM / GROUP * pixelPerGroup;
  constexpr uint32_t sumThreads = 1;
  constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  constexpr uint32_t baseOffsetInc8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
  constexpr uint32_t K_DIM_DIV_4096 = (K_DIM + 4095) / 4096;
  constexpr uint32_t K_DIM_MOD_4096 = (K_DIM - (K_DIM_DIV_4096-1)*4096);
  constexpr uint32_t K_DIM_REDUCE_T = (K_DIM_MOD_4096 +  (4096 / 8)-1) / (4096 / 8); // 8 threads, 512 ele per thread in total 4096
  __ESIMD_NS::slm_init(16 * sizeof(float));
  int hh = ndi.get_local_id(0); // [0, 64)
  int h = ndi.get_group(0); // [0, 256)
  int rowSize = ndi.get_group_range(0) * pixelPerGroup;
  int offsetABase = (h * pixelPerGroup * K_DIM + hh * 8 * 8 * 4 * 2) >> 1;
  int offsetQuanBase = /*rowSize * K_DIM / 2 +*/ h * quantPerGroup * sizeof(fp16) + hh * 512 / GROUP * sizeof(fp16);
  int offsetB = hh * 128 * 2 * 2 * sizeof(fp16);
  int outputOffset = pixelPerGroup * h;
  int offsetSLMThread = hh * 2 * 64 * sizeof(float);
  simd<unsigned char, 256> aaa;
  simd<fp16, 80> quant;
  simd<fp16, 2560> bb;
  simd<float, 8 * 16 * 4> aa;
  simd<float, 16> cc(0.0f);
  uint32_t offsetA;
  uint32_t offsetQuan;
  offsetA = offsetABase;
  offsetQuan = offsetQuanBase;

#pragma unroll
  for (int k = 0; k < K_DIM_DIV_4096; k++) {
    if (k != K_DIM_DIV_4096-1 || hh < K_DIM_REDUCE_T)
    {
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*1) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 1 * 128 * sizeof(fp16));
      // special for 768
      if (hh == 1)
      {
        bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*2) = 0;
        bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*3) = 0;
      }
      else
      {
        bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*2) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 2 * 128 * sizeof(fp16));
        bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*3) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 3 * 128 * sizeof(fp16));
      }

    }
    else
    {
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*1) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*2) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*3) = 0;
    }

    offsetB += 4096 * sizeof(fp16);
  }

  for (int n = 0; n < pixelPerGroup; n++) {
    cc = 0.0f;
    offsetQuan = offsetQuanBase + n * K_DIM / GROUP * sizeof(fp16);
#pragma unroll
    for (int k = 0; k < K_DIM_DIV_4096-1; k++) {
      quant.template bit_cast_view<unsigned char>().template select<512 / GROUP * 2, 1>(512 / GROUP * 2 * k) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        512 / GROUP * 2,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuan);
      offsetQuan += 4096 / GROUP * sizeof(fp16);
    }
    if (hh < K_DIM_REDUCE_T)
    {
        // special for 768
      if (hh == 1)
      {
        quant.template bit_cast_view<unsigned char>().template select<256 / GROUP * 2, 1>(512 / GROUP * 2 * (K_DIM_DIV_4096-1)) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256 / GROUP * 2,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuan);
      }
      else
      {
        quant.template bit_cast_view<unsigned char>().template select<512 / GROUP * 2, 1>(512 / GROUP * 2 * (K_DIM_DIV_4096-1)) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          512 / GROUP * 2,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuan);
      }
    }
    else
    {
      quant.template bit_cast_view<unsigned char>().template select<32, 1>(32 * (K_DIM_DIV_4096-1)) = 0;
    }
    offsetQuan += 4096 / GROUP * sizeof(fp16);

    offsetA = offsetABase + n * K_DIM / 2;

#pragma unroll
    for (int k = 0; k < K_DIM_DIV_4096; k++) {
      if (k != K_DIM_DIV_4096-1 || hh < K_DIM_REDUCE_T)
      {
        simd<float, 512 / GROUP> fp32Q = quant.select<512 / GROUP, 1>(512 / GROUP * k);
         // special for 768
        if (hh == 1)
        {
          aaa.template bit_cast_view<unsigned char>().template select<128, 1>(0) =
            __ESIMD_ENS::lsc_block_load<
            uint8_t,
            128,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offsetA);
          fp32Q.select<256 / GROUP, 1>(256 / GROUP) = 0.0f;
        }
        else
        {
          aaa.template bit_cast_view<unsigned char>().template select<256, 1>(0) =
            __ESIMD_ENS::lsc_block_load<
            uint8_t,
            256,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offsetA);
        }

#pragma unroll
        for (int kk = 0; kk < 16; kk++) {
          aa.select<16, 2>(32 * kk) = aaa.select<16, 1>(16 * kk) & 0xf;
          aa.select<16, 2>(32 * kk + 1) = aaa.select<16, 1>(16 * kk) >> 4;
        }

        aa = aa - 8.0f;
#pragma unroll
        for (int kk = 0; kk < 512 / GROUP; kk++) {
          aa.select<GROUP, 1>(GROUP * kk) = fp32Q[kk] * aa.select<GROUP, 1>(GROUP * kk);
        }
#pragma unroll
        for (int kk = 0; kk < 32; kk++) {
          cc += aa.select<16, 1>(16 * kk) * bb.select<16, 1>(16 * kk + 128 * 2 * 2 * k); // note: 128 * 2 easy to have mistake!
        }
      }

      if (k == K_DIM_DIV_4096-1)
      {
        offsetA += K_DIM_REDUCE_T * 512 / 2;
      }
      else
      {
        offsetA += 2048;
      }
    }

    cc.select<8, 1>(0) += cc.select<8, 1>(8);
    cc.select<4, 1>(0) += cc.select<4, 1>(4);
    cc.select<2, 1>(0) += cc.select<2, 1>(2);
    simd<float, 1> slmAccumulationTemp = cc[0] + cc[1];
    uint32_t slmAccumulationOffset = (hh * pixelPerGroup + n) * sizeof(float);
    slm_block_store<float, 1>(slmAccumulationOffset, slmAccumulationTemp);
  }
  barrier();
  if (hh < sumThreads) {
if constexpr (pixelPerGroupShift == 3) {
#pragma unroll
      for (int k = 0; k < 2; k++) {
        aa.select<64, 1>(64 * k) = slm_block_load<float, 64>(64 * k * sizeof(float));
      }
#pragma unroll
      for (int k = 1; k < 8; k++) {
        aa.select<16, 1>(0) += aa.select<16, 1>(16 * k);
      }
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
    } else if constexpr (pixelPerGroupShift == 2) {
      aa.select<64, 1>(0) = slm_block_load<float, 64>(0);
#pragma unroll
      for (int k = 1; k < 4; k++) {
        aa.select<16, 1>(0) += aa.select<16, 1>(16 * k);
      }
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
      aa.select<4, 1>(0) += aa.select<4, 1>(4);
    } else if constexpr (pixelPerGroupShift == 1) {
      aa.select<32, 1>(0) = slm_block_load<float, 32>(0);
      aa.select<16, 1>(0) += aa.select<16, 1>(16 * 1);
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
      aa.select<4, 1>(0) += aa.select<4, 1>(4);
      aa.select<2, 1>(0) += aa.select<2, 1>(2);
    } else if constexpr (pixelPerGroupShift == 0) {
      aa.select<16, 1>(0) = slm_block_load<float, 16>(0);
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
      aa.select<4, 1>(0) += aa.select<4, 1>(4);
      aa.select<2, 1>(0) += aa.select<2, 1>(2);
      aa.select<1, 1>(0) += aa.select<1, 1>(1);
    }

    __ESIMD_ENS::lsc_block_store<
      fp16,
      pixelPerGroup,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((fp16*)c + outputOffset, aa.select<pixelPerGroup, 1>(0));
  }
}


template<uint32_t K_DIM, uint32_t pixelPerGroupShift>
ESIMD_INLINE void MOE_up_or_gate_GEMV_Int4Weight_FP16InOutNx16Temp_largeGRF_block_ppg8_8T_32g(uint8_t* b, uint8_t* c,
  int64_t* indexs,
  uint64_t* weight_addrs,
  uint64_t* scale_addrs,
  nd_item<2>& ndi) 
{
  constexpr uint32_t pixelPerGroup = 1 << pixelPerGroupShift;
  constexpr uint32_t quantPerGroup = K_DIM / GROUP * pixelPerGroup;
  constexpr uint32_t sumThreads = 1;
  constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  constexpr uint32_t baseOffsetInc8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
  constexpr uint32_t K_DIM_DIV_4096 = (K_DIM + 4095) / 4096;
  constexpr uint32_t K_DIM_MOD_4096 = (K_DIM - (K_DIM_DIV_4096-1)*4096);
  constexpr uint32_t K_DIM_REDUCE_T = K_DIM_MOD_4096 / (4096 / 8); // 8 threads, 512 ele per thread in total 4096
  __ESIMD_NS::slm_init(16 * sizeof(float));
  int hh = ndi.get_local_id(1); // [0, 64)
  int h = ndi.get_group(1); // [0, 256)
  int eid = ndi.get_group(0); // [0, 256)
  int rowSize = ndi.get_group_range(1) * pixelPerGroup;
  int offsetABase = (h * pixelPerGroup * K_DIM + hh * 8 * 8 * 4 * 2) >> 1;
  int offsetQuanBase = /*rowSize * K_DIM / 2 +*/ h * quantPerGroup * sizeof(fp16) + hh * 512 / GROUP * sizeof(fp16);
  int offsetB = hh * 128 * 2 * 2 * sizeof(fp16);
  int outputOffset = pixelPerGroup * h + rowSize * eid;
  int offsetSLMThread = hh * 2 * 64 * sizeof(float);
  simd<unsigned char, 256> aaa;
  simd<fp16, 80> quant;
  simd<fp16, 2560> bb;
  simd<float, 8 * 16 * 4> aa;
  simd<float, 16> cc(0.0f);
  uint32_t offsetA;
  uint32_t offsetQuan;
  offsetA = offsetABase;
  offsetQuan = offsetQuanBase;

  uint8_t* a = (uint8_t *)(weight_addrs[indexs[eid]]);
  uint8_t* d = (uint8_t *)(scale_addrs[indexs[eid]]);

#pragma unroll
  for (int k = 0; k < K_DIM_DIV_4096; k++) {
    if (k != K_DIM_DIV_4096-1 || hh < K_DIM_REDUCE_T)
    {
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*1) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 1 * 128 * sizeof(fp16));
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*2) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 2 * 128 * sizeof(fp16));
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*3) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 3 * 128 * sizeof(fp16));
    }
    else
    {
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*1) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*2) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*3) = 0;
    }

    offsetB += 4096 * sizeof(fp16);
  }

  for (int n = 0; n < pixelPerGroup; n++) {
    cc = 0.0f;
    offsetQuan = offsetQuanBase + n * K_DIM / GROUP * sizeof(fp16);
#pragma unroll
    for (int k = 0; k < K_DIM_DIV_4096-1; k++) {
      quant.template bit_cast_view<unsigned char>().template select<512 / GROUP * 2, 1>(512 / GROUP * 2 * k) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        512 / GROUP * 2,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuan);
      offsetQuan += 4096 / GROUP * sizeof(fp16);
    }
    if (hh < K_DIM_REDUCE_T)
    {
      quant.template bit_cast_view<unsigned char>().template select<512 / GROUP * 2, 1>(512 / GROUP * 2 * (K_DIM_DIV_4096-1)) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        512 / GROUP * 2,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuan);
    }
    else
    {
      quant.template bit_cast_view<unsigned char>().template select<32, 1>(32 * (K_DIM_DIV_4096-1)) = 0;
    }
    offsetQuan += 4096 / GROUP * sizeof(fp16);

    offsetA = offsetABase + n * K_DIM / 2;

#pragma unroll
    for (int k = 0; k < K_DIM_DIV_4096; k++) {
      if (k != K_DIM_DIV_4096-1 || hh < K_DIM_REDUCE_T)
      {
        simd<float, 512 / GROUP> fp32Q = quant.select<512 / GROUP, 1>(512 / GROUP * k);
        aaa.template bit_cast_view<unsigned char>().template select<256, 1>(0) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offsetA);

#pragma unroll
        for (int kk = 0; kk < 16; kk++) {
          aa.select<16, 2>(32 * kk) = aaa.select<16, 1>(16 * kk) & 0xf;
          aa.select<16, 2>(32 * kk + 1) = aaa.select<16, 1>(16 * kk) >> 4;
        }

        aa = aa - 8.0f;
#pragma unroll
        for (int kk = 0; kk < 512 / GROUP; kk++) {
          aa.select<GROUP, 1>(GROUP * kk) = fp32Q[kk] * aa.select<GROUP, 1>(GROUP * kk);
        }
#pragma unroll
        for (int kk = 0; kk < 32; kk++) {
          cc += aa.select<16, 1>(16 * kk) * bb.select<16, 1>(16 * kk + 128 * 2 * 2 * k); // note: 128 * 2 easy to have mistake!
        }
      }

      if (k == K_DIM_DIV_4096-1)
      {
        offsetA += K_DIM_REDUCE_T * 512 / 2;
      }
      else
      {
        offsetA += 2048;
      }
    }

    cc.select<8, 1>(0) += cc.select<8, 1>(8);
    cc.select<4, 1>(0) += cc.select<4, 1>(4);
    cc.select<2, 1>(0) += cc.select<2, 1>(2);
    simd<float, 1> slmAccumulationTemp = cc[0] + cc[1];
    uint32_t slmAccumulationOffset = (hh * pixelPerGroup + n) * sizeof(float);
    slm_block_store<float, 1>(slmAccumulationOffset, slmAccumulationTemp);
  }
  barrier();
  if (hh < sumThreads) {
if constexpr (pixelPerGroupShift == 3) {
#pragma unroll
      for (int k = 0; k < 2; k++) {
        aa.select<64, 1>(64 * k) = slm_block_load<float, 64>(64 * k * sizeof(float));
      }
#pragma unroll
      for (int k = 1; k < 8; k++) {
        aa.select<16, 1>(0) += aa.select<16, 1>(16 * k);
      }
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
    } else if constexpr (pixelPerGroupShift == 2) {
      aa.select<64, 1>(0) = slm_block_load<float, 64>(0);
#pragma unroll
      for (int k = 1; k < 4; k++) {
        aa.select<16, 1>(0) += aa.select<16, 1>(16 * k);
      }
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
      aa.select<4, 1>(0) += aa.select<4, 1>(4);
    } else if constexpr (pixelPerGroupShift == 1) {
      aa.select<32, 1>(0) = slm_block_load<float, 32>(0);
      aa.select<16, 1>(0) += aa.select<16, 1>(16 * 1);
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
      aa.select<4, 1>(0) += aa.select<4, 1>(4);
      aa.select<2, 1>(0) += aa.select<2, 1>(2);
    } else if constexpr (pixelPerGroupShift == 0) {
      aa.select<16, 1>(0) = slm_block_load<float, 16>(0);
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
      aa.select<4, 1>(0) += aa.select<4, 1>(4);
      aa.select<2, 1>(0) += aa.select<2, 1>(2);
      aa.select<1, 1>(0) += aa.select<1, 1>(1);
    }

    __ESIMD_ENS::lsc_block_store<
      fp16,
      pixelPerGroup,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((fp16*)c + outputOffset, aa.select<pixelPerGroup, 1>(0));
  }
}


template<uint32_t K_DIM, uint32_t pixelPerGroupShift>
ESIMD_INLINE void MOE_down_GEMV_Int4Weight_FP16InOutNx16Temp_largeGRF_block_ppg8_8T_32g_768(uint8_t* b, uint8_t* c,
  int64_t* indexs,
  uint64_t* weight_addrs,
  uint64_t* scale_addrs,
  nd_item<2>& ndi) {
  constexpr uint32_t pixelPerGroup = 1 << pixelPerGroupShift;
  constexpr uint32_t quantPerGroup = K_DIM / GROUP * pixelPerGroup;
  constexpr uint32_t sumThreads = 1;
  constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  constexpr uint32_t baseOffsetInc8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
  constexpr uint32_t K_DIM_DIV_4096 = (K_DIM + 4095) / 4096;
  constexpr uint32_t K_DIM_MOD_4096 = (K_DIM - (K_DIM_DIV_4096-1)*4096);
  constexpr uint32_t K_DIM_REDUCE_T = (K_DIM_MOD_4096 +  (4096 / 8)-1) / (4096 / 8); // 8 threads, 512 ele per thread in total 4096
  __ESIMD_NS::slm_init(16 * sizeof(float));
  int hh = ndi.get_local_id(1); // [0, 64)
  int h = ndi.get_group(1); // [0, 256)
  int eid = ndi.get_group(0); // [0, 256)
  int rowSize = ndi.get_group_range(1) * pixelPerGroup;
  int offsetABase = (h * pixelPerGroup * K_DIM + hh * 8 * 8 * 4 * 2) >> 1;
  int offsetQuanBase = /*rowSize * K_DIM / 2 +*/ h * quantPerGroup * sizeof(fp16) + hh * 512 / GROUP * sizeof(fp16);
  int offsetB = hh * 128 * 2 * 2 * sizeof(fp16) + eid * K_DIM * sizeof(fp16);
  int outputOffset = pixelPerGroup * h + rowSize * eid;
  int offsetSLMThread = hh * 2 * 64 * sizeof(float);
  simd<unsigned char, 256> aaa;
  simd<fp16, 80> quant;
  simd<fp16, 2560> bb;
  simd<float, 8 * 16 * 4> aa;
  simd<float, 16> cc(0.0f);
  uint32_t offsetA;
  uint32_t offsetQuan;
  offsetA = offsetABase;
  offsetQuan = offsetQuanBase;

  uint8_t* a = (uint8_t *)(weight_addrs[indexs[eid]]);
  uint8_t* d = (uint8_t *)(scale_addrs[indexs[eid]]);

#pragma unroll
  for (int k = 0; k < K_DIM_DIV_4096; k++) {
    if (k != K_DIM_DIV_4096-1 || hh < K_DIM_REDUCE_T)
    {
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*1) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 1 * 128 * sizeof(fp16));
      // special for 768
      if (hh == 1)
      {
        bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*2) = 0;
        bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*3) = 0;
      }
      else
      {
        bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*2) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 2 * 128 * sizeof(fp16));
        bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*3) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 3 * 128 * sizeof(fp16));
      }

    }
    else
    {
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*1) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*2) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*3) = 0;
    }

    offsetB += 4096 * sizeof(fp16);
  }

  for (int n = 0; n < pixelPerGroup; n++) {
    cc = 0.0f;
    offsetQuan = offsetQuanBase + n * K_DIM / GROUP * sizeof(fp16);
#pragma unroll
    for (int k = 0; k < K_DIM_DIV_4096-1; k++) {
      quant.template bit_cast_view<unsigned char>().template select<512 / GROUP * 2, 1>(512 / GROUP * 2 * k) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        512 / GROUP * 2,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuan);
      offsetQuan += 4096 / GROUP * sizeof(fp16);
    }
    if (hh < K_DIM_REDUCE_T)
    {
        // special for 768
      if (hh == 1)
      {
        quant.template bit_cast_view<unsigned char>().template select<256 / GROUP * 2, 1>(512 / GROUP * 2 * (K_DIM_DIV_4096-1)) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256 / GROUP * 2,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuan);
      }
      else
      {
        quant.template bit_cast_view<unsigned char>().template select<512 / GROUP * 2, 1>(512 / GROUP * 2 * (K_DIM_DIV_4096-1)) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          512 / GROUP * 2,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuan);
      }
    }
    else
    {
      quant.template bit_cast_view<unsigned char>().template select<32, 1>(32 * (K_DIM_DIV_4096-1)) = 0;
    }
    offsetQuan += 4096 / GROUP * sizeof(fp16);

    offsetA = offsetABase + n * K_DIM / 2;

#pragma unroll
    for (int k = 0; k < K_DIM_DIV_4096; k++) {
      if (k != K_DIM_DIV_4096-1 || hh < K_DIM_REDUCE_T)
      {
        simd<float, 512 / GROUP> fp32Q = quant.select<512 / GROUP, 1>(512 / GROUP * k);
         // special for 768
        if (hh == 1)
        {
          aaa.template bit_cast_view<unsigned char>().template select<128, 1>(0) =
            __ESIMD_ENS::lsc_block_load<
            uint8_t,
            128,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offsetA);
          fp32Q.select<256 / GROUP, 1>(256 / GROUP) = 0.0f;
        }
        else
        {
          aaa.template bit_cast_view<unsigned char>().template select<256, 1>(0) =
            __ESIMD_ENS::lsc_block_load<
            uint8_t,
            256,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offsetA);
        }

#pragma unroll
        for (int kk = 0; kk < 16; kk++) {
          aa.select<16, 2>(32 * kk) = aaa.select<16, 1>(16 * kk) & 0xf;
          aa.select<16, 2>(32 * kk + 1) = aaa.select<16, 1>(16 * kk) >> 4;
        }

        aa = aa - 8.0f;
#pragma unroll
        for (int kk = 0; kk < 512 / GROUP; kk++) {
          aa.select<GROUP, 1>(GROUP * kk) = fp32Q[kk] * aa.select<GROUP, 1>(GROUP * kk);
        }
#pragma unroll
        for (int kk = 0; kk < 32; kk++) {
          cc += aa.select<16, 1>(16 * kk) * bb.select<16, 1>(16 * kk + 128 * 2 * 2 * k); // note: 128 * 2 easy to have mistake!
        }
      }

      if (k == K_DIM_DIV_4096-1)
      {
        offsetA += K_DIM_REDUCE_T * 512 / 2;
      }
      else
      {
        offsetA += 2048;
      }
    }

    cc.select<8, 1>(0) += cc.select<8, 1>(8);
    cc.select<4, 1>(0) += cc.select<4, 1>(4);
    cc.select<2, 1>(0) += cc.select<2, 1>(2);
    simd<float, 1> slmAccumulationTemp = cc[0] + cc[1];
    uint32_t slmAccumulationOffset = (hh * pixelPerGroup + n) * sizeof(float);
    slm_block_store<float, 1>(slmAccumulationOffset, slmAccumulationTemp);
  }
  barrier();
  if (hh < sumThreads) {
if constexpr (pixelPerGroupShift == 3) {
#pragma unroll
      for (int k = 0; k < 2; k++) {
        aa.select<64, 1>(64 * k) = slm_block_load<float, 64>(64 * k * sizeof(float));
      }
#pragma unroll
      for (int k = 1; k < 8; k++) {
        aa.select<16, 1>(0) += aa.select<16, 1>(16 * k);
      }
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
    } else if constexpr (pixelPerGroupShift == 2) {
      aa.select<64, 1>(0) = slm_block_load<float, 64>(0);
#pragma unroll
      for (int k = 1; k < 4; k++) {
        aa.select<16, 1>(0) += aa.select<16, 1>(16 * k);
      }
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
      aa.select<4, 1>(0) += aa.select<4, 1>(4);
    } else if constexpr (pixelPerGroupShift == 1) {
      aa.select<32, 1>(0) = slm_block_load<float, 32>(0);
      aa.select<16, 1>(0) += aa.select<16, 1>(16 * 1);
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
      aa.select<4, 1>(0) += aa.select<4, 1>(4);
      aa.select<2, 1>(0) += aa.select<2, 1>(2);
    } else if constexpr (pixelPerGroupShift == 0) {
      aa.select<16, 1>(0) = slm_block_load<float, 16>(0);
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
      aa.select<4, 1>(0) += aa.select<4, 1>(4);
      aa.select<2, 1>(0) += aa.select<2, 1>(2);
      aa.select<1, 1>(0) += aa.select<1, 1>(1);
    }

    __ESIMD_ENS::lsc_block_store<
      fp16,
      pixelPerGroup,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((fp16*)c + outputOffset, aa.select<pixelPerGroup, 1>(0));
  }
}


#undef GROUP


#define GROUP 128

template<uint32_t K_DIM, uint32_t pixelPerGroupShift>
ESIMD_INLINE void GEMV_Int4Weight_FP16InOutNx16Temp_largeGRF_block_ppg8_8T_128g_768(uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* d, nd_item<1>& ndi) {
  constexpr uint32_t pixelPerGroup = 1 << pixelPerGroupShift;
  constexpr uint32_t quantPerGroup = K_DIM / GROUP * pixelPerGroup;
  constexpr uint32_t sumThreads = 1;
  constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  constexpr uint32_t baseOffsetInc8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
  constexpr uint32_t K_DIM_DIV_4096 = (K_DIM + 4095) / 4096;
  constexpr uint32_t K_DIM_MOD_4096 = (K_DIM - (K_DIM_DIV_4096-1)*4096);
  constexpr uint32_t K_DIM_REDUCE_T = (K_DIM_MOD_4096 +  (4096 / 8)-1) / (4096 / 8); // 8 threads, 512 ele per thread in total 4096
  __ESIMD_NS::slm_init(16 * sizeof(float));
  int hh = ndi.get_local_id(0); // [0, 64)
  int h = ndi.get_group(0); // [0, 256)
  int rowSize = ndi.get_group_range(0) * pixelPerGroup;
  int offsetABase = (h * pixelPerGroup * K_DIM + hh * 8 * 8 * 4 * 2) >> 1;
  int offsetQuanBase = /*rowSize * K_DIM / 2 +*/ h * quantPerGroup * sizeof(fp16) + hh * 512 / GROUP * sizeof(fp16);
  int offsetB = hh * 128 * 2 * 2 * sizeof(fp16);
  int outputOffset = pixelPerGroup * h;
  int offsetSLMThread = hh * 2 * 64 * sizeof(float);
  simd<unsigned char, 256> aaa;
  simd<fp16, 80> quant;
  simd<fp16, 2560> bb;
  simd<float, 8 * 16 * 4> aa;
  simd<float, 16> cc(0.0f);
  uint32_t offsetA;
  uint32_t offsetQuan;
  offsetA = offsetABase;
  offsetQuan = offsetQuanBase;

#pragma unroll
  for (int k = 0; k < K_DIM_DIV_4096; k++) {
    if (k != K_DIM_DIV_4096-1 || hh < K_DIM_REDUCE_T)
    {
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*1) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 1 * 128 * sizeof(fp16));
      // special for 768
      if (hh == 1)
      {
        bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*2) = 0;
        bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*3) = 0;
      }
      else
      {
        bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*2) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 2 * 128 * sizeof(fp16));
        bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*3) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 3 * 128 * sizeof(fp16));
      }

    }
    else
    {
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*1) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*2) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*3) = 0;
    }

    offsetB += 4096 * sizeof(fp16);
  }

  for (int n = 0; n < pixelPerGroup; n++) {
    cc = 0.0f;
    offsetQuan = offsetQuanBase + n * K_DIM / GROUP * sizeof(fp16);
#pragma unroll
    for (int k = 0; k < K_DIM_DIV_4096-1; k++) {
      quant.template bit_cast_view<unsigned char>().template select<512 / GROUP * 2, 1>(512 / GROUP * 2 * k) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        512 / GROUP * 2,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuan);
      offsetQuan += 4096 / GROUP * sizeof(fp16);
    }
    if (hh < K_DIM_REDUCE_T)
    {
        // special for 768
      if (hh == 1)
      {
        quant.template bit_cast_view<unsigned char>().template select<256 / GROUP * 2, 1>(512 / GROUP * 2 * (K_DIM_DIV_4096-1)) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256 / GROUP * 2,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuan);
      }
      else
      {
        quant.template bit_cast_view<unsigned char>().template select<512 / GROUP * 2, 1>(512 / GROUP * 2 * (K_DIM_DIV_4096-1)) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          512 / GROUP * 2,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuan);
      }
    }
    else
    {
      quant.template bit_cast_view<unsigned char>().template select<32, 1>(32 * (K_DIM_DIV_4096-1)) = 0;
    }
    offsetQuan += 4096 / GROUP * sizeof(fp16);

    offsetA = offsetABase + n * K_DIM / 2;

#pragma unroll
    for (int k = 0; k < K_DIM_DIV_4096; k++) {
      if (k != K_DIM_DIV_4096-1 || hh < K_DIM_REDUCE_T)
      {
        simd<float, 512 / GROUP> fp32Q = quant.select<512 / GROUP, 1>(512 / GROUP * k);
         // special for 768
        if (hh == 1)
        {
          aaa.template bit_cast_view<unsigned char>().template select<128, 1>(0) =
            __ESIMD_ENS::lsc_block_load<
            uint8_t,
            128,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offsetA);
          fp32Q.select<256 / GROUP, 1>(256 / GROUP) = 0.0f;
        }
        else
        {
          aaa.template bit_cast_view<unsigned char>().template select<256, 1>(0) =
            __ESIMD_ENS::lsc_block_load<
            uint8_t,
            256,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offsetA);
        }

#pragma unroll
        for (int kk = 0; kk < 16; kk++) {
          aa.select<16, 2>(32 * kk) = aaa.select<16, 1>(16 * kk) & 0xf;
          aa.select<16, 2>(32 * kk + 1) = aaa.select<16, 1>(16 * kk) >> 4;
        }

        aa = aa - 8.0f;
#pragma unroll
        for (int kk = 0; kk < 512 / GROUP; kk++) {
          aa.select<GROUP, 1>(GROUP * kk) = fp32Q[kk] * aa.select<GROUP, 1>(GROUP * kk);
        }
#pragma unroll
        for (int kk = 0; kk < 32; kk++) {
          cc += aa.select<16, 1>(16 * kk) * bb.select<16, 1>(16 * kk + 128 * 2 * 2 * k); // note: 128 * 2 easy to have mistake!
        }
      }

      if (k == K_DIM_DIV_4096-1)
      {
        offsetA += K_DIM_REDUCE_T * 512 / 2;
      }
      else
      {
        offsetA += 2048;
      }
    }

    cc.select<8, 1>(0) += cc.select<8, 1>(8);
    cc.select<4, 1>(0) += cc.select<4, 1>(4);
    cc.select<2, 1>(0) += cc.select<2, 1>(2);
    simd<float, 1> slmAccumulationTemp = cc[0] + cc[1];
    uint32_t slmAccumulationOffset = (hh * pixelPerGroup + n) * sizeof(float);
    slm_block_store<float, 1>(slmAccumulationOffset, slmAccumulationTemp);
  }
  barrier();
  if (hh < sumThreads) {
if constexpr (pixelPerGroupShift == 3) {
#pragma unroll
      for (int k = 0; k < 2; k++) {
        aa.select<64, 1>(64 * k) = slm_block_load<float, 64>(64 * k * sizeof(float));
      }
#pragma unroll
      for (int k = 1; k < 8; k++) {
        aa.select<16, 1>(0) += aa.select<16, 1>(16 * k);
      }
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
    } else if constexpr (pixelPerGroupShift == 2) {
      aa.select<64, 1>(0) = slm_block_load<float, 64>(0);
#pragma unroll
      for (int k = 1; k < 4; k++) {
        aa.select<16, 1>(0) += aa.select<16, 1>(16 * k);
      }
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
      aa.select<4, 1>(0) += aa.select<4, 1>(4);
    } else if constexpr (pixelPerGroupShift == 1) {
      aa.select<32, 1>(0) = slm_block_load<float, 32>(0);
      aa.select<16, 1>(0) += aa.select<16, 1>(16 * 1);
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
      aa.select<4, 1>(0) += aa.select<4, 1>(4);
      aa.select<2, 1>(0) += aa.select<2, 1>(2);
    } else if constexpr (pixelPerGroupShift == 0) {
      aa.select<16, 1>(0) = slm_block_load<float, 16>(0);
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
      aa.select<4, 1>(0) += aa.select<4, 1>(4);
      aa.select<2, 1>(0) += aa.select<2, 1>(2);
      aa.select<1, 1>(0) += aa.select<1, 1>(1);
    }

    __ESIMD_ENS::lsc_block_store<
      fp16,
      pixelPerGroup,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((fp16*)c + outputOffset, aa.select<pixelPerGroup, 1>(0));
  }
}

template<uint32_t K_DIM, uint32_t pixelPerGroupShift>
ESIMD_INLINE void MOE_up_or_gate_GEMV_Int4Weight_FP16InOutNx16Temp_largeGRF_block_ppg8_8T_128g(uint8_t* b, uint8_t* c,
  int64_t* indexs,
  uint64_t* weight_addrs,
  uint64_t* scale_addrs,
  nd_item<2>& ndi) 
{
  constexpr uint32_t pixelPerGroup = 1 << pixelPerGroupShift;
  constexpr uint32_t quantPerGroup = K_DIM / GROUP * pixelPerGroup;
  constexpr uint32_t sumThreads = 1;
  constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  constexpr uint32_t baseOffsetInc8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
  constexpr uint32_t K_DIM_DIV_4096 = (K_DIM + 4095) / 4096;
  constexpr uint32_t K_DIM_MOD_4096 = (K_DIM - (K_DIM_DIV_4096-1)*4096);
  constexpr uint32_t K_DIM_REDUCE_T = K_DIM_MOD_4096 / (4096 / 8); // 8 threads, 512 ele per thread in total 4096
  __ESIMD_NS::slm_init(16 * sizeof(float));
  int hh = ndi.get_local_id(1); // [0, 64)
  int h = ndi.get_group(1); // [0, 256)
  int eid = ndi.get_group(0); // [0, 256)
  int rowSize = ndi.get_group_range(1) * pixelPerGroup;
  int offsetABase = (h * pixelPerGroup * K_DIM + hh * 8 * 8 * 4 * 2) >> 1;
  int offsetQuanBase = /*rowSize * K_DIM / 2 +*/ h * quantPerGroup * sizeof(fp16) + hh * 512 / GROUP * sizeof(fp16);
  int offsetB = hh * 128 * 2 * 2 * sizeof(fp16);
  int outputOffset = pixelPerGroup * h + rowSize * eid;
  int offsetSLMThread = hh * 2 * 64 * sizeof(float);
  simd<unsigned char, 256> aaa;
  simd<fp16, 80> quant;
  simd<fp16, 2560> bb;
  simd<float, 8 * 16 * 4> aa;
  simd<float, 16> cc(0.0f);
  uint32_t offsetA;
  uint32_t offsetQuan;
  offsetA = offsetABase;
  offsetQuan = offsetQuanBase;

  uint8_t* a = (uint8_t *)(weight_addrs[indexs[eid]]);
  uint8_t* d = (uint8_t *)(scale_addrs[indexs[eid]]);

#pragma unroll
  for (int k = 0; k < K_DIM_DIV_4096; k++) {
    if (k != K_DIM_DIV_4096-1 || hh < K_DIM_REDUCE_T)
    {
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*1) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 1 * 128 * sizeof(fp16));
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*2) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 2 * 128 * sizeof(fp16));
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*3) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 3 * 128 * sizeof(fp16));
    }
    else
    {
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*1) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*2) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*3) = 0;
    }

    offsetB += 4096 * sizeof(fp16);
  }

  for (int n = 0; n < pixelPerGroup; n++) {
    cc = 0.0f;
    offsetQuan = offsetQuanBase + n * K_DIM / GROUP * sizeof(fp16);
#pragma unroll
    for (int k = 0; k < K_DIM_DIV_4096-1; k++) {
      quant.template bit_cast_view<unsigned char>().template select<512 / GROUP * 2, 1>(512 / GROUP * 2 * k) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        512 / GROUP * 2,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuan);
      offsetQuan += 4096 / GROUP * sizeof(fp16);
    }
    if (hh < K_DIM_REDUCE_T)
    {
      quant.template bit_cast_view<unsigned char>().template select<512 / GROUP * 2, 1>(512 / GROUP * 2 * (K_DIM_DIV_4096-1)) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        512 / GROUP * 2,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuan);
    }
    else
    {
      quant.template bit_cast_view<unsigned char>().template select<32, 1>(32 * (K_DIM_DIV_4096-1)) = 0;
    }
    offsetQuan += 4096 / GROUP * sizeof(fp16);

    offsetA = offsetABase + n * K_DIM / 2;

#pragma unroll
    for (int k = 0; k < K_DIM_DIV_4096; k++) {
      if (k != K_DIM_DIV_4096-1 || hh < K_DIM_REDUCE_T)
      {
        simd<float, 512 / GROUP> fp32Q = quant.select<512 / GROUP, 1>(512 / GROUP * k);
        aaa.template bit_cast_view<unsigned char>().template select<256, 1>(0) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offsetA);

#pragma unroll
        for (int kk = 0; kk < 16; kk++) {
          aa.select<16, 2>(32 * kk) = aaa.select<16, 1>(16 * kk) & 0xf;
          aa.select<16, 2>(32 * kk + 1) = aaa.select<16, 1>(16 * kk) >> 4;
        }

        aa = aa - 8.0f;
#pragma unroll
        for (int kk = 0; kk < 512 / GROUP; kk++) {
          aa.select<GROUP, 1>(GROUP * kk) = fp32Q[kk] * aa.select<GROUP, 1>(GROUP * kk);
        }
#pragma unroll
        for (int kk = 0; kk < 32; kk++) {
          cc += aa.select<16, 1>(16 * kk) * bb.select<16, 1>(16 * kk + 128 * 2 * 2 * k); // note: 128 * 2 easy to have mistake!
        }
      }

      if (k == K_DIM_DIV_4096-1)
      {
        offsetA += K_DIM_REDUCE_T * 512 / 2;
      }
      else
      {
        offsetA += 2048;
      }
    }

    cc.select<8, 1>(0) += cc.select<8, 1>(8);
    cc.select<4, 1>(0) += cc.select<4, 1>(4);
    cc.select<2, 1>(0) += cc.select<2, 1>(2);
    simd<float, 1> slmAccumulationTemp = cc[0] + cc[1];
    uint32_t slmAccumulationOffset = (hh * pixelPerGroup + n) * sizeof(float);
    slm_block_store<float, 1>(slmAccumulationOffset, slmAccumulationTemp);
  }
  barrier();
  if (hh < sumThreads) {
if constexpr (pixelPerGroupShift == 3) {
#pragma unroll
      for (int k = 0; k < 2; k++) {
        aa.select<64, 1>(64 * k) = slm_block_load<float, 64>(64 * k * sizeof(float));
      }
#pragma unroll
      for (int k = 1; k < 8; k++) {
        aa.select<16, 1>(0) += aa.select<16, 1>(16 * k);
      }
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
    } else if constexpr (pixelPerGroupShift == 2) {
      aa.select<64, 1>(0) = slm_block_load<float, 64>(0);
#pragma unroll
      for (int k = 1; k < 4; k++) {
        aa.select<16, 1>(0) += aa.select<16, 1>(16 * k);
      }
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
      aa.select<4, 1>(0) += aa.select<4, 1>(4);
    } else if constexpr (pixelPerGroupShift == 1) {
      aa.select<32, 1>(0) = slm_block_load<float, 32>(0);
      aa.select<16, 1>(0) += aa.select<16, 1>(16 * 1);
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
      aa.select<4, 1>(0) += aa.select<4, 1>(4);
      aa.select<2, 1>(0) += aa.select<2, 1>(2);
    } else if constexpr (pixelPerGroupShift == 0) {
      aa.select<16, 1>(0) = slm_block_load<float, 16>(0);
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
      aa.select<4, 1>(0) += aa.select<4, 1>(4);
      aa.select<2, 1>(0) += aa.select<2, 1>(2);
      aa.select<1, 1>(0) += aa.select<1, 1>(1);
    }

    __ESIMD_ENS::lsc_block_store<
      fp16,
      pixelPerGroup,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((fp16*)c + outputOffset, aa.select<pixelPerGroup, 1>(0));
  }
}


template<uint32_t K_DIM, uint32_t pixelPerGroupShift>
ESIMD_INLINE void MOE_down_GEMV_Int4Weight_FP16InOutNx16Temp_largeGRF_block_ppg8_8T_128g_768(uint8_t* b, uint8_t* c,
  int64_t* indexs,
  uint64_t* weight_addrs,
  uint64_t* scale_addrs,
  nd_item<2>& ndi) {
  constexpr uint32_t pixelPerGroup = 1 << pixelPerGroupShift;
  constexpr uint32_t quantPerGroup = K_DIM / GROUP * pixelPerGroup;
  constexpr uint32_t sumThreads = 1;
  constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  constexpr uint32_t baseOffsetInc8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
  constexpr uint32_t K_DIM_DIV_4096 = (K_DIM + 4095) / 4096;
  constexpr uint32_t K_DIM_MOD_4096 = (K_DIM - (K_DIM_DIV_4096-1)*4096);
  constexpr uint32_t K_DIM_REDUCE_T = (K_DIM_MOD_4096 +  (4096 / 8)-1) / (4096 / 8); // 8 threads, 512 ele per thread in total 4096
  __ESIMD_NS::slm_init(16 * sizeof(float));
  int hh = ndi.get_local_id(1); // [0, 64)
  int h = ndi.get_group(1); // [0, 256)
  int eid = ndi.get_group(0); // [0, 256)
  int rowSize = ndi.get_group_range(1) * pixelPerGroup;
  int offsetABase = (h * pixelPerGroup * K_DIM + hh * 8 * 8 * 4 * 2) >> 1;
  int offsetQuanBase = /*rowSize * K_DIM / 2 +*/ h * quantPerGroup * sizeof(fp16) + hh * 512 / GROUP * sizeof(fp16);
  int offsetB = hh * 128 * 2 * 2 * sizeof(fp16) + eid * K_DIM * sizeof(fp16);
  int outputOffset = pixelPerGroup * h + rowSize * eid;
  int offsetSLMThread = hh * 2 * 64 * sizeof(float);
  simd<unsigned char, 256> aaa;
  simd<fp16, 80> quant;
  simd<fp16, 2560> bb;
  simd<float, 8 * 16 * 4> aa;
  simd<float, 16> cc(0.0f);
  uint32_t offsetA;
  uint32_t offsetQuan;
  offsetA = offsetABase;
  offsetQuan = offsetQuanBase;

  uint8_t* a = (uint8_t *)(weight_addrs[indexs[eid]]);
  uint8_t* d = (uint8_t *)(scale_addrs[indexs[eid]]);

#pragma unroll
  for (int k = 0; k < K_DIM_DIV_4096; k++) {
    if (k != K_DIM_DIV_4096-1 || hh < K_DIM_REDUCE_T)
    {
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*1) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 1 * 128 * sizeof(fp16));
      // special for 768
      if (hh == 1)
      {
        bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*2) = 0;
        bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*3) = 0;
      }
      else
      {
        bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*2) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 2 * 128 * sizeof(fp16));
        bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*3) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 3 * 128 * sizeof(fp16));
      }

    }
    else
    {
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*1) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*2) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*3) = 0;
    }

    offsetB += 4096 * sizeof(fp16);
  }

  for (int n = 0; n < pixelPerGroup; n++) {
    cc = 0.0f;
    offsetQuan = offsetQuanBase + n * K_DIM / GROUP * sizeof(fp16);
#pragma unroll
    for (int k = 0; k < K_DIM_DIV_4096-1; k++) {
      quant.template bit_cast_view<unsigned char>().template select<512 / GROUP * 2, 1>(512 / GROUP * 2 * k) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        512 / GROUP * 2,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuan);
      offsetQuan += 4096 / GROUP * sizeof(fp16);
    }
    if (hh < K_DIM_REDUCE_T)
    {
        // special for 768
      if (hh == 1)
      {
        quant.template bit_cast_view<unsigned char>().template select<256 / GROUP * 2, 1>(512 / GROUP * 2 * (K_DIM_DIV_4096-1)) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256 / GROUP * 2,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuan);
      }
      else
      {
        quant.template bit_cast_view<unsigned char>().template select<512 / GROUP * 2, 1>(512 / GROUP * 2 * (K_DIM_DIV_4096-1)) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          512 / GROUP * 2,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuan);
      }
    }
    else
    {
      quant.template bit_cast_view<unsigned char>().template select<32, 1>(32 * (K_DIM_DIV_4096-1)) = 0;
    }
    offsetQuan += 4096 / GROUP * sizeof(fp16);

    offsetA = offsetABase + n * K_DIM / 2;

#pragma unroll
    for (int k = 0; k < K_DIM_DIV_4096; k++) {
      if (k != K_DIM_DIV_4096-1 || hh < K_DIM_REDUCE_T)
      {
        simd<float, 512 / GROUP> fp32Q = quant.select<512 / GROUP, 1>(512 / GROUP * k);
         // special for 768
        if (hh == 1)
        {
          aaa.template bit_cast_view<unsigned char>().template select<128, 1>(0) =
            __ESIMD_ENS::lsc_block_load<
            uint8_t,
            128,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offsetA);
          fp32Q.select<256 / GROUP, 1>(256 / GROUP) = 0.0f;
        }
        else
        {
          aaa.template bit_cast_view<unsigned char>().template select<256, 1>(0) =
            __ESIMD_ENS::lsc_block_load<
            uint8_t,
            256,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offsetA);
        }

#pragma unroll
        for (int kk = 0; kk < 16; kk++) {
          aa.select<16, 2>(32 * kk) = aaa.select<16, 1>(16 * kk) & 0xf;
          aa.select<16, 2>(32 * kk + 1) = aaa.select<16, 1>(16 * kk) >> 4;
        }

        aa = aa - 8.0f;
#pragma unroll
        for (int kk = 0; kk < 512 / GROUP; kk++) {
          aa.select<GROUP, 1>(GROUP * kk) = fp32Q[kk] * aa.select<GROUP, 1>(GROUP * kk);
        }
#pragma unroll
        for (int kk = 0; kk < 32; kk++) {
          cc += aa.select<16, 1>(16 * kk) * bb.select<16, 1>(16 * kk + 128 * 2 * 2 * k); // note: 128 * 2 easy to have mistake!
        }
      }

      if (k == K_DIM_DIV_4096-1)
      {
        offsetA += K_DIM_REDUCE_T * 512 / 2;
      }
      else
      {
        offsetA += 2048;
      }
    }

    cc.select<8, 1>(0) += cc.select<8, 1>(8);
    cc.select<4, 1>(0) += cc.select<4, 1>(4);
    cc.select<2, 1>(0) += cc.select<2, 1>(2);
    simd<float, 1> slmAccumulationTemp = cc[0] + cc[1];
    uint32_t slmAccumulationOffset = (hh * pixelPerGroup + n) * sizeof(float);
    slm_block_store<float, 1>(slmAccumulationOffset, slmAccumulationTemp);
  }
  barrier();
  if (hh < sumThreads) {
if constexpr (pixelPerGroupShift == 3) {
#pragma unroll
      for (int k = 0; k < 2; k++) {
        aa.select<64, 1>(64 * k) = slm_block_load<float, 64>(64 * k * sizeof(float));
      }
#pragma unroll
      for (int k = 1; k < 8; k++) {
        aa.select<16, 1>(0) += aa.select<16, 1>(16 * k);
      }
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
    } else if constexpr (pixelPerGroupShift == 2) {
      aa.select<64, 1>(0) = slm_block_load<float, 64>(0);
#pragma unroll
      for (int k = 1; k < 4; k++) {
        aa.select<16, 1>(0) += aa.select<16, 1>(16 * k);
      }
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
      aa.select<4, 1>(0) += aa.select<4, 1>(4);
    } else if constexpr (pixelPerGroupShift == 1) {
      aa.select<32, 1>(0) = slm_block_load<float, 32>(0);
      aa.select<16, 1>(0) += aa.select<16, 1>(16 * 1);
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
      aa.select<4, 1>(0) += aa.select<4, 1>(4);
      aa.select<2, 1>(0) += aa.select<2, 1>(2);
    } else if constexpr (pixelPerGroupShift == 0) {
      aa.select<16, 1>(0) = slm_block_load<float, 16>(0);
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
      aa.select<4, 1>(0) += aa.select<4, 1>(4);
      aa.select<2, 1>(0) += aa.select<2, 1>(2);
      aa.select<1, 1>(0) += aa.select<1, 1>(1);
    }

    __ESIMD_ENS::lsc_block_store<
      fp16,
      pixelPerGroup,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((fp16*)c + outputOffset, aa.select<pixelPerGroup, 1>(0));
  }
}

#undef GROUP



// =======================================================================================

#define GROUP 128
template<uint32_t K_DIM, uint32_t pixelPerGroupShift>
ESIMD_INLINE void GEMV_bias_Int4Weight_FP16InOutNx16Temp_largeGRF_block_ppg8_8T_128g(uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* d, uint8_t* bias, nd_item<1>& ndi) {
  constexpr uint32_t pixelPerGroup = 1 << pixelPerGroupShift;
  constexpr uint32_t quantPerGroup = K_DIM / GROUP * pixelPerGroup;
  constexpr uint32_t sumThreads = 1;
  constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  constexpr uint32_t baseOffsetInc8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
  constexpr uint32_t K_DIM_DIV_4096 = (K_DIM + 4095) / 4096;
  constexpr uint32_t K_DIM_MOD_4096 = (K_DIM - (K_DIM_DIV_4096-1)*4096);
  constexpr uint32_t K_DIM_REDUCE_T = K_DIM_MOD_4096 / (4096 / 8); // 8 threads, 512 ele per thread in total 4096
  __ESIMD_NS::slm_init(16 * sizeof(float));
  int hh = ndi.get_local_id(0); // [0, 64)
  int h = ndi.get_group(0); // [0, 256)
  int rowSize = ndi.get_group_range(0) * pixelPerGroup;
  int offsetABase = (h * pixelPerGroup * K_DIM + hh * 8 * 8 * 4 * 2) >> 1;
  int offsetQuanBase = /*rowSize * K_DIM / 2 +*/ h * quantPerGroup * sizeof(fp16) + hh * 512 / GROUP * sizeof(fp16);
  int offsetB = hh * 128 * 2 * 2 * sizeof(fp16);
  int outputOffset = pixelPerGroup * h;
  int offsetSLMThread = hh * 2 * 64 * sizeof(float);
  simd<unsigned char, 256> aaa;
  simd<fp16, 80> quant;
  simd<fp16, 2560> bb;
  simd<float, 8 * 16 * 4> aa;
  simd<float, 16> cc(0.0f);
  uint32_t offsetA;
  uint32_t offsetQuan;
  offsetA = offsetABase;
  offsetQuan = offsetQuanBase;

#pragma unroll
  for (int k = 0; k < K_DIM_DIV_4096; k++) {
    if (k != K_DIM_DIV_4096-1 || hh < K_DIM_REDUCE_T)
    {
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*1) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 1 * 128 * sizeof(fp16));
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*2) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 2 * 128 * sizeof(fp16));
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*3) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 3 * 128 * sizeof(fp16));
    }
    else
    {
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*1) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*2) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*3) = 0;
    }

    offsetB += 4096 * sizeof(fp16);
  }

  for (int n = 0; n < pixelPerGroup; n++) {
    cc = 0.0f;
    offsetQuan = offsetQuanBase + n * K_DIM / GROUP * sizeof(fp16);
#pragma unroll
    for (int k = 0; k < K_DIM_DIV_4096-1; k++) {
      quant.template bit_cast_view<unsigned char>().template select<512 / GROUP * 2, 1>(512 / GROUP * 2 * k) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        512 / GROUP * 2,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuan);
      offsetQuan += 4096 / GROUP * sizeof(fp16);
    }
    if (hh < K_DIM_REDUCE_T)
    {
      quant.template bit_cast_view<unsigned char>().template select<512 / GROUP * 2, 1>(512 / GROUP * 2 * (K_DIM_DIV_4096-1)) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        512 / GROUP * 2,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuan);
    }
    else
    {
      quant.template bit_cast_view<unsigned char>().template select<32, 1>(32 * (K_DIM_DIV_4096-1)) = 0;
    }
    offsetQuan += 4096 / GROUP * sizeof(fp16);

    offsetA = offsetABase + n * K_DIM / 2;

#pragma unroll
    for (int k = 0; k < K_DIM_DIV_4096; k++) {
      if (k != K_DIM_DIV_4096-1 || hh < K_DIM_REDUCE_T)
      {
        simd<float, 512 / GROUP> fp32Q = quant.select<512 / GROUP, 1>(512 / GROUP * k);
        aaa.template bit_cast_view<unsigned char>().template select<256, 1>(0) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offsetA);

#pragma unroll
        for (int kk = 0; kk < 16; kk++) {
          aa.select<16, 2>(32 * kk) = aaa.select<16, 1>(16 * kk) & 0xf;
          aa.select<16, 2>(32 * kk + 1) = aaa.select<16, 1>(16 * kk) >> 4;
        }

        aa = aa - 8.0f;
#pragma unroll
        for (int kk = 0; kk < 512 / GROUP; kk++) {
          aa.select<GROUP, 1>(GROUP * kk) = fp32Q[kk] * aa.select<GROUP, 1>(GROUP * kk);
        }
#pragma unroll
        for (int kk = 0; kk < 32; kk++) {
          cc += aa.select<16, 1>(16 * kk) * bb.select<16, 1>(16 * kk + 128 * 2 * 2 * k); // note: 128 * 2 easy to have mistake!
        }
      }

      if (k == K_DIM_DIV_4096-1)
      {
        offsetA += K_DIM_REDUCE_T * 512 / 2;
      }
      else
      {
        offsetA += 2048;
      }
    }

    cc.select<8, 1>(0) += cc.select<8, 1>(8);
    cc.select<4, 1>(0) += cc.select<4, 1>(4);
    cc.select<2, 1>(0) += cc.select<2, 1>(2);
    simd<float, 1> slmAccumulationTemp = cc[0] + cc[1];
    uint32_t slmAccumulationOffset = (hh * pixelPerGroup + n) * sizeof(float);
    slm_block_store<float, 1>(slmAccumulationOffset, slmAccumulationTemp);
  }
  barrier();
  if (hh < sumThreads) {
if constexpr (pixelPerGroupShift == 3) {
#pragma unroll
      for (int k = 0; k < 2; k++) {
        aa.select<64, 1>(64 * k) = slm_block_load<float, 64>(64 * k * sizeof(float));
      }
#pragma unroll
      for (int k = 1; k < 8; k++) {
        aa.select<16, 1>(0) += aa.select<16, 1>(16 * k);
      }
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
    } else if constexpr (pixelPerGroupShift == 2) {
      aa.select<64, 1>(0) = slm_block_load<float, 64>(0);
#pragma unroll
      for (int k = 1; k < 4; k++) {
        aa.select<16, 1>(0) += aa.select<16, 1>(16 * k);
      }
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
      aa.select<4, 1>(0) += aa.select<4, 1>(4);
    } else if constexpr (pixelPerGroupShift == 1) {
      aa.select<32, 1>(0) = slm_block_load<float, 32>(0);
      aa.select<16, 1>(0) += aa.select<16, 1>(16 * 1);
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
      aa.select<4, 1>(0) += aa.select<4, 1>(4);
      aa.select<2, 1>(0) += aa.select<2, 1>(2);
    } else if constexpr (pixelPerGroupShift == 0) {
      aa.select<16, 1>(0) = slm_block_load<float, 16>(0);
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
      aa.select<4, 1>(0) += aa.select<4, 1>(4);
      aa.select<2, 1>(0) += aa.select<2, 1>(2);
      aa.select<1, 1>(0) += aa.select<1, 1>(1);
    }

    simd<fp16, 8 * 16 * 4> bias_data;
    bias_data.template bit_cast_view<unsigned char>().template select<pixelPerGroup*2, 1>(0) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        pixelPerGroup * sizeof(fp16),
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)bias + outputOffset*sizeof(fp16));
    aa.select<pixelPerGroup, 1>(0) += bias_data.select<pixelPerGroup, 1>(0);
    
    __ESIMD_ENS::lsc_block_store<
      fp16,
      pixelPerGroup,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((fp16*)c + outputOffset, aa.select<pixelPerGroup, 1>(0));
  }
}

template<uint32_t K_DIM, uint32_t pixelPerGroupShift>
ESIMD_INLINE void GEMV_bias_Int4Weight_FP16InOutNx16Temp_largeGRF_block_8T_128g(uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* d, uint8_t* bias, nd_item<1>& ndi) {
  constexpr uint32_t pixelPerGroup = 1 << pixelPerGroupShift;
  constexpr uint32_t quantPerGroup = K_DIM / GROUP * pixelPerGroup;
  constexpr uint32_t sumThreads = pixelPerGroup / 16;
  constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  constexpr uint32_t baseOffsetInc8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
  constexpr uint32_t K_DIM_DIV_4096 = (K_DIM + 4095) / 4096;
  constexpr uint32_t K_DIM_MOD_4096 = (K_DIM - (K_DIM_DIV_4096-1)*4096);
  constexpr uint32_t K_DIM_REDUCE_T = K_DIM_MOD_4096 / (4096 / 8); // 8 threads, 512 ele per thread in total 4096
  __ESIMD_NS::slm_init(16 * 128 * sizeof(float));
  int hh = ndi.get_local_id(0); // [0, 64)
  int h = ndi.get_group(0); // [0, 256)
  int rowSize = ndi.get_group_range(0) * pixelPerGroup;
  int offsetABase = (h * pixelPerGroup * K_DIM + hh * 8 * 8 * 4 * 2) >> 1;
  int offsetQuanBase = /*rowSize * K_DIM / 2 +*/ h * quantPerGroup * sizeof(fp16) + hh * 512 / GROUP * sizeof(fp16);
  int offsetB = hh * 128 * 2 * 2 * sizeof(fp16);
  int outputOffset = pixelPerGroup * h;
  int offsetSLMThread = hh * 2 * 64 * sizeof(float);
  simd<unsigned char, 256> aaa;
  simd<fp16, 80> quant;
  simd<fp16, 2560> bb;
  simd<float, 8 * 16 * 4> aa;
  simd<float, 16> cc(0.0f);
  uint32_t offsetA;
  uint32_t offsetQuan;
  offsetA = offsetABase;
  offsetQuan = offsetQuanBase;

#pragma unroll
  for (int k = 0; k < K_DIM_DIV_4096; k++) {
    if (k != K_DIM_DIV_4096-1 || hh < K_DIM_REDUCE_T)
    {
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*1) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 1 * 128 * sizeof(fp16));
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*2) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 2 * 128 * sizeof(fp16));
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*3) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 3 * 128 * sizeof(fp16));
    }
    else
    {
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*1) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*2) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*3) = 0;
    }

    offsetB += 4096 * sizeof(fp16);
  }

  for (int n = 0; n < pixelPerGroup; n++) {
    cc = 0.0f;
    offsetQuan = offsetQuanBase + n * K_DIM / GROUP * sizeof(fp16);
#pragma unroll
    for (int k = 0; k < K_DIM_DIV_4096-1; k++) {
      quant.template bit_cast_view<unsigned char>().template select<512 / GROUP * 2, 1>(512 / GROUP * 2 * k) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        512 / GROUP * 2,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuan);
      offsetQuan += 4096 / GROUP * sizeof(fp16);
    }
    if (hh < K_DIM_REDUCE_T)
    {
      quant.template bit_cast_view<unsigned char>().template select<512 / GROUP * 2, 1>(512 / GROUP * 2 * (K_DIM_DIV_4096-1)) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        512 / GROUP * 2,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuan);
    }
    else
    {
      quant.template bit_cast_view<unsigned char>().template select<512 / GROUP * 2, 1>(512 / GROUP * 2 * (K_DIM_DIV_4096-1)) = 0;
    }
    offsetQuan += 4096 / GROUP * sizeof(fp16);

    offsetA = offsetABase + n * K_DIM / 2;

#pragma unroll
    for (int k = 0; k < K_DIM_DIV_4096; k++) {
      if (k != K_DIM_DIV_4096-1 || hh < K_DIM_REDUCE_T)
      {
        simd<float, 512 / GROUP> fp32Q = quant.select<512 / GROUP, 1>(512 / GROUP * k);
        aaa.template bit_cast_view<unsigned char>().template select<256, 1>(0) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offsetA);

#pragma unroll
        for (int kk = 0; kk < 16; kk++) {
          aa.select<16, 2>(32 * kk) = aaa.select<16, 1>(16 * kk) & 0xf;
          aa.select<16, 2>(32 * kk + 1) = aaa.select<16, 1>(16 * kk) >> 4;
        }

        aa = aa - 8.0f;
#pragma unroll
        for (int kk = 0; kk < 512 / GROUP; kk++) {
          aa.select<GROUP, 1>(GROUP * kk) = fp32Q[kk] * aa.select<GROUP, 1>(GROUP * kk);
        }
#pragma unroll
        for (int kk = 0; kk < 32; kk++) {
          cc += aa.select<16, 1>(16 * kk) * bb.select<16, 1>(16 * kk + 128 * 2 * 2 * k); // note: 128 * 2 easy to have mistake!
        }
      }

      if (k == K_DIM_DIV_4096-1)
      {
        offsetA += K_DIM_REDUCE_T * 512 / 2;
      }
      else
      {
        offsetA += 2048;
      }
    }

    cc.select<8, 1>(0) += cc.select<8, 1>(8);
    cc.select<4, 1>(0) += cc.select<4, 1>(4);
    cc.select<2, 1>(0) += cc.select<2, 1>(2);
    simd<float, 1> slmAccumulationTemp = cc[0] + cc[1];
    uint32_t slmGroup = n >> 4;
    uint32_t slmInnerGroupOffset = n & 0xf;
    uint32_t slmAccumulationOffset = (slmGroup * 16 * 16 + hh * 16 + slmInnerGroupOffset) * sizeof(float);
    //slm_scalar_store(slmAccumulationOffset, slmAccumulationTemp);
    slm_block_store<float, 1>(slmAccumulationOffset, slmAccumulationTemp);
  }
  barrier();
  if (hh < sumThreads) {
    uint32_t slmSumPhase1LoadOffset = hh * 16 * 16 * sizeof(float);
#pragma unroll
    for (int k = 0; k < 4; k++) {
      aa.select<64, 1>(64 * k) = slm_block_load<float, 64>(slmSumPhase1LoadOffset + 64 * k * sizeof(float));
    }

#pragma unroll
    for (int k = 1; k < 16; k++) {
      aa.select<16, 1>(0) = aa.select<16, 1>(0) + aa.select<16, 1>(16 * k);
    }

    simd<fp16, 8 * 16 * 4> bias_data;
    bias_data.template bit_cast_view<unsigned char>().template select<16*2, 1>(0) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        16 * sizeof(fp16),
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)bias + (outputOffset + hh * 16)*sizeof(fp16));
    aa.select<16, 1>(0) += bias_data.select<16, 1>(0);

    __ESIMD_ENS::lsc_block_store<
      fp16,
      16,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((fp16*)c + outputOffset + hh * 16, aa.select<16, 1>(0));
  }
}

template<uint32_t K_DIM, uint32_t pixelPerGroupShift>
ESIMD_INLINE void GEMV_kqv_fuse_bias_Int4Weight_FP16InOutNx16Temp_largeGRF_block_ppg8_8T_128g(uint8_t* a, uint8_t* b, 
  uint8_t* cq, uint8_t* ck, uint8_t* cv, uint8_t* d, uint8_t* bias, 
  uint32_t out_q_features, uint32_t out_k_features, nd_item<1>& ndi) {
  constexpr uint32_t pixelPerGroup = 1 << pixelPerGroupShift;
  constexpr uint32_t quantPerGroup = K_DIM / GROUP * pixelPerGroup;
  constexpr uint32_t sumThreads = 1;
  constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  constexpr uint32_t baseOffsetInc8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
  constexpr uint32_t K_DIM_DIV_4096 = (K_DIM + 4095) / 4096;
  constexpr uint32_t K_DIM_MOD_4096 = (K_DIM - (K_DIM_DIV_4096-1)*4096);
  constexpr uint32_t K_DIM_REDUCE_T = K_DIM_MOD_4096 / (4096 / 8); // 8 threads, 512 ele per thread in total 4096
  __ESIMD_NS::slm_init(16 * sizeof(float));
  int hh = ndi.get_local_id(0); // [0, 64)
  int h = ndi.get_group(0); // [0, 256)
  int rowSize = ndi.get_group_range(0) * pixelPerGroup;
  int offsetABase = (h * pixelPerGroup * K_DIM + hh * 8 * 8 * 4 * 2) >> 1;
  int offsetQuanBase = /*rowSize * K_DIM / 2 +*/ h * quantPerGroup * sizeof(fp16) + hh * 512 / GROUP * sizeof(fp16);
  int offsetB = hh * 128 * 2 * 2 * sizeof(fp16);
  int outputOffset = pixelPerGroup * h;
  int outputOffset_bias = outputOffset;
  int offsetSLMThread = hh * 2 * 64 * sizeof(float);
  simd<unsigned char, 256> aaa;
  simd<fp16, 80> quant;
  simd<fp16, 2560> bb;
  simd<float, 8 * 16 * 4> aa;
  simd<float, 16> cc(0.0f);
  uint32_t offsetA;
  uint32_t offsetQuan;
  offsetA = offsetABase;
  offsetQuan = offsetQuanBase;
  uint8_t* c = cq;
  if (outputOffset >= out_q_features && outputOffset < (out_q_features + out_k_features))
  {
    // k part
    outputOffset -= out_q_features;
    c = ck;
  }
  else if (outputOffset >= (out_q_features + out_k_features))
  {
    // v part
    outputOffset -= (out_q_features + out_k_features);
    c = cv;
  }

#pragma unroll
  for (int k = 0; k < K_DIM_DIV_4096; k++) {
    if (k != K_DIM_DIV_4096-1 || hh < K_DIM_REDUCE_T)
    {
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*1) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 1 * 128 * sizeof(fp16));
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*2) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 2 * 128 * sizeof(fp16));
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*3) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 3 * 128 * sizeof(fp16));
    }
    else
    {
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*1) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*2) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*3) = 0;
    }

    offsetB += 4096 * sizeof(fp16);
  }

  for (int n = 0; n < pixelPerGroup; n++) {
    cc = 0.0f;
    offsetQuan = offsetQuanBase + n * K_DIM / GROUP * sizeof(fp16);
#pragma unroll
    for (int k = 0; k < K_DIM_DIV_4096-1; k++) {
      quant.template bit_cast_view<unsigned char>().template select<512 / GROUP * 2, 1>(512 / GROUP * 2 * k) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        512 / GROUP * 2,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuan);
      offsetQuan += 4096 / GROUP * sizeof(fp16);
    }
    if (hh < K_DIM_REDUCE_T)
    {
      quant.template bit_cast_view<unsigned char>().template select<512 / GROUP * 2, 1>(512 / GROUP * 2 * (K_DIM_DIV_4096-1)) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        512 / GROUP * 2,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuan);
    }
    else
    {
      quant.template bit_cast_view<unsigned char>().template select<32, 1>(32 * (K_DIM_DIV_4096-1)) = 0;
    }
    offsetQuan += 4096 / GROUP * sizeof(fp16);

    offsetA = offsetABase + n * K_DIM / 2;

#pragma unroll
    for (int k = 0; k < K_DIM_DIV_4096; k++) {
      if (k != K_DIM_DIV_4096-1 || hh < K_DIM_REDUCE_T)
      {
        simd<float, 512 / GROUP> fp32Q = quant.select<512 / GROUP, 1>(512 / GROUP * k);
        aaa.template bit_cast_view<unsigned char>().template select<256, 1>(0) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offsetA);

#pragma unroll
        for (int kk = 0; kk < 16; kk++) {
          aa.select<16, 2>(32 * kk) = aaa.select<16, 1>(16 * kk) & 0xf;
          aa.select<16, 2>(32 * kk + 1) = aaa.select<16, 1>(16 * kk) >> 4;
        }

        aa = aa - 8.0f;
#pragma unroll
        for (int kk = 0; kk < 512 / GROUP; kk++) {
          aa.select<GROUP, 1>(GROUP * kk) = fp32Q[kk] * aa.select<GROUP, 1>(GROUP * kk);
        }
#pragma unroll
        for (int kk = 0; kk < 32; kk++) {
          cc += aa.select<16, 1>(16 * kk) * bb.select<16, 1>(16 * kk + 128 * 2 * 2 * k); // note: 128 * 2 easy to have mistake!
        }
      }

      if (k == K_DIM_DIV_4096-1)
      {
        offsetA += K_DIM_REDUCE_T * 512 / 2;
      }
      else
      {
        offsetA += 2048;
      }
    }

    cc.select<8, 1>(0) += cc.select<8, 1>(8);
    cc.select<4, 1>(0) += cc.select<4, 1>(4);
    cc.select<2, 1>(0) += cc.select<2, 1>(2);
    simd<float, 1> slmAccumulationTemp = cc[0] + cc[1];
    uint32_t slmAccumulationOffset = (hh * pixelPerGroup + n) * sizeof(float);
    slm_block_store<float, 1>(slmAccumulationOffset, slmAccumulationTemp);
  }
  barrier();
  if (hh < sumThreads) {
if constexpr (pixelPerGroupShift == 3) {
#pragma unroll
      for (int k = 0; k < 2; k++) {
        aa.select<64, 1>(64 * k) = slm_block_load<float, 64>(64 * k * sizeof(float));
      }
#pragma unroll
      for (int k = 1; k < 8; k++) {
        aa.select<16, 1>(0) += aa.select<16, 1>(16 * k);
      }
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
    } else if constexpr (pixelPerGroupShift == 2) {
      aa.select<64, 1>(0) = slm_block_load<float, 64>(0);
#pragma unroll
      for (int k = 1; k < 4; k++) {
        aa.select<16, 1>(0) += aa.select<16, 1>(16 * k);
      }
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
      aa.select<4, 1>(0) += aa.select<4, 1>(4);
    } else if constexpr (pixelPerGroupShift == 1) {
      aa.select<32, 1>(0) = slm_block_load<float, 32>(0);
      aa.select<16, 1>(0) += aa.select<16, 1>(16 * 1);
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
      aa.select<4, 1>(0) += aa.select<4, 1>(4);
      aa.select<2, 1>(0) += aa.select<2, 1>(2);
    } else if constexpr (pixelPerGroupShift == 0) {
      aa.select<16, 1>(0) = slm_block_load<float, 16>(0);
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
      aa.select<4, 1>(0) += aa.select<4, 1>(4);
      aa.select<2, 1>(0) += aa.select<2, 1>(2);
      aa.select<1, 1>(0) += aa.select<1, 1>(1);
    }

    simd<fp16, 8 * 16 * 4> bias_data;
    bias_data.template bit_cast_view<unsigned char>().template select<pixelPerGroup*2, 1>(0) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        pixelPerGroup * sizeof(fp16),
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)bias + outputOffset_bias*sizeof(fp16));
    aa.select<pixelPerGroup, 1>(0) += bias_data.select<pixelPerGroup, 1>(0);
    
    __ESIMD_ENS::lsc_block_store<
      fp16,
      pixelPerGroup,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((fp16*)c + outputOffset, aa.select<pixelPerGroup, 1>(0));
  }
}

#undef GROUP

#define GROUP 32
template<uint32_t K_DIM, uint32_t pixelPerGroupShift>
ESIMD_INLINE void GEMV_bias_Int4Weight_FP16InOutNx16Temp_largeGRF_block_ppg8_8T_32g(uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* d, uint8_t* bias, nd_item<1>& ndi) {
  constexpr uint32_t pixelPerGroup = 1 << pixelPerGroupShift;
  constexpr uint32_t quantPerGroup = K_DIM / GROUP * pixelPerGroup;
  constexpr uint32_t sumThreads = 1;
  constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  constexpr uint32_t baseOffsetInc8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
  constexpr uint32_t K_DIM_DIV_4096 = (K_DIM + 4095) / 4096;
  constexpr uint32_t K_DIM_MOD_4096 = (K_DIM - (K_DIM_DIV_4096-1)*4096);
  constexpr uint32_t K_DIM_REDUCE_T = K_DIM_MOD_4096 / (4096 / 8); // 8 threads, 512 ele per thread in total 4096
  __ESIMD_NS::slm_init(16 * sizeof(float));
  int hh = ndi.get_local_id(0); // [0, 64)
  int h = ndi.get_group(0); // [0, 256)
  int rowSize = ndi.get_group_range(0) * pixelPerGroup;
  int offsetABase = (h * pixelPerGroup * K_DIM + hh * 8 * 8 * 4 * 2) >> 1;
  int offsetQuanBase = /*rowSize * K_DIM / 2 +*/ h * quantPerGroup * sizeof(fp16) + hh * 512 / GROUP * sizeof(fp16);
  int offsetB = hh * 128 * 2 * 2 * sizeof(fp16);
  int outputOffset = pixelPerGroup * h;
  int offsetSLMThread = hh * 2 * 64 * sizeof(float);
  simd<unsigned char, 256> aaa;
  simd<fp16, 80> quant;
  simd<fp16, 2560> bb;
  simd<float, 8 * 16 * 4> aa;
  simd<float, 16> cc(0.0f);
  uint32_t offsetA;
  uint32_t offsetQuan;
  offsetA = offsetABase;
  offsetQuan = offsetQuanBase;

#pragma unroll
  for (int k = 0; k < K_DIM_DIV_4096; k++) {
    if (k != K_DIM_DIV_4096-1 || hh < K_DIM_REDUCE_T)
    {
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*1) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 1 * 128 * sizeof(fp16));
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*2) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 2 * 128 * sizeof(fp16));
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*3) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 3 * 128 * sizeof(fp16));
    }
    else
    {
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*1) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*2) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*3) = 0;
    }

    offsetB += 4096 * sizeof(fp16);
  }

  for (int n = 0; n < pixelPerGroup; n++) {
    cc = 0.0f;
    offsetQuan = offsetQuanBase + n * K_DIM / GROUP * sizeof(fp16);
#pragma unroll
    for (int k = 0; k < K_DIM_DIV_4096-1; k++) {
      quant.template bit_cast_view<unsigned char>().template select<512 / GROUP * 2, 1>(512 / GROUP * 2 * k) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        512 / GROUP * 2,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuan);
      offsetQuan += 4096 / GROUP * sizeof(fp16);
    }
    if (hh < K_DIM_REDUCE_T)
    {
      quant.template bit_cast_view<unsigned char>().template select<512 / GROUP * 2, 1>(512 / GROUP * 2 * (K_DIM_DIV_4096-1)) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        512 / GROUP * 2,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuan);
    }
    else
    {
      quant.template bit_cast_view<unsigned char>().template select<32, 1>(32 * (K_DIM_DIV_4096-1)) = 0;
    }
    offsetQuan += 4096 / GROUP * sizeof(fp16);

    offsetA = offsetABase + n * K_DIM / 2;

#pragma unroll
    for (int k = 0; k < K_DIM_DIV_4096; k++) {
      if (k != K_DIM_DIV_4096-1 || hh < K_DIM_REDUCE_T)
      {
        simd<float, 512 / GROUP> fp32Q = quant.select<512 / GROUP, 1>(512 / GROUP * k);
        aaa.template bit_cast_view<unsigned char>().template select<256, 1>(0) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offsetA);

#pragma unroll
        for (int kk = 0; kk < 16; kk++) {
          aa.select<16, 2>(32 * kk) = aaa.select<16, 1>(16 * kk) & 0xf;
          aa.select<16, 2>(32 * kk + 1) = aaa.select<16, 1>(16 * kk) >> 4;
        }

        aa = aa - 8.0f;
#pragma unroll
        for (int kk = 0; kk < 512 / GROUP; kk++) {
          aa.select<GROUP, 1>(GROUP * kk) = fp32Q[kk] * aa.select<GROUP, 1>(GROUP * kk);
        }
#pragma unroll
        for (int kk = 0; kk < 32; kk++) {
          cc += aa.select<16, 1>(16 * kk) * bb.select<16, 1>(16 * kk + 128 * 2 * 2 * k); // note: 128 * 2 easy to have mistake!
        }
      }

      if (k == K_DIM_DIV_4096-1)
      {
        offsetA += K_DIM_REDUCE_T * 512 / 2;
      }
      else
      {
        offsetA += 2048;
      }
    }

    cc.select<8, 1>(0) += cc.select<8, 1>(8);
    cc.select<4, 1>(0) += cc.select<4, 1>(4);
    cc.select<2, 1>(0) += cc.select<2, 1>(2);
    simd<float, 1> slmAccumulationTemp = cc[0] + cc[1];
    uint32_t slmAccumulationOffset = (hh * pixelPerGroup + n) * sizeof(float);
    slm_block_store<float, 1>(slmAccumulationOffset, slmAccumulationTemp);
  }
  barrier();
  if (hh < sumThreads) {
if constexpr (pixelPerGroupShift == 3) {
#pragma unroll
      for (int k = 0; k < 2; k++) {
        aa.select<64, 1>(64 * k) = slm_block_load<float, 64>(64 * k * sizeof(float));
      }
#pragma unroll
      for (int k = 1; k < 8; k++) {
        aa.select<16, 1>(0) += aa.select<16, 1>(16 * k);
      }
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
    } else if constexpr (pixelPerGroupShift == 2) {
      aa.select<64, 1>(0) = slm_block_load<float, 64>(0);
#pragma unroll
      for (int k = 1; k < 4; k++) {
        aa.select<16, 1>(0) += aa.select<16, 1>(16 * k);
      }
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
      aa.select<4, 1>(0) += aa.select<4, 1>(4);
    } else if constexpr (pixelPerGroupShift == 1) {
      aa.select<32, 1>(0) = slm_block_load<float, 32>(0);
      aa.select<16, 1>(0) += aa.select<16, 1>(16 * 1);
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
      aa.select<4, 1>(0) += aa.select<4, 1>(4);
      aa.select<2, 1>(0) += aa.select<2, 1>(2);
    } else if constexpr (pixelPerGroupShift == 0) {
      aa.select<16, 1>(0) = slm_block_load<float, 16>(0);
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
      aa.select<4, 1>(0) += aa.select<4, 1>(4);
      aa.select<2, 1>(0) += aa.select<2, 1>(2);
      aa.select<1, 1>(0) += aa.select<1, 1>(1);
    }

    simd<fp16, 8 * 16 * 4> bias_data;
    bias_data.template bit_cast_view<unsigned char>().template select<pixelPerGroup*2, 1>(0) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        pixelPerGroup * sizeof(fp16),
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)bias + outputOffset*sizeof(fp16));
    aa.select<pixelPerGroup, 1>(0) += bias_data.select<pixelPerGroup, 1>(0);

    __ESIMD_ENS::lsc_block_store<
      fp16,
      pixelPerGroup,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((fp16*)c + outputOffset, aa.select<pixelPerGroup, 1>(0));
  }
}

template<uint32_t K_DIM, uint32_t pixelPerGroupShift>
ESIMD_INLINE void GEMV_bias_Int4Weight_FP16InOutNx16Temp_largeGRF_block_8T_32g(uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* d, uint8_t* bias, nd_item<1>& ndi) {
  constexpr uint32_t pixelPerGroup = 1 << pixelPerGroupShift;
  constexpr uint32_t quantPerGroup = K_DIM / GROUP * pixelPerGroup;
  constexpr uint32_t sumThreads = pixelPerGroup / 16;
  constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  constexpr uint32_t baseOffsetInc8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
  constexpr uint32_t K_DIM_DIV_4096 = (K_DIM + 4095) / 4096;
  constexpr uint32_t K_DIM_MOD_4096 = (K_DIM - (K_DIM_DIV_4096-1)*4096);
  constexpr uint32_t K_DIM_REDUCE_T = K_DIM_MOD_4096 / (4096 / 8); // 8 threads, 512 ele per thread in total 4096
  __ESIMD_NS::slm_init(16 * 128 * sizeof(float));
  int hh = ndi.get_local_id(0); // [0, 64)
  int h = ndi.get_group(0); // [0, 256)
  int rowSize = ndi.get_group_range(0) * pixelPerGroup;
  int offsetABase = (h * pixelPerGroup * K_DIM + hh * 8 * 8 * 4 * 2) >> 1;
  int offsetQuanBase = /*rowSize * K_DIM / 2 +*/ h * quantPerGroup * sizeof(fp16) + hh * 512 / GROUP * sizeof(fp16);
  int offsetB = hh * 128 * 2 * 2 * sizeof(fp16);
  int outputOffset = pixelPerGroup * h;
  int offsetSLMThread = hh * 2 * 64 * sizeof(float);
  simd<unsigned char, 256> aaa;
  simd<fp16, 80> quant;
  simd<fp16, 2560> bb;
  simd<float, 8 * 16 * 4> aa;
  simd<float, 16> cc(0.0f);
  uint32_t offsetA;
  uint32_t offsetQuan;
  offsetA = offsetABase;
  offsetQuan = offsetQuanBase;

#pragma unroll
  for (int k = 0; k < K_DIM_DIV_4096; k++) {
    if (k != K_DIM_DIV_4096-1 || hh < K_DIM_REDUCE_T)
    {
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*1) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 1 * 128 * sizeof(fp16));
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*2) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 2 * 128 * sizeof(fp16));
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*3) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 3 * 128 * sizeof(fp16));
    }
    else
    {
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*1) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*2) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*3) = 0;
    }

    offsetB += 4096 * sizeof(fp16);
  }

  for (int n = 0; n < pixelPerGroup; n++) {
    cc = 0.0f;
    offsetQuan = offsetQuanBase + n * K_DIM / GROUP * sizeof(fp16);
#pragma unroll
    for (int k = 0; k < K_DIM_DIV_4096-1; k++) {
      quant.template bit_cast_view<unsigned char>().template select<512 / GROUP * 2, 1>(512 / GROUP * 2 * k) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        512 / GROUP * 2,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuan);
      offsetQuan += 4096 / GROUP * sizeof(fp16);
    }
    if (hh < K_DIM_REDUCE_T)
    {
      quant.template bit_cast_view<unsigned char>().template select<512 / GROUP * 2, 1>(512 / GROUP * 2 * (K_DIM_DIV_4096-1)) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        512 / GROUP * 2,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuan);
    }
    else
    {
      quant.template bit_cast_view<unsigned char>().template select<512 / GROUP * 2, 1>(512 / GROUP * 2 * (K_DIM_DIV_4096-1)) = 0;
    }
    offsetQuan += 4096 / GROUP * sizeof(fp16);

    offsetA = offsetABase + n * K_DIM / 2;

#pragma unroll
    for (int k = 0; k < K_DIM_DIV_4096; k++) {
      if (k != K_DIM_DIV_4096-1 || hh < K_DIM_REDUCE_T)
      {
        simd<float, 512 / GROUP> fp32Q = quant.select<512 / GROUP, 1>(512 / GROUP * k);
        aaa.template bit_cast_view<unsigned char>().template select<256, 1>(0) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offsetA);

#pragma unroll
        for (int kk = 0; kk < 16; kk++) {
          aa.select<16, 2>(32 * kk) = aaa.select<16, 1>(16 * kk) & 0xf;
          aa.select<16, 2>(32 * kk + 1) = aaa.select<16, 1>(16 * kk) >> 4;
        }

        aa = aa - 8.0f;
#pragma unroll
        for (int kk = 0; kk < 512 / GROUP; kk++) {
          aa.select<GROUP, 1>(GROUP * kk) = fp32Q[kk] * aa.select<GROUP, 1>(GROUP * kk);
        }
#pragma unroll
        for (int kk = 0; kk < 32; kk++) {
          cc += aa.select<16, 1>(16 * kk) * bb.select<16, 1>(16 * kk + 128 * 2 * 2 * k); // note: 128 * 2 easy to have mistake!
        }
      }

      if (k == K_DIM_DIV_4096-1)
      {
        offsetA += K_DIM_REDUCE_T * 512 / 2;
      }
      else
      {
        offsetA += 2048;
      }
    }

    cc.select<8, 1>(0) += cc.select<8, 1>(8);
    cc.select<4, 1>(0) += cc.select<4, 1>(4);
    cc.select<2, 1>(0) += cc.select<2, 1>(2);
    simd<float, 1> slmAccumulationTemp = cc[0] + cc[1];
    uint32_t slmGroup = n >> 4;
    uint32_t slmInnerGroupOffset = n & 0xf;
    uint32_t slmAccumulationOffset = (slmGroup * 16 * 16 + hh * 16 + slmInnerGroupOffset) * sizeof(float);
    //slm_scalar_store(slmAccumulationOffset, slmAccumulationTemp);
    slm_block_store<float, 1>(slmAccumulationOffset, slmAccumulationTemp);
  }
  barrier();
  if (hh < sumThreads) {
    uint32_t slmSumPhase1LoadOffset = hh * 16 * 16 * sizeof(float);
#pragma unroll
    for (int k = 0; k < 4; k++) {
      aa.select<64, 1>(64 * k) = slm_block_load<float, 64>(slmSumPhase1LoadOffset + 64 * k * sizeof(float));
    }

#pragma unroll
    for (int k = 1; k < 16; k++) {
      aa.select<16, 1>(0) = aa.select<16, 1>(0) + aa.select<16, 1>(16 * k);
    }

    simd<fp16, 8 * 16 * 4> bias_data;
    bias_data.template bit_cast_view<unsigned char>().template select<16*2, 1>(0) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        16 * sizeof(fp16),
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)bias + (outputOffset + hh * 16)*sizeof(fp16));
    aa.select<16, 1>(0) += bias_data.select<16, 1>(0);

    __ESIMD_ENS::lsc_block_store<
      fp16,
      16,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((fp16*)c + outputOffset + hh * 16, aa.select<16, 1>(0));
  }
}

template<uint32_t K_DIM, uint32_t pixelPerGroupShift>
ESIMD_INLINE void GEMV_kqv_fuse_bias_Int4Weight_FP16InOutNx16Temp_largeGRF_block_ppg8_8T_32g(uint8_t* a, uint8_t* b, 
  uint8_t* cq, uint8_t* ck, uint8_t* cv, uint8_t* d, uint8_t* bias, 
  uint32_t out_q_features, uint32_t out_k_features, nd_item<1>& ndi) {
  constexpr uint32_t pixelPerGroup = 1 << pixelPerGroupShift;
  constexpr uint32_t quantPerGroup = K_DIM / GROUP * pixelPerGroup;
  constexpr uint32_t sumThreads = 1;
  constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  constexpr uint32_t baseOffsetInc8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
  constexpr uint32_t K_DIM_DIV_4096 = (K_DIM + 4095) / 4096;
  constexpr uint32_t K_DIM_MOD_4096 = (K_DIM - (K_DIM_DIV_4096-1)*4096);
  constexpr uint32_t K_DIM_REDUCE_T = K_DIM_MOD_4096 / (4096 / 8); // 8 threads, 512 ele per thread in total 4096
  __ESIMD_NS::slm_init(16 * sizeof(float));
  int hh = ndi.get_local_id(0); // [0, 64)
  int h = ndi.get_group(0); // [0, 256)
  int rowSize = ndi.get_group_range(0) * pixelPerGroup;
  int offsetABase = (h * pixelPerGroup * K_DIM + hh * 8 * 8 * 4 * 2) >> 1;
  int offsetQuanBase = /*rowSize * K_DIM / 2 +*/ h * quantPerGroup * sizeof(fp16) + hh * 512 / GROUP * sizeof(fp16);
  int offsetB = hh * 128 * 2 * 2 * sizeof(fp16);
  int outputOffset = pixelPerGroup * h;
  int outputOffset_bias = outputOffset;
  int offsetSLMThread = hh * 2 * 64 * sizeof(float);
  simd<unsigned char, 256> aaa;
  simd<fp16, 80> quant;
  simd<fp16, 2560> bb;
  simd<float, 8 * 16 * 4> aa;
  simd<float, 16> cc(0.0f);
  uint32_t offsetA;
  uint32_t offsetQuan;
  offsetA = offsetABase;
  offsetQuan = offsetQuanBase;
  uint8_t* c = cq;
  if (outputOffset >= out_q_features && outputOffset < (out_q_features + out_k_features))
  {
    // k part
    outputOffset -= out_q_features;
    c = ck;
  }
  else if (outputOffset >= (out_q_features + out_k_features))
  {
    // v part
    outputOffset -= (out_q_features + out_k_features);
    c = cv;
  }

#pragma unroll
  for (int k = 0; k < K_DIM_DIV_4096; k++) {
    if (k != K_DIM_DIV_4096-1 || hh < K_DIM_REDUCE_T)
    {
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*1) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 1 * 128 * sizeof(fp16));
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*2) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 2 * 128 * sizeof(fp16));
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*3) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 3 * 128 * sizeof(fp16));
    }
    else
    {
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*1) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*2) = 0;
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(1024 * k + 256*3) = 0;
    }

    offsetB += 4096 * sizeof(fp16);
  }

  for (int n = 0; n < pixelPerGroup; n++) {
    cc = 0.0f;
    offsetQuan = offsetQuanBase + n * K_DIM / GROUP * sizeof(fp16);
#pragma unroll
    for (int k = 0; k < K_DIM_DIV_4096-1; k++) {
      quant.template bit_cast_view<unsigned char>().template select<512 / GROUP * 2, 1>(512 / GROUP * 2 * k) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        512 / GROUP * 2,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuan);
      offsetQuan += 4096 / GROUP * sizeof(fp16);
    }
    if (hh < K_DIM_REDUCE_T)
    {
      quant.template bit_cast_view<unsigned char>().template select<512 / GROUP * 2, 1>(512 / GROUP * 2 * (K_DIM_DIV_4096-1)) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        512 / GROUP * 2,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuan);
    }
    else
    {
      quant.template bit_cast_view<unsigned char>().template select<32, 1>(32 * (K_DIM_DIV_4096-1)) = 0;
    }
    offsetQuan += 4096 / GROUP * sizeof(fp16);

    offsetA = offsetABase + n * K_DIM / 2;

#pragma unroll
    for (int k = 0; k < K_DIM_DIV_4096; k++) {
      if (k != K_DIM_DIV_4096-1 || hh < K_DIM_REDUCE_T)
      {
        simd<float, 512 / GROUP> fp32Q = quant.select<512 / GROUP, 1>(512 / GROUP * k);
        aaa.template bit_cast_view<unsigned char>().template select<256, 1>(0) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offsetA);

#pragma unroll
        for (int kk = 0; kk < 16; kk++) {
          aa.select<16, 2>(32 * kk) = aaa.select<16, 1>(16 * kk) & 0xf;
          aa.select<16, 2>(32 * kk + 1) = aaa.select<16, 1>(16 * kk) >> 4;
        }

        aa = aa - 8.0f;
#pragma unroll
        for (int kk = 0; kk < 512 / GROUP; kk++) {
          aa.select<GROUP, 1>(GROUP * kk) = fp32Q[kk] * aa.select<GROUP, 1>(GROUP * kk);
        }
#pragma unroll
        for (int kk = 0; kk < 32; kk++) {
          cc += aa.select<16, 1>(16 * kk) * bb.select<16, 1>(16 * kk + 128 * 2 * 2 * k); // note: 128 * 2 easy to have mistake!
        }
      }

      if (k == K_DIM_DIV_4096-1)
      {
        offsetA += K_DIM_REDUCE_T * 512 / 2;
      }
      else
      {
        offsetA += 2048;
      }
    }

    cc.select<8, 1>(0) += cc.select<8, 1>(8);
    cc.select<4, 1>(0) += cc.select<4, 1>(4);
    cc.select<2, 1>(0) += cc.select<2, 1>(2);
    simd<float, 1> slmAccumulationTemp = cc[0] + cc[1];
    uint32_t slmAccumulationOffset = (hh * pixelPerGroup + n) * sizeof(float);
    slm_block_store<float, 1>(slmAccumulationOffset, slmAccumulationTemp);
  }
  barrier();
  if (hh < sumThreads) {
if constexpr (pixelPerGroupShift == 3) {
#pragma unroll
      for (int k = 0; k < 2; k++) {
        aa.select<64, 1>(64 * k) = slm_block_load<float, 64>(64 * k * sizeof(float));
      }
#pragma unroll
      for (int k = 1; k < 8; k++) {
        aa.select<16, 1>(0) += aa.select<16, 1>(16 * k);
      }
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
    } else if constexpr (pixelPerGroupShift == 2) {
      aa.select<64, 1>(0) = slm_block_load<float, 64>(0);
#pragma unroll
      for (int k = 1; k < 4; k++) {
        aa.select<16, 1>(0) += aa.select<16, 1>(16 * k);
      }
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
      aa.select<4, 1>(0) += aa.select<4, 1>(4);
    } else if constexpr (pixelPerGroupShift == 1) {
      aa.select<32, 1>(0) = slm_block_load<float, 32>(0);
      aa.select<16, 1>(0) += aa.select<16, 1>(16 * 1);
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
      aa.select<4, 1>(0) += aa.select<4, 1>(4);
      aa.select<2, 1>(0) += aa.select<2, 1>(2);
    } else if constexpr (pixelPerGroupShift == 0) {
      aa.select<16, 1>(0) = slm_block_load<float, 16>(0);
      aa.select<8, 1>(0) += aa.select<8, 1>(8);
      aa.select<4, 1>(0) += aa.select<4, 1>(4);
      aa.select<2, 1>(0) += aa.select<2, 1>(2);
      aa.select<1, 1>(0) += aa.select<1, 1>(1);
    }

    simd<fp16, 8 * 16 * 4> bias_data;
    bias_data.template bit_cast_view<unsigned char>().template select<pixelPerGroup*2, 1>(0) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        pixelPerGroup * sizeof(fp16),
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)bias + outputOffset_bias*sizeof(fp16));
    aa.select<pixelPerGroup, 1>(0) += bias_data.select<pixelPerGroup, 1>(0);
    
    __ESIMD_ENS::lsc_block_store<
      fp16,
      pixelPerGroup,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((fp16*)c + outputOffset, aa.select<pixelPerGroup, 1>(0));
  }
}

#undef GROUP