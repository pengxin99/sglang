#include "utils.h"

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using fp16 = sycl::half;
using namespace sycl;


ESIMD_INLINE void rmsNorm128PerThread_fp16(uint8_t* weight, uint8_t* hidden_states, uint8_t* hidden_states_out, int64_t hidden_size, float variance_epsilon ,nd_item<1>& ndi) {

  __ESIMD_NS::slm_init(16 * sizeof(fp16)); 

  int hh = ndi.get_local_linear_id();
  uint32_t inputOffset = 128 * hh * sizeof(fp16);
  int active_thread_num = 16;

  simd<fp16, 128> input_FP16;
  simd<fp16, 128> input;
  simd<fp16, 128> input_powered;
  simd<fp16, 128> weight_FP16;
  simd<fp16, 16> variance = 0;

  simd<fp16, 16> varianceSum = 0;

  // if (hh < 16)
  {
    input_FP16.template bit_cast_view<uint8_t>().template select<256, 1>(0) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)hidden_states + inputOffset);

    weight_FP16.template bit_cast_view<uint8_t>().template select<256, 1>(0) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)weight + inputOffset);
  }
  // else
  // {
  //   input_FP16 = 0;
  //   weight_FP16 = 0;
  // }

  input = input_FP16;

#pragma unroll
  for (int ll = 0; ll < 8; ll++) {
    input_powered.select<16, 1>(ll *16) = pow<float, 16, float>(input.select<16, 1>(ll *16), 2.0f);
    variance += input_powered.select<16, 1>(ll *16);
  }

  variance.select<8, 1>(0) += variance.select<8, 1>(8);
  variance.select<4, 1>(0) += variance.select<4, 1>(4);
  variance.select<2, 1>(0) += variance.select<2, 1>(2);
  variance[0] += variance[1];

  slm_block_store<fp16, 1>(hh * sizeof(fp16), variance[0]);

  barrier();
  
  varianceSum.select<16, 1>(0) = slm_block_load<fp16, 16>(0);
  
  varianceSum.select<8, 1>(0) += varianceSum.select<8, 1>(8);
  varianceSum.select<4, 1>(0) += varianceSum.select<4, 1>(4);
  varianceSum.select<2, 1>(0) += varianceSum.select<2, 1>(2);
  varianceSum[0] += varianceSum[1];

  varianceSum[0] = varianceSum[0] / 2048.0f;
  varianceSum[0] =  sqrt<fp16, 1, fp16>(varianceSum[0] + variance_epsilon);

  simd<fp16, 128> varianceAVG;
  // fill 128 length varianceAVG
  varianceAVG[0] = varianceSum[0];
  varianceAVG[1] = varianceAVG[0];
  varianceAVG.select<2, 1>(2) = varianceAVG.select<2, 1>(0);
  varianceAVG.select<4, 1>(4) = varianceAVG.select<4, 1>(0);
  varianceAVG.select<8, 1>(8) = varianceAVG.select<8, 1>(0);
  varianceAVG.select<16, 1>(16) = varianceAVG.select<16, 1>(0);
  varianceAVG.select<32, 1>(32) = varianceAVG.select<32, 1>(0);
  varianceAVG.select<64, 1>(64) = varianceAVG.select<64, 1>(0);
  // simd<f, 128> varianceAVG{(float)varianceSum[0]};

  input = input / varianceAVG;

  input_FP16 = input;
  input_FP16 = input_FP16 * weight_FP16;

  // if (hh < 16)
  {
    __ESIMD_ENS::lsc_block_store<
      fp16,
      128,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((fp16*)hidden_states_out + 128 * hh, input_FP16.select<128, 1>(0));
  }
}

ESIMD_INLINE void rmsNorm128PerThread(uint8_t* weight, uint8_t* hidden_states, uint8_t* hidden_states_out, int64_t hidden_size, int64_t input_len, float variance_epsilon ,nd_item<1>& ndi) {

  __ESIMD_NS::slm_init(16 * sizeof(float)); 

  int h = ndi.get_group(0);
  if (h >= input_len) return;

  int hh = ndi.get_local_linear_id();
  uint32_t inputOffset = 128 * hh * sizeof(fp16);
  int active_thread_num = 16;

  simd<fp16, 128> input_FP16;
  simd<float, 128> input;
  simd<float, 128> input_powered;
  simd<fp16, 128> weight_FP16;
  simd<float, 16> variance = 0;

  simd<float, 16> varianceSum = 0;

  // if (hh < 16)
  {
    input_FP16.template bit_cast_view<uint8_t>().template select<256, 1>(0) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)hidden_states + h * hidden_size * sizeof(fp16) + inputOffset);

    weight_FP16.template bit_cast_view<uint8_t>().template select<256, 1>(0) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)weight + inputOffset);
  }
  // else
  // {
  //   input_FP16 = 0;
  //   weight_FP16 = 0;
  // }

  input = input_FP16;

#pragma unroll
  for (int ll = 0; ll < 8; ll++) {
    input_powered.select<16, 1>(ll *16) = input.select<16, 1>(ll *16) * input.select<16, 1>(ll *16);// pow<float, 16, float>(input.select<16, 1>(ll *16), 2.0f);
    variance += input_powered.select<16, 1>(ll *16);
  }

  variance.select<8, 1>(0) = variance.select<8, 1>(0) + variance.select<8, 1>(8);
  variance.select<4, 1>(0) = variance.select<4, 1>(0) + variance.select<4, 1>(4);
  variance.select<2, 1>(0) = variance.select<2, 1>(0) + variance.select<2, 1>(2);
  variance[0] = variance[0] + variance[1];

  variance[0] = variance[0] / 128.0f;

  slm_block_store<float, 1>(hh * sizeof(float), variance[0]);

  barrier();
  
  varianceSum.select<16, 1>(0) = slm_block_load<float, 16>(0);
  
  varianceSum.select<8, 1>(0) = varianceSum.select<8, 1>(0) + varianceSum.select<8, 1>(8);
  varianceSum.select<4, 1>(0) = varianceSum.select<4, 1>(0) + varianceSum.select<4, 1>(4);
  varianceSum.select<2, 1>(0) = varianceSum.select<2, 1>(0) + varianceSum.select<2, 1>(2);
  varianceSum[0] = varianceSum[0] + varianceSum[1];

  varianceSum[0] = varianceSum[0] / 16.0f;
  // varianceSum[0] =  sqrt<float, 1, float>(varianceSum[0] + variance_epsilon);
  // varianceSum =  sqrt<float, 16, float>(varianceSum + variance_epsilon);
  // varianceSum[0] =  sqrt<float, 1, float>(varianceSum[0]);
  varianceSum[0] = sqrt(varianceSum[0] + variance_epsilon);

  simd<float, 128> varianceAVG = 0;
  // fill 128 length varianceAVG
  varianceAVG[0] = varianceSum[0];
  varianceAVG[1] = varianceAVG[0];
  varianceAVG.select<2, 1>(2) = varianceAVG.select<2, 1>(0);
  varianceAVG.select<4, 1>(4) = varianceAVG.select<4, 1>(0);
  varianceAVG.select<8, 1>(8) = varianceAVG.select<8, 1>(0);
  varianceAVG.select<16, 1>(16) = varianceAVG.select<16, 1>(0);
  varianceAVG.select<32, 1>(32) = varianceAVG.select<32, 1>(0);
  varianceAVG.select<64, 1>(64) = varianceAVG.select<64, 1>(0);
  // simd<float, 128> varianceAVG{varianceSum[0]};

  input = input / varianceAVG;

  input_FP16 = input;
  input_FP16 = input_FP16 * weight_FP16;

  // if (hh < 16)
  {
    __ESIMD_ENS::lsc_block_store<
      fp16,
      128,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((fp16*)hidden_states_out + h * hidden_size + 128 * hh, input_FP16.select<128, 1>(0));
  }
}


ESIMD_INLINE void rmsNorm128PerThread_32t(uint8_t* weight, uint8_t* hidden_states, uint8_t* hidden_states_out, int64_t hidden_size, int64_t input_len, float variance_epsilon ,nd_item<1>& ndi) {

  __ESIMD_NS::slm_init(32 * sizeof(float)); 

  int h = ndi.get_group(0);
  int hh = ndi.get_local_linear_id();
  uint32_t inputOffset = 128 * hh * sizeof(fp16);
  int active_thread_num = hidden_size / 128;
  if (h >= input_len) return;

  simd<fp16, 128> input_FP16;
  simd<float, 128> input;
  simd<float, 128> input_powered;
  simd<fp16, 128> weight_FP16;
  simd<float, 16> variance = 0;

  simd<float, 32> varianceSum = 0;

  if (hh < active_thread_num)
  {
    input_FP16.template bit_cast_view<uint8_t>().template select<256, 1>(0) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)hidden_states + h * hidden_size * sizeof(fp16) + inputOffset);

    weight_FP16.template bit_cast_view<uint8_t>().template select<256, 1>(0) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)weight + inputOffset);
  }
  else
  {
    input_FP16 = 0;
    weight_FP16 = 0;
  }

  input = input_FP16;

#pragma unroll
  for (int ll = 0; ll < 8; ll++) {
    input_powered.select<16, 1>(ll *16) = pow<float, 16, float>(input.select<16, 1>(ll *16), 2.0f);
    variance += input_powered.select<16, 1>(ll *16);
  }

  variance.select<8, 1>(0) += variance.select<8, 1>(8);
  variance.select<4, 1>(0) += variance.select<4, 1>(4);
  variance.select<2, 1>(0) += variance.select<2, 1>(2);
  variance[0] += variance[1];

  slm_block_store<float, 1>(hh * sizeof(float), variance[0]);

  barrier();
  
  varianceSum.select<32, 1>(0) = slm_block_load<float, 32>(0);
  
  varianceSum.select<16, 1>(0) += varianceSum.select<16, 1>(16);
  varianceSum.select<8, 1>(0) += varianceSum.select<8, 1>(8);
  varianceSum.select<4, 1>(0) += varianceSum.select<4, 1>(4);
  varianceSum.select<2, 1>(0) += varianceSum.select<2, 1>(2);
  varianceSum[0] += varianceSum[1];

  varianceSum[0] = varianceSum[0] / hidden_size;
  varianceSum[0] =  sqrt(varianceSum[0] + variance_epsilon);

  simd<float, 128> varianceAVG;
  // fill 128 length varianceAVG
  varianceAVG[0] = varianceSum[0];
  varianceAVG[1] = varianceAVG[0];
  varianceAVG.select<2, 1>(2) = varianceAVG.select<2, 1>(0);
  varianceAVG.select<4, 1>(4) = varianceAVG.select<4, 1>(0);
  varianceAVG.select<8, 1>(8) = varianceAVG.select<8, 1>(0);
  varianceAVG.select<16, 1>(16) = varianceAVG.select<16, 1>(0);
  varianceAVG.select<32, 1>(32) = varianceAVG.select<32, 1>(0);
  varianceAVG.select<64, 1>(64) = varianceAVG.select<64, 1>(0);
  // simd<float, 128> varianceAVG{varianceSum[0]};

  input = input / varianceAVG;

  input_FP16 = input;
  input_FP16 = input_FP16 * weight_FP16;

  if (hh < active_thread_num)
  {
    __ESIMD_ENS::lsc_block_store<
      fp16,
      128,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((fp16*)hidden_states_out + h * hidden_size + 128 * hh, input_FP16.select<128, 1>(0));
  }
}


ESIMD_INLINE void kq_rmsNorm128PerThread_32t(uint8_t* weight_1, uint8_t* weight_2, uint8_t* hidden_states_1, uint8_t* hidden_states_2, 
  uint8_t* hidden_states_out_1, uint8_t* hidden_states_out_2,

  int64_t hidden_size,  // hidden_size
  int64_t input_len,  // input len
  int64_t input_len_2,  // input len
  float variance_epsilon_1,
  float variance_epsilon_2,
  nd_item<1>& ndi) {

  __ESIMD_NS::slm_init(32 * sizeof(float)); 

  int h = ndi.get_group(0);

  uint8_t* weight = weight_1;
  uint8_t* hidden_states = hidden_states_1;
  uint8_t* hidden_states_out = hidden_states_out_1;
  float variance_epsilon = variance_epsilon_1;

  if (h >= input_len)
  {
    h -= input_len;
    weight = weight_2;
    hidden_states = hidden_states_2;
    hidden_states_out = hidden_states_out_2;
    variance_epsilon = variance_epsilon_2;
  }

  int hh = ndi.get_local_linear_id();
  uint32_t inputOffset = 128 * hh * sizeof(fp16);
  int active_thread_num = hidden_size / 128;
  
  simd<fp16, 128> input_FP16;
  simd<float, 128> input;
  simd<float, 128> input_powered;
  simd<fp16, 128> weight_FP16;
  simd<float, 16> variance = 0;

  simd<float, 32> varianceSum = 0;

  if (hh < active_thread_num)
  {
    input_FP16.template bit_cast_view<uint8_t>().template select<256, 1>(0) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)hidden_states + h * hidden_size * sizeof(fp16) + inputOffset);

    weight_FP16.template bit_cast_view<uint8_t>().template select<256, 1>(0) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)weight + inputOffset);
  }
  else
  {
    input_FP16 = 0;
    weight_FP16 = 0;
  }

  input = input_FP16;

#pragma unroll
  for (int ll = 0; ll < 8; ll++) {
    input_powered.select<16, 1>(ll *16) = pow<float, 16, float>(input.select<16, 1>(ll *16), 2.0f);
    variance += input_powered.select<16, 1>(ll *16);
  }

  variance.select<8, 1>(0) += variance.select<8, 1>(8);
  variance.select<4, 1>(0) += variance.select<4, 1>(4);
  variance.select<2, 1>(0) += variance.select<2, 1>(2);
  variance[0] += variance[1];

  slm_block_store<float, 1>(hh * sizeof(float), variance[0]);

  barrier();
  
  varianceSum.select<32, 1>(0) = slm_block_load<float, 32>(0);
  
  varianceSum.select<16, 1>(0) += varianceSum.select<16, 1>(16);
  varianceSum.select<8, 1>(0) += varianceSum.select<8, 1>(8);
  varianceSum.select<4, 1>(0) += varianceSum.select<4, 1>(4);
  varianceSum.select<2, 1>(0) += varianceSum.select<2, 1>(2);
  varianceSum[0] += varianceSum[1];

  varianceSum[0] = varianceSum[0] / hidden_size;
  varianceSum[0] =  sqrt(varianceSum[0] + variance_epsilon);

  simd<float, 128> varianceAVG;
  // fill 128 length varianceAVG
  varianceAVG[0] = varianceSum[0];
  varianceAVG[1] = varianceAVG[0];
  varianceAVG.select<2, 1>(2) = varianceAVG.select<2, 1>(0);
  varianceAVG.select<4, 1>(4) = varianceAVG.select<4, 1>(0);
  varianceAVG.select<8, 1>(8) = varianceAVG.select<8, 1>(0);
  varianceAVG.select<16, 1>(16) = varianceAVG.select<16, 1>(0);
  varianceAVG.select<32, 1>(32) = varianceAVG.select<32, 1>(0);
  varianceAVG.select<64, 1>(64) = varianceAVG.select<64, 1>(0);
  // simd<float, 128> varianceAVG{varianceSum[0]};

  input = input / varianceAVG;

  input_FP16 = input;
  input_FP16 = input_FP16 * weight_FP16;

  if (hh < active_thread_num)
  {
    __ESIMD_ENS::lsc_block_store<
      fp16,
      128,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((fp16*)hidden_states_out + h * hidden_size + 128 * hh, input_FP16.select<128, 1>(0));
  }
}

ESIMD_INLINE void residual_rmsNorm128PerThread(uint8_t* weight, uint8_t* residual, uint8_t* hidden_states, uint8_t* hidden_states_out, int64_t hidden_size, int64_t input_len, int64_t add_residual, float variance_epsilon ,nd_item<1>& ndi) {

  __ESIMD_NS::slm_init(16 * sizeof(float)); 

  int h = ndi.get_group(0);
  if (h >= input_len) return;

  int hh = ndi.get_local_linear_id();
  uint32_t inputOffset = 128 * hh * sizeof(fp16);
  int active_thread_num = 16;

  simd<fp16, 128> input_FP16;
  simd<float, 128> input;
  simd<float, 128> input_powered;
  simd<fp16, 128> weight_FP16;
  simd<float, 16> variance = 0;
  simd<fp16, 128> residual_FP16;

  simd<float, 16> varianceSum = 0;

  // if (hh < 16)
  {
    input_FP16.template bit_cast_view<uint8_t>().template select<256, 1>(0) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)hidden_states + h * hidden_size * sizeof(fp16) + inputOffset);
    
    if (add_residual == 1)
    {
      residual_FP16.template bit_cast_view<uint8_t>().template select<256, 1>(0) =
            __ESIMD_ENS::lsc_block_load<
            uint8_t,
            256,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)residual + h * hidden_size * sizeof(fp16) + inputOffset);
    }

    weight_FP16.template bit_cast_view<uint8_t>().template select<256, 1>(0) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)weight + inputOffset);
  }
  // else
  // {
  //   input_FP16 = 0;
  //   weight_FP16 = 0;
  // }

  if (add_residual == 1)
  {
    input = input_FP16 + residual_FP16;
  }
  else
  {
    input = input_FP16;
  }

  // write back new residual
  if (hh < active_thread_num)
  {
    __ESIMD_ENS::lsc_block_store<
      fp16,
      128,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((fp16*)residual + h * hidden_size + 128 * hh, input.select<128, 1>(0));
  }

#pragma unroll
  for (int ll = 0; ll < 8; ll++) {
    input_powered.select<16, 1>(ll *16) = input.select<16, 1>(ll *16) * input.select<16, 1>(ll *16);// pow<float, 16, float>(input.select<16, 1>(ll *16), 2.0f);
    variance += input_powered.select<16, 1>(ll *16);
  }

  variance.select<8, 1>(0) = variance.select<8, 1>(0) + variance.select<8, 1>(8);
  variance.select<4, 1>(0) = variance.select<4, 1>(0) + variance.select<4, 1>(4);
  variance.select<2, 1>(0) = variance.select<2, 1>(0) + variance.select<2, 1>(2);
  variance[0] = variance[0] + variance[1];

  variance[0] = variance[0] / 128.0f;

  slm_block_store<float, 1>(hh * sizeof(float), variance[0]);

  barrier();
  
  varianceSum.select<16, 1>(0) = slm_block_load<float, 16>(0);
  
  varianceSum.select<8, 1>(0) = varianceSum.select<8, 1>(0) + varianceSum.select<8, 1>(8);
  varianceSum.select<4, 1>(0) = varianceSum.select<4, 1>(0) + varianceSum.select<4, 1>(4);
  varianceSum.select<2, 1>(0) = varianceSum.select<2, 1>(0) + varianceSum.select<2, 1>(2);
  varianceSum[0] = varianceSum[0] + varianceSum[1];

  varianceSum[0] = varianceSum[0] / 16.0f;
  // varianceSum[0] =  sqrt<float, 1, float>(varianceSum[0] + variance_epsilon);
  // varianceSum =  sqrt<float, 16, float>(varianceSum + variance_epsilon);
  // varianceSum[0] =  sqrt<float, 1, float>(varianceSum[0]);
  varianceSum[0] = sqrt(varianceSum[0] + variance_epsilon);

  simd<float, 128> varianceAVG = 0;
  // fill 128 length varianceAVG
  varianceAVG[0] = varianceSum[0];
  varianceAVG[1] = varianceAVG[0];
  varianceAVG.select<2, 1>(2) = varianceAVG.select<2, 1>(0);
  varianceAVG.select<4, 1>(4) = varianceAVG.select<4, 1>(0);
  varianceAVG.select<8, 1>(8) = varianceAVG.select<8, 1>(0);
  varianceAVG.select<16, 1>(16) = varianceAVG.select<16, 1>(0);
  varianceAVG.select<32, 1>(32) = varianceAVG.select<32, 1>(0);
  varianceAVG.select<64, 1>(64) = varianceAVG.select<64, 1>(0);
  // simd<float, 128> varianceAVG{varianceSum[0]};

  input = input / varianceAVG;

  input_FP16 = input;
  input_FP16 = input_FP16 * weight_FP16;

  // if (hh < 16)
  {
    __ESIMD_ENS::lsc_block_store<
      fp16,
      128,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((fp16*)hidden_states_out + h * hidden_size + 128 * hh, input_FP16.select<128, 1>(0));
  }
}


ESIMD_INLINE void residual_rmsNorm128PerThread_32t(uint8_t* weight, uint8_t* residual, uint8_t* hidden_states, uint8_t* hidden_states_out, int64_t hidden_size, int64_t input_len, int64_t add_residual, float variance_epsilon ,nd_item<1>& ndi) {

  __ESIMD_NS::slm_init(32 * sizeof(float)); 

  int h = ndi.get_group(0);
  int hh = ndi.get_local_linear_id();
  uint32_t inputOffset = 128 * hh * sizeof(fp16);
  int active_thread_num = hidden_size / 128;
  if (h >= input_len) return;

  simd<fp16, 128> input_FP16;
  simd<float, 128> input;
  simd<float, 128> input_powered;
  simd<fp16, 128> weight_FP16;
  simd<float, 16> variance = 0;
  simd<fp16, 128> residual_FP16;

  simd<float, 32> varianceSum = 0;

  if (hh < active_thread_num)
  {
    input_FP16.template bit_cast_view<uint8_t>().template select<256, 1>(0) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)hidden_states + h * hidden_size * sizeof(fp16) + inputOffset);
    
    if (add_residual == 1)
    {
      residual_FP16.template bit_cast_view<uint8_t>().template select<256, 1>(0) =
            __ESIMD_ENS::lsc_block_load<
            uint8_t,
            256,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)residual + h * hidden_size * sizeof(fp16) + inputOffset);
    }

    weight_FP16.template bit_cast_view<uint8_t>().template select<256, 1>(0) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)weight + inputOffset);
  }
  else
  {
    input_FP16 = 0;
    weight_FP16 = 0;
    residual_FP16 = 0;
  }

  if (add_residual == 1)
  {
    input = input_FP16 + residual_FP16;
  }
  else
  {
    input = input_FP16;
  }
  
  // write back new residual
  if (hh < active_thread_num)
  {
    __ESIMD_ENS::lsc_block_store<
      fp16,
      128,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((fp16*)residual + h * hidden_size + 128 * hh, input.select<128, 1>(0));
  }

#pragma unroll
  for (int ll = 0; ll < 8; ll++) {
    input_powered.select<16, 1>(ll *16) = pow<float, 16, float>(input.select<16, 1>(ll *16), 2.0f);
    variance += input_powered.select<16, 1>(ll *16);
  }

  variance.select<8, 1>(0) += variance.select<8, 1>(8);
  variance.select<4, 1>(0) += variance.select<4, 1>(4);
  variance.select<2, 1>(0) += variance.select<2, 1>(2);
  variance[0] += variance[1];

  slm_block_store<float, 1>(hh * sizeof(float), variance[0]);

  barrier();
  
  varianceSum.select<32, 1>(0) = slm_block_load<float, 32>(0);
  
  varianceSum.select<16, 1>(0) += varianceSum.select<16, 1>(16);
  varianceSum.select<8, 1>(0) += varianceSum.select<8, 1>(8);
  varianceSum.select<4, 1>(0) += varianceSum.select<4, 1>(4);
  varianceSum.select<2, 1>(0) += varianceSum.select<2, 1>(2);
  varianceSum[0] += varianceSum[1];

  varianceSum[0] = varianceSum[0] / hidden_size;
  varianceSum[0] =  sqrt(varianceSum[0] + variance_epsilon);

  simd<float, 128> varianceAVG;
  // fill 128 length varianceAVG
  varianceAVG[0] = varianceSum[0];
  varianceAVG[1] = varianceAVG[0];
  varianceAVG.select<2, 1>(2) = varianceAVG.select<2, 1>(0);
  varianceAVG.select<4, 1>(4) = varianceAVG.select<4, 1>(0);
  varianceAVG.select<8, 1>(8) = varianceAVG.select<8, 1>(0);
  varianceAVG.select<16, 1>(16) = varianceAVG.select<16, 1>(0);
  varianceAVG.select<32, 1>(32) = varianceAVG.select<32, 1>(0);
  varianceAVG.select<64, 1>(64) = varianceAVG.select<64, 1>(0);
  // simd<float, 128> varianceAVG{varianceSum[0]};

  input = input / varianceAVG;

  input_FP16 = input;
  input_FP16 = input_FP16 * weight_FP16;

  if (hh < active_thread_num)
  {
    __ESIMD_ENS::lsc_block_store<
      fp16,
      128,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((fp16*)hidden_states_out + h * hidden_size + 128 * hh, input_FP16.select<128, 1>(0));
  }
}

