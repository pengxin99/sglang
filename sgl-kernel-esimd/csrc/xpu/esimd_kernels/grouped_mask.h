#include "utils.h"

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using fp16 = sycl::half;
using namespace sycl;


ESIMD_INLINE void GroupMask_Ashape(uint8_t* mask, uint8_t* grouped_mask, int mask_shape_0, int mask_shape_1, int mask_stride_1, int mask_g_shape_0, int mask_g_shape_1, int block_size, float neg_inf, nd_item<2>& ndi) {
  int h = ndi.get_group(0); // [0, mask_shape_0 // 16)
  int v = ndi.get_group(1); // [0, mask_shape_1 // 16 // 16)
  int GroupRange_v = ndi.get_group_range(1);
  int localRange_v = ndi.get_local_range(1);
  int localLinearId = ndi.get_local_id(1); // [0, 16)

  simd<fp16, 16> in_up;
  simd<fp16, 16> in_down;
  simd<int8_t, 16> out;
  __ESIMD_NS::slm_init(16 * sizeof(int8_t));

  unsigned int maskOffset_v = v * block_size * localRange_v + localLinearId * block_size;
  unsigned int maskOffset_up = (h * block_size * mask_stride_1 + maskOffset_v);
  unsigned int maskOffset_down = ((h * block_size + block_size - 2) * mask_stride_1 + maskOffset_v);
  unsigned int outputOffset = (h * GroupRange_v * localRange_v + v * localRange_v) * sizeof(int8_t);
  unsigned int slmOffset = localLinearId * 1;
  
  if (maskOffset_v < mask_shape_1)
  {
    in_up.template bit_cast_view<fp16>().template select<16, 1>(0) =
      __ESIMD_ENS::lsc_block_load<
      fp16,
      16,
      __ESIMD_ENS::lsc_data_size::u16,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::uncached>((fp16*)mask + maskOffset_up);


    in_down.template bit_cast_view<fp16>().template select<16, 1>(0) =
      __ESIMD_ENS::lsc_block_load<
      fp16,
      16,
      __ESIMD_ENS::lsc_data_size::u16,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::uncached>((fp16*)mask + maskOffset_down);

    fp16 in_right_up = in_up.select<1,1>(15);
    fp16 in_left_down = in_down.select<1,1>(0);

    if (in_right_up == (fp16)neg_inf && in_left_down == (fp16)neg_inf)
    {
      slm_block_store<int8_t, 1>(slmOffset, -1);
    }
    else if (in_right_up == 0.0 && in_left_down == (fp16)neg_inf)
    {
      slm_block_store<int8_t, 1>(slmOffset, 1);
    }
    else if (in_right_up == (fp16)neg_inf && in_left_down == 0.0)
    {
      slm_block_store<int8_t, 1>(slmOffset, 2);
    }
    else{
      slm_block_store<int8_t, 1>(slmOffset, 0);
    }
  }else{
      slm_block_store<int8_t, 1>(slmOffset, -1);
  }

  // out.select<1, 1>(0) = GroupRange_v;
  // out.select<1, 1>(2) = localRange_v;
  // out.select<1, 1>(4) = localLinearId;
  // slm_block_store<int8_t, 1>(0, 0);
  // slm_block_store<int8_t, 1>(1, 1);
  // slm_block_store<int8_t, 1>(2, 2);
  // slm_block_store<int8_t, 1>(5, 7);
  // slm_block_store<int8_t, 1>(slmOffset, localLinearId);
  

  barrier();

  if (localLinearId == 0)
  {
    out.select<16, 1>(0) = slm_block_load<int8_t, 16>(0);
    // out.select<16, 1>(0) = in_up.select<16,1>(0);
    __ESIMD_ENS::lsc_block_store<
    int8_t,
    16,
    __ESIMD_ENS::lsc_data_size::u8,
    __ESIMD_ENS::cache_hint::write_back,
    __ESIMD_ENS::cache_hint::write_back
    >((int8_t*)grouped_mask + outputOffset, out.select<16, 1>(0));
  
  }

}



ESIMD_INLINE void GroupMask_Ashape_no_ori_mask(uint8_t* grouped_mask, int mask_shape_0, int mask_shape_1, int mask_stride_1, int mask_g_shape_0, int mask_g_shape_1, int block_size, int init_size, int local_size, float neg_inf, nd_item<2>& ndi) {
  int h = ndi.get_group(0); // [0, mask_shape_0 // 16)
  int v = ndi.get_group(1); // [0, mask_shape_1 // 16 // 16)
  int GroupRange_v = ndi.get_group_range(1);
  int localRange_v = ndi.get_local_range(1);
  int localLinearId = ndi.get_local_id(1); // [0, 16)

  simd<fp16, 16> in_up;
  simd<fp16, 16> in_down;
  simd<int8_t, 16> out;
  __ESIMD_NS::slm_init(16 * sizeof(int8_t));

  unsigned int maskOffset_v = v * block_size * localRange_v + localLinearId * block_size;
  unsigned int maskOffset_up = (h * block_size * mask_stride_1 + maskOffset_v);
  unsigned int maskOffset_down = ((h * block_size + block_size - 2) * mask_stride_1 + maskOffset_v);
  unsigned int outputOffset = (h * GroupRange_v * localRange_v + v * localRange_v) * sizeof(int8_t);
  unsigned int slmOffset = localLinearId * 1;
  
  unsigned int row_idx = h * block_size;
  unsigned int vertical_idx = v * block_size * localRange_v + localLinearId * block_size;

  if (maskOffset_v < mask_shape_1)
  {
    if (row_idx == vertical_idx)
    {
      slm_block_store<int8_t, 1>(slmOffset, 2);
    }
    else if (row_idx > vertical_idx && vertical_idx < init_size)
    {
      slm_block_store<int8_t, 1>(slmOffset, 0);
    }
    else if (row_idx - vertical_idx == local_size)
    {
      slm_block_store<int8_t, 1>(slmOffset, 1);
    }
    else if (row_idx - vertical_idx < local_size && row_idx > vertical_idx)
    {
      slm_block_store<int8_t, 1>(slmOffset, 0);
    }
    else{
      slm_block_store<int8_t, 1>(slmOffset, -1);
    }
  }else{
      slm_block_store<int8_t, 1>(slmOffset, -1);
  }
  // out.select<1, 1>(0) = GroupRange_v;
  // out.select<1, 1>(2) = localRange_v;
  // out.select<1, 1>(4) = localLinearId;
  // slm_block_store<int8_t, 1>(0, 0);
  // slm_block_store<int8_t, 1>(1, 1);
  // slm_block_store<int8_t, 1>(2, 2);
  // slm_block_store<int8_t, 1>(5, 7);
  // slm_block_store<int8_t, 1>(slmOffset, localLinearId);
  

  barrier();

  if (localLinearId == 0)
  {
    out.select<16, 1>(0) = slm_block_load<int8_t, 16>(0);
    // out.select<16, 1>(0) = in_up.select<16,1>(0);
    __ESIMD_ENS::lsc_block_store<
    int8_t,
    16,
    __ESIMD_ENS::lsc_data_size::u8,
    __ESIMD_ENS::cache_hint::write_back,
    __ESIMD_ENS::cache_hint::write_back
    >((int8_t*)grouped_mask + outputOffset, out.select<16, 1>(0));
  
  }

}


/**
 * def draw_lines_on_matrix(matrix, vertical_lines, slash_lines):
    """
    创建一个 1024x1024 的矩阵，根据给定的竖线和斜率为 -1 的斜线着色。

    Args:
        vertical_lines: 一个包含竖线 x 坐标的列表。
        slash_lines: 一个列表，包含以矩阵的列方向为起始点的斜线起始 x 坐标。
                                  斜率固定为 -1。 假设起始 y 坐标为 0。

    Returns:
        一个 1024x1024 的 NumPy 矩阵，着色处为 1，其余为 0。
    """

    # matrix = torch.zeros((M, N), dtype=torch.uint8)  # 使用 uint8 节省内存
    M, N = matrix.shape
    vertical_lines = vertical_lines.reshape(-1)
    slash_lines = slash_lines.reshape(-1)

    # 绘制竖线
    for x in vertical_lines:
        if 0 <= x < N:  # 确保 x 坐标在范围内
            matrix[:, x] = 1  # 将整列设置为 1
    print(f"=== [draw_lines_on_matrix] vertical_lines down! ")
    # 绘制斜率为 -1 的斜线
    print(f"=== slash_lines: {slash_lines}")
    for start_y in slash_lines:
        if 0 <= start_y < M:  # 确保起始 x 坐标在范围内
            # print(f"=== [draw_lines_on_matrix] slash_lines down! {start_y}")
            start_x = 0
            while 0 <= start_y < M and 0 <= start_x < N:
                # print(start_y, start_x)
                matrix[start_y, start_x] = 1
                start_y += 1
                start_x += 1

    return matrix
*/
// ESIMD_INLINE void GroupMask_Bshape(uint8_t* grouped_mask, int grouped_mask_size, uint8_t* vertical_indexs, uint8_t* slash_indexs, int vertical_size, int slash_size, nd_item<2>& ndi) {
//   int group_1 = ndi.get_group(0); // [0, slash_index)
//   int group_2 = ndi.get_group(1); // [0, grouped_mask_size / 16)
//   int itemRange_1 = ndi.get_local_range(0);
//   int itemRange_2 = ndi.get_local_range(1);
//   int item_1 = ndi.get_local_id(0); // [0, grouped_mask_size)
//   int item_2 = ndi.get_local_id(1); // [0, grouped_mask_size)
//   // int GroupRange_v = ndi.get_group_range(1);
//   // int localRange_v = ndi.get_local_range(1);
//   // int localLinearId = ndi.get_local_id(1); // [0, 16)

//   if (group_2 * itemRange_2 < vertical_size && group_1 == 0 && item_1 == 0)
//   {
//     uint32_t vertical_idx = ((uint64_t*)vertical_indexs)[group_2];
//     for (size_t i = 0; i < grouped_mask_size; i++)
//     {
//       if (i >= vertical_idx)
//         ((uint8_t*)grouped_mask)[i * grouped_mask_size + vertical_idx] = 1;
//     }

//   }

//   if (group_2 * itemRange_2 < slash_size && group_1 == 0 && item_1 == 0)
//   {
//     uint32_t slash_idx = ((uint64_t*)slash_indexs)[group_2];

//     for (size_t i = 0; i < grouped_mask_size; i++)
//     {
//       uint32_t row = slash_idx + i;
//       uint32_t col = 0 + i;
//       if (row < grouped_mask_size && col < grouped_mask_size && row >= col)
//       {
//         ((uint8_t*)grouped_mask)[row * grouped_mask_size + col] = 1;
//       }
//     }
//   }
// }

ESIMD_INLINE void GroupMask_Bshape_multi_head(uint8_t* grouped_mask, int grouped_mask_size, int grouped_mask_head_stride, uint8_t* vertical_indexs, uint8_t* slash_indexs, int vertical_size, int slash_size, nd_item<2>& ndi) {
  int group_1 = ndi.get_group(0); // [0, head_num)
  int group_2 = ndi.get_group(1); // [0, grouped_mask_size / 16)
  int itemRange_1 = ndi.get_local_range(0);   // head_num
  int itemRange_2 = ndi.get_local_range(1);
  int item_1 = ndi.get_local_id(0); // [0, head_num)
  int item_2 = ndi.get_local_id(1); // [0, grouped_mask_size)

  int grouped_mask_head_offset = group_1 * grouped_mask_head_stride;
  int vertical_head_offset = group_1 * vertical_size;
  int slash_head_offset = group_1 * slash_size;

  if (group_2 * itemRange_2 < vertical_size)
  {
    uint32_t vertical_idx = ((uint64_t*)vertical_indexs)[vertical_head_offset + group_2];
    for (size_t i = 0; i < grouped_mask_size; i++)
    {
      if (i == vertical_idx)
        ((uint8_t*)grouped_mask)[grouped_mask_head_offset + i * grouped_mask_size + vertical_idx] = 2;
      else if (i > vertical_idx)
        ((uint8_t*)grouped_mask)[grouped_mask_head_offset + i * grouped_mask_size + vertical_idx] = 0;
    }
  }

  if (group_2 * itemRange_2 < slash_size)
  {
    uint32_t slash_idx = ((uint64_t*)slash_indexs)[slash_head_offset + group_2];

    for (size_t i = 0; i < grouped_mask_size; i++)
    {
      uint32_t row = slash_idx + i;
      uint32_t col = 0 + i;
      if (row < grouped_mask_size && col < grouped_mask_size && row >= col)
      {
        if (row == col)
          ((uint8_t*)grouped_mask)[grouped_mask_head_offset + row * grouped_mask_size + col] = 2;
        else
          ((uint8_t*)grouped_mask)[grouped_mask_head_offset + row * grouped_mask_size + col] = 0;
      }
    }
  }
}
