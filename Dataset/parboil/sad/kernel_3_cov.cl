/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#ifndef MAX_POS
#define MAX_POS 1089
#define CEIL_POS 61
#define POS_PER_THREAD 18
#define MAX_POS_PADDED 1096
#define THREADS_W 1
#define THREADS_H 1
#define SEARCH_RANGE 16
#define SEARCH_DIMENSION 33
#endif

/* The compute kernel. */
/* The macros THREADS_W and THREADS_H specify the width and height of the
 * area to be processed by one thread, measured in 4-by-4 pixel blocks.
 * Larger numbers mean more computation per thread block.
 *
 * The macro POS_PER_THREAD specifies the number of search positions for which
 * an SAD is computed.  A larger value indicates more computation per thread,
 * and fewer threads per thread block.  It must be a multiple of 3 and also
 * must be at most 33 because the loop to copy from shared memory uses
 * 32 threads per 4-by-4 pixel block.
 *
 */

// AMD OpenCL fails UINT_CUDA_V
#define SHORT2_V 0
#define UINT_CUDA_V 0

// Either works
#define VEC_LOAD 0

// CAST_STORE is only method that works for all implementations of OpenCL tested
#define VEC_STORE 0
#define CAST_STORE 0
#define SCALAR_STORE 1





__kernel void larger_sad_calc_16(__global unsigned short *blk_sad,
				   int mb_width,
				   int mb_height, __global int* ocl_kernel_loop_recorder)
{__local int my_ocl_kernel_loop_recorder[1];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 1; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[1];
bool private_ocl_kernel_loop_boundary_not_reached[1];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  // Macroblock coordinates
  int mb_x = get_group_id(0);
  int mb_y = get_group_id(1);
  int search_pos = get_local_id(0);

  // Number of macroblocks in a frame
  int macroblocks = mul24(mb_width, mb_height) * MAX_POS_PADDED;
  int macroblock_index = (mul24(mb_y, mb_width) + mb_x) * MAX_POS_PADDED;

  __global unsigned short *bi;
  __global unsigned short *bo_3, *bo_2, *bo_1;

  //bi = blk_sad + macroblocks * 5 + macroblock_index * 4;
  bi = blk_sad + ((macroblocks + macroblock_index) << 2) + macroblocks;

  // Block type 3: 8x16
  //bo_3 = blk_sad + macroblocks * 3 + macroblock_index * 2;
  bo_3 = blk_sad + ((macroblocks + macroblock_index) << 1) + macroblocks;

  // Block type 5: 8x4
  bo_2 = blk_sad + macroblocks + macroblock_index * 2;

  // Block type 4: 8x8
  bo_1 = blk_sad + macroblock_index;

  private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
for ( ; search_pos < (MAX_POS+1)/2 || (private_ocl_kernel_loop_boundary_not_reached[0] = false); search_pos += 32)
    {
private_ocl_kernel_loop_iter_counter[0]++;

#if SHORT2_V
  #if VEC_LOAD
      ushort2 s00 = vload2(search_pos,                    bi);
      ushort2 s01 = vload2(search_pos+  MAX_POS_PADDED/2, bi);
      ushort2 s10 = vload2(search_pos+2*MAX_POS_PADDED/2, bi);
      ushort2 s11 = vload2(search_pos+3*MAX_POS_PADDED/2, bi);
  #else
      ushort2 s00 = (ushort2) (bi[search_pos*2], bi[search_pos*2+1]);
      ushort2 s01 = (ushort2) (bi[(search_pos + MAX_POS_PADDED/2)*2], bi[(search_pos + MAX_POS_PADDED/2)*2+1]);
      ushort2 s10 = (ushort2) (bi[(search_pos + 2*MAX_POS_PADDED/2)*2], bi[(search_pos + 2*MAX_POS_PADDED/2)*2+1]);
      ushort2 s11 = (ushort2) (bi[(search_pos + 3*MAX_POS_PADDED/2)*2], bi[(search_pos + 3*MAX_POS_PADDED/2)*2+1]);
  #endif

  #if VEC_STORE
      ushort2 s0010 = s00 + s10;
      ushort2 s0111 = s01 + s11;
      ushort2 s0001 = s00 + s01;
      ushort2 s1011 = s10 + s11;
      ushort2 s00011011 = s0001 + s1011;

      vstore2(s0010, search_pos, bo_3);
      vstore2(s0111, search_pos+MAX_POS_PADDED/2, bo_3);
      vstore2(s0001, search_pos, bo_2);
      vstore2(s1011, search_pos+MAX_POS_PADDED/2, bo_2);
      vstore2(s00011011, search_pos, bo_1);
  #elif CAST_STORE
      ((__global ushort2 *)bo_3)[search_pos]                  = s00 + s10;
      ((__global ushort2 *)bo_3)[search_pos+MAX_POS_PADDED/2] = s01 + s11;
      ((__global ushort2 *)bo_2)[search_pos]                  = s00 + s01;
      ((__global ushort2 *)bo_2)[search_pos+MAX_POS_PADDED/2] = s10 + s11;
      ((__global ushort2 *)bo_1)[search_pos]                  = (s00 + s01) + (s10 + s11);
  #else // SCALAR_STORE
      bo_3[search_pos*2] = s00.x + s10.x;
      bo_3[search_pos*2+1] = s00.y + s10.y;
      bo_3[(search_pos+MAX_POS_PADDED/2)*2] = s01.x + s11.x;
      bo_3[(search_pos+MAX_POS_PADDED/2)*2+1] = s01.y + s11.y;
      bo_2[search_pos*2] = s00.x + s01.x;
      bo_2[search_pos*2+1] = s00.y + s01.y;
      bo_2[(search_pos+MAX_POS_PADDED/2)*2] = s10.x + s11.x;
      bo_2[(search_pos+MAX_POS_PADDED/2)*2+1] = s10.y + s11.y;
      bo_1[search_pos*2] = (s00.x + s01.x) + (s10.x + s11.x);
      bo_1[search_pos*2+1] = (s00.y + s01.y) + (s10.y + s11.y);
  #endif
#else // UINT_CUDA_V
      uint i00 = ((__global uint *)bi)[search_pos];
      uint i01 = ((__global uint *)bi)[search_pos + MAX_POS_PADDED/2];
      uint i10 = ((__global uint *)bi)[search_pos + 2*MAX_POS_PADDED/2];
      uint i11 = ((__global uint *)bi)[search_pos + 3*MAX_POS_PADDED/2];

      ((__global uint *)bo_3)[search_pos]                  = i00 + i10;
      ((__global uint *)bo_3)[search_pos+MAX_POS_PADDED/2] = i01 + i11;
      ((__global uint *)bo_2)[search_pos]                  = i00 + i01;
      ((__global uint *)bo_2)[search_pos+MAX_POS_PADDED/2] = i10 + i11;
      ((__global uint *)bo_1)[search_pos]                  = (i00 + i01) + (i10 + i11);
#endif
    }
if (private_ocl_kernel_loop_iter_counter[0] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[0], 1);
}if (private_ocl_kernel_loop_iter_counter[0] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[0], 2);
}if (private_ocl_kernel_loop_iter_counter[0] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[0], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[0]) {
    atomic_or(&my_ocl_kernel_loop_recorder[0], 8);
}
for (int update_recorder_i = 0; update_recorder_i < 1; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}
