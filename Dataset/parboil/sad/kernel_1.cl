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

__kernel void mb_sad_calc(__global unsigned short *blk_sad,
                            __global unsigned short *frame,
                            int mb_width,
                            int mb_height,
                            __global unsigned short* img_ref) // __read_only image2d_t img_ref)
{
  int tx = (get_local_id(0) / CEIL_POS) % THREADS_W;
  int ty = (get_local_id(0) / CEIL_POS) / THREADS_W;
  int bx = get_group_id(0);
  int by = get_group_id(1);
  int img_width = mb_width*16;
  int lidx = get_local_id(0);

  // Macroblock and sub-block coordinates
  int mb_x = (tx + bx * THREADS_W) >> 2;
  int mb_y = (ty + by * THREADS_H) >> 2;
  int block_x = (tx + bx * THREADS_W) & 0x03;
  int block_y = (ty + by * THREADS_H) & 0x03;

  // If this thread is assigned to an invalid 4x4 block, do nothing
  if ((mb_x < mb_width) && (mb_y < mb_height))
    {
      // Pixel offset of the origin of the current 4x4 block
      int frame_x = ((mb_x << 2) + block_x) << 2;
      int frame_y = ((mb_y << 2) + block_y) << 2;

      // Origin of the search area for this 4x4 block
      int ref_x = frame_x - SEARCH_RANGE;
      int ref_y = frame_y - SEARCH_RANGE;

      // Origin in the current frame for this 4x4 block
      int cur_o = frame_y * img_width + frame_x;

      int search_pos;
      int search_pos_base =
        (lidx % CEIL_POS) * POS_PER_THREAD;
      int search_pos_end = search_pos_base + POS_PER_THREAD;

      // Don't go past bounds
      if (search_pos_end > MAX_POS) {
        search_pos_end = MAX_POS;
      }

      // For each search position, within the range allocated to this thread
      for (search_pos = search_pos_base;
           search_pos < search_pos_end;
           search_pos++) {
        unsigned short sad4x4 = 0;
        int search_off_x = ref_x + (search_pos % SEARCH_DIMENSION);
        int search_off_y = ref_y + (search_pos / SEARCH_DIMENSION);

        // 4x4 SAD computation
        for(int y=0; y<4; y++) {
          for (int x=0; x<4; x++) {

          // ([unsigned] short)read_imageui or
          //                   read_imagei  is required for correct calculation.
          // Though read_imagei() is shorter, its results are undefined by specification since the input
          // is an unsigned type, CL_UNSIGNED_INT16

            int sx = search_off_x + x;
            sx = (sx < 0) ? 0 : sx;
            sx = (sx >= img_width) ? img_width - 1 : sx;
            int sy = search_off_y + y;
            sy = (sy < 0) ? 0 : sy;
            sy = (sy >= mb_height * 16) ? mb_height * 16 - 1 : sy;
            sad4x4 += abs((unsigned short) img_ref[(sx) + (sy) * img_width]  -
                  frame[cur_o + y * img_width + x]);
          }
        }

        // Save this value into the local SAD array
        blk_sad[mb_width * mb_height * MAX_POS_PADDED * (9 + 16) +
        (mb_y * mb_width + mb_x) * MAX_POS_PADDED * 16 +
        (4 * block_y + block_x) * MAX_POS_PADDED+search_pos] = sad4x4;
      }
    }

}


//typedef unsigned int uint;
