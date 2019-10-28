/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

#ifndef CUTOFF2_VAL
#define CUTOFF2_VAL 6.250000
#define CUTOFF_VAL 2.500000
#define CEIL_CUTOFF_VAL 3.000000
#define GRIDSIZE_VAL1 256
#define GRIDSIZE_VAL2 256
#define GRIDSIZE_VAL3 256
#define SIZE_XY_VAL 65536
#define ONE_OVER_CUTOFF2_VAL 0.160000
#endif

#ifndef DYN_LOCAL_MEM_SIZE
#define DYN_LOCAL_MEM_SIZE 1092
#endif

typedef struct{
  float real;
  float imag;
  float kX;
  float kY;
  float kZ;
  float sdc;
} ReconstructionSample;

__kernel void reorder_kernel(int n,
                               __global unsigned int* idxValue_g,
                               __global ReconstructionSample* samples_g,
                               __global ReconstructionSample* sortedSample_g){
  unsigned int index = get_global_id(0); //blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int old_index;
  ReconstructionSample pt;

  if (index < n){
    old_index = idxValue_g[index];
    pt = samples_g[old_index];
    sortedSample_g[index] = pt;
  }
}
