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

__kernel void binning_kernel (unsigned int n,
                              __global ReconstructionSample* sample_g,
                              __global unsigned int* idxKey_g,
                              __global unsigned int* idxValue_g,
                              __global unsigned int* binCount_g,
                              unsigned int binsize, unsigned int gridNumElems, __global int* ocl_kernel_branch_triggered_recorder){__local int my_ocl_kernel_branch_triggered_recorder[4];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 4; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  unsigned int key;
  unsigned int sampleIdx = get_global_id(0); //blockIdx.x*blockDim.x+threadIdx.x;
  ReconstructionSample pt;
  unsigned int binIdx;
  unsigned int count;

  if (sampleIdx < n){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

    pt = sample_g[sampleIdx];

    binIdx = (unsigned int)(pt.kZ)*((int) ( SIZE_XY_VAL )) + (unsigned int)(pt.kY)*((int)( GRIDSIZE_VAL1 )) + (unsigned int)(pt.kX);

    count = atom_add(binCount_g+binIdx, 1);
    if (count < binsize){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);

      key = binIdx;
    } else {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);

      atom_sub(binCount_g+binIdx, 1);
      key = gridNumElems;
    }

    idxKey_g[sampleIdx] = key;
    idxValue_g[sampleIdx] = sampleIdx;
  }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}
for (int update_recorder_i = 0; update_recorder_i < 4; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
}
