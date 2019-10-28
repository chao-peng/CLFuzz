#include "macros.h"

__kernel void
ComputePhiMag_GPU(__global float* phiR, __global float* phiI, __global float* phiMag, int numK, __global int* ocl_kernel_branch_triggered_recorder) {__local int my_ocl_kernel_branch_triggered_recorder[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  int indexK = get_global_id(0);
  float real = indexK;
  float imag = indexK;
  if (indexK < numK) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

    /*float*/ real = phiR[indexK];
    /*float*/ imag = phiI[indexK];
    phiMag[indexK] = real*real + imag*imag;
  }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}
for (int update_recorder_i = 0; update_recorder_i < 2; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
}
