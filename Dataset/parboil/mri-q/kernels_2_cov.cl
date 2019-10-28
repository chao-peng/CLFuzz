#include "macros.h"



__kernel void
ComputeQ_GPU(int numK, int kGlobalIndex,
	     __global float* x, __global float* y, __global float* z,
	     __global float* Qr, __global float* Qi, __global struct kValues* ck, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_kernel_loop_recorder) 
{__local int my_ocl_kernel_branch_triggered_recorder[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int my_ocl_kernel_loop_recorder[1];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 1; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[1];
bool private_ocl_kernel_loop_boundary_not_reached[1];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  float sX;
  float sY;
  float sZ;
  float sQr;
  float sQi;

  // Determine the element of the X arrays computed by this thread
  int xIndex = get_group_id(0)*KERNEL_Q_THREADS_PER_BLOCK + get_local_id(0);

  // Read block's X values from global mem to shared mem
  sX = x[xIndex];
  sY = y[xIndex];
  sZ = z[xIndex];
  sQr = Qr[xIndex];
  sQi = Qi[xIndex];

  private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
for (int kIndex = 0; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID) || (private_ocl_kernel_loop_boundary_not_reached[0] = false); kIndex++) {
private_ocl_kernel_loop_iter_counter[0]++;

    if (kGlobalIndex + kIndex < numK) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

      float expArg;
      expArg = PIx2 * (ck[kIndex].Kx * sX +
			   ck[kIndex].Ky * sY +
			   ck[kIndex].Kz * sZ);
      sQr = sQr + ck[kIndex].PhiMag * cos(expArg); // native_cos(expArg);
      sQi = sQi + ck[kIndex].PhiMag * sin(expArg); // native_sin(expArg);
    }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}
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

  Qr[xIndex] = sQr;
  Qi[xIndex] = sQi;
for (int update_recorder_i = 0; update_recorder_i < 2; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
for (int update_recorder_i = 0; update_recorder_i < 1; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}
