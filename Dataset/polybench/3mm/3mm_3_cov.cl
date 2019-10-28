/**
 * 3mm.cl: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

typedef float DATA_TYPE;



__kernel void mm3_kernel3(__global DATA_TYPE *E, __global DATA_TYPE *F, __global DATA_TYPE *G, int ni, int nl, int nj, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_kernel_loop_recorder)
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

	int j = get_global_id(0);
	int i = get_global_id(1);

	if ((i < ni) && (j < nl))
	{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

		int k;
		G[i*nl + j] = 0;
		private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
for(k=0; k < nj || (private_ocl_kernel_loop_boundary_not_reached[0] = false); k++)
		{
private_ocl_kernel_loop_iter_counter[0]++;

			G[i * nl + j] += E[i * nj + k] * F[k * nl + j];
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
	}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}
for (int update_recorder_i = 0; update_recorder_i < 2; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
for (int update_recorder_i = 0; update_recorder_i < 1; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}
