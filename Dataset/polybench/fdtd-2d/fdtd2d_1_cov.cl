/**
 * fdtd2d.cl: This file is part of the PolyBench/GPU 1.0 test suite.
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



__kernel void fdtd_kernel1(__global DATA_TYPE *_fict_, __global DATA_TYPE *ex, __global DATA_TYPE *ey, __global DATA_TYPE *hz, int t, int nx, int ny, __global int* ocl_kernel_branch_triggered_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[4];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 4; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	int j = get_global_id(0);
	int i = get_global_id(1);

	if ((i < nx) && (j < ny))
	{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

		int tid = i * ny + j;

		if (i == 0)
		{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);

			ey[i * ny + j] = _fict_[t];
		}
		else
		{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);

			ey[i * ny + j] = ey[i * ny + j] - 0.5*(hz[i * ny + j] - hz[(i-1) * ny + j]);
		}
	}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}
for (int update_recorder_i = 0; update_recorder_i < 4; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
}
