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



__kernel void fdtd_kernel2(__global DATA_TYPE *ex, __global DATA_TYPE *ey, __global DATA_TYPE *hz, int nx, int ny)
{
	int j = get_global_id(0);
	int i = get_global_id(1);

	if ((i < nx) && (j < ny) && (j > 0))
	{
		ex[i * ny + j] = ex[i * ny + j] - 0.5*(hz[i * ny + j] - hz[i * ny + (j-1)]);
	}
}
