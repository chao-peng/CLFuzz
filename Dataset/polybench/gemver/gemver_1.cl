/**
 * gemver.cl: This file is part of the PolyBench/GPU 1.0 test suite.
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



__kernel void gemver_kernel1(__global DATA_TYPE *A, __global DATA_TYPE *V1, __global DATA_TYPE *V2, __global DATA_TYPE *U1, __global DATA_TYPE *U2, int n)
{
	int j = get_global_id(0);
	int i = get_global_id(1);

	if ((i < n) && (j < n))
	{
		A[i*n + j] += U1[i] * V1[j] + U2[i] * V2[j];
	}
}
