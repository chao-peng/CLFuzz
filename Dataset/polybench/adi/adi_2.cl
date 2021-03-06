/**
 * adi.cl: This file is part of the PolyBench/GPU 1.0 test suite.
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




__kernel void adi_kernel2(__global DATA_TYPE* A, __global DATA_TYPE* B, __global DATA_TYPE* X, int n)
{
	int i1 = get_global_id(0);

	if ((i1 < n))
	{
		X[i1*n + (n-1)] = X[i1*n + (n-1)] / B[i1*n + (n-1)];
	}
}
