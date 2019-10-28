/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include "common.h"

__kernel void naive_kernel(float c0,float c1,__global float* A0,__global float *Anext,int nx,int ny,int nz, __global int* ocl_kernel_branch_triggered_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    	int i = get_global_id(0)+1;
    	int j = get_global_id(1)+1;
    	int k = get_global_id(2)+1;

if(i<nx-1)
{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

		Anext[Index3D (nx, ny, i, j, k)] = c1 *
		( A0[Index3D (nx, ny, i, j, k + 1)] +
		  A0[Index3D (nx, ny, i, j, k - 1)] +
		  A0[Index3D (nx, ny, i, j + 1, k)] +
		  A0[Index3D (nx, ny, i, j - 1, k)] +
	 	  A0[Index3D (nx, ny, i + 1, j, k)] +
		  A0[Index3D (nx, ny, i - 1, j, k)] )
		- A0[Index3D (nx, ny, i, j, k)] * c0;
}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}
for (int update_recorder_i = 0; update_recorder_i < 2; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
}
