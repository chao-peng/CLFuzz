//========================================================================================================================================================================================================200
//	DEFINE / INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	MAIN FUNCTION HEADER
//======================================================================================================================================================150

#include "./main.h"

//======================================================================================================================================================150
//	End
//======================================================================================================================================================150

//========================================================================================================================================================================================================200
//	Extract KERNEL
//========================================================================================================================================================================================================200



//========================================================================================================================================================================================================200
//	Prepare KERNEL
//========================================================================================================================================================================================================200

__kernel void
prepare_kernel(	long d_Ne,
				__global fp* d_I,											// pointer to output image (DEVICE GLOBAL MEMORY)
				__global fp* d_sums,										// pointer to input image (DEVICE GLOBAL MEMORY)
				__global fp* d_sums2, __global int* ocl_kernel_branch_triggered_recorder){__local int my_ocl_kernel_branch_triggered_recorder[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);


	// indexes
	int bx = get_group_id(0);												// get current horizontal block index
	int tx = get_local_id(0);												// get current horizontal thread index
	int ei = (bx*NUMBER_THREADS)+tx;										// unique thread id, more threads than actual elements !!!

	// copy input to output & log uncompress
	if(ei<d_Ne){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);
															// do only for the number of elements, omit extra threads

		d_sums[ei] = d_I[ei];
		d_sums2[ei] = d_I[ei]*d_I[ei];

	}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}

for (int update_recorder_i = 0; update_recorder_i < 2; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
}
