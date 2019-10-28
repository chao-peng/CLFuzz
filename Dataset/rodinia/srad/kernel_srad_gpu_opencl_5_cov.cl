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

__kernel void
srad2_kernel(	fp d_lambda,
				int d_Nr,
				int d_Nc,
				long d_Ne,
				__global int* d_iN,
				__global int* d_iS,
				__global int* d_jE,
				__global int* d_jW,
				__global fp* d_dN,
				__global fp* d_dS,
				__global fp* d_dE,
				__global fp* d_dW,
				__global fp* d_c,
				__global fp* d_I, __global int* ocl_kernel_branch_triggered_recorder){__local int my_ocl_kernel_branch_triggered_recorder[4];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 4; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);


	// indexes
    int bx = get_group_id(0);												// get current horizontal block index
	int tx = get_local_id(0);												// get current horizontal thread index
	int ei = bx*NUMBER_THREADS+tx;											// more threads than actual elements !!!
	int row;																// column, x position
	int col;																// row, y position

	// variables
	fp d_cN,d_cS,d_cW,d_cE;
	fp d_D;

	// figure out row/col location in new matrix
	row = (ei+1) % d_Nr - 1;												// (0-n) row
	col = (ei+1) / d_Nr + 1 - 1;											// (0-n) column
	if((ei+1) % d_Nr == 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

		row = d_Nr - 1;
		col = col - 1;
	}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}

	if(ei<d_Ne){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);
															// make sure that only threads matching jobs run

		// diffusion coefficent
		d_cN = d_c[ei];														// north diffusion coefficient
		d_cS = d_c[d_iS[row] + d_Nr*col];										// south diffusion coefficient
		d_cW = d_c[ei];														// west diffusion coefficient
		d_cE = d_c[row + d_Nr * d_jE[col]];									// east diffusion coefficient

		// divergence (equ 58)
		d_D = d_cN*d_dN[ei] + d_cS*d_dS[ei] + d_cW*d_dW[ei] + d_cE*d_dE[ei];// divergence

		// image update (equ 61) (every element of IMAGE)
		d_I[ei] = d_I[ei] + 0.25*d_lambda*d_D;								// updates image (based on input time step and divergence)

	}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);
}

for (int update_recorder_i = 0; update_recorder_i < 4; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
}

//========================================================================================================================================================================================================200
//	Compress KERNEL
//========================================================================================================================================================================================================200


//========================================================================================================================================================================================================200
//	End
//========================================================================================================================================================================================================200
