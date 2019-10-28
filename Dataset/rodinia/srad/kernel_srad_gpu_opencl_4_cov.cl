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
//	SRAD KERNEL
//========================================================================================================================================================================================================200


__kernel void
srad_kernel(fp d_lambda,
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
			fp d_q0sqr,
			__global fp* d_c,
			__global fp* d_I, __global int* ocl_kernel_branch_triggered_recorder){__local int my_ocl_kernel_branch_triggered_recorder[8];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 8; ++ocl_kernel_init_i) {
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
	fp d_Jc;
	fp d_dN_loc, d_dS_loc, d_dW_loc, d_dE_loc;
	fp d_c_loc;
	fp d_G2,d_L,d_num,d_den,d_qsqr;

	// figure out row/col location in new matrix
	row = (ei+1) % d_Nr - 1;													// (0-n) row
	col = (ei+1) / d_Nr + 1 - 1;												// (0-n) column
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

		// directional derivatives, ICOV, diffusion coefficent
		d_Jc = d_I[ei];														// get value of the current element

		// directional derivates (every element of IMAGE)(try to copy to shared memory or temp files)
		d_dN_loc = d_I[d_iN[row] + d_Nr*col] - d_Jc;						// north direction derivative
		d_dS_loc = d_I[d_iS[row] + d_Nr*col] - d_Jc;						// south direction derivative
		d_dW_loc = d_I[row + d_Nr*d_jW[col]] - d_Jc;						// west direction derivative
		d_dE_loc = d_I[row + d_Nr*d_jE[col]] - d_Jc;						// east direction derivative

		// normalized discrete gradient mag squared (equ 52,53)
		d_G2 = (d_dN_loc*d_dN_loc + d_dS_loc*d_dS_loc + d_dW_loc*d_dW_loc + d_dE_loc*d_dE_loc) / (d_Jc*d_Jc);	// gradient (based on derivatives)

		// normalized discrete laplacian (equ 54)
		d_L = (d_dN_loc + d_dS_loc + d_dW_loc + d_dE_loc) / d_Jc;			// laplacian (based on derivatives)

		// ICOV (equ 31/35)
		d_num  = (0.5*d_G2) - ((1.0/16.0)*(d_L*d_L)) ;						// num (based on gradient and laplacian)
		d_den  = 1 + (0.25*d_L);												// den (based on laplacian)
		d_qsqr = d_num/(d_den*d_den);										// qsqr (based on num and den)

		// diffusion coefficent (equ 33) (every element of IMAGE)
		d_den = (d_qsqr-d_q0sqr) / (d_q0sqr * (1+d_q0sqr)) ;				// den (based on qsqr and q0sqr)
		d_c_loc = 1.0 / (1.0+d_den) ;										// diffusion coefficient (based on den)

		// saturate diffusion coefficent to 0-1 range
		if (d_c_loc < 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[4], 1);
													// if diffusion coefficient < 0
			d_c_loc = 0;													// ... set to 0
		}
		else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[5], 1);

if (d_c_loc > 1){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[6], 1);
												// if diffusion coefficient > 1
			d_c_loc = 1;													// ... set to 1
		}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[7], 1);
}
}

		// save data to global memory
		d_dN[ei] = d_dN_loc;
		d_dS[ei] = d_dS_loc;
		d_dW[ei] = d_dW_loc;
		d_dE[ei] = d_dE_loc;
		d_c[ei] = d_c_loc;

	}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);
}

for (int update_recorder_i = 0; update_recorder_i < 8; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
}
