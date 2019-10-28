#define BLOCK_SIZE 16


__kernel void
srad_cuda_2(
		  __global float *E_C,
		  __global float *W_C,
		  __global float *N_C,
		  __global float *S_C,
		  __global float * J_cuda,
		  __global float * C_cuda,
		  int cols,
		  int rows,
		  float lambda,
		  float q0sqr
)
{
	//block id
	int bx = get_group_id(0);
    int by = get_group_id(1);

	//thread id
    int tx = get_local_id(0);
    int ty = get_local_id(1);

	//indices
    int index   = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
	int index_s = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
    int index_e = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;
	float cc, cn, cs, ce, cw, d_sum;

	//shared memory allocation
	__local float south_c[BLOCK_SIZE][BLOCK_SIZE];
    __local float  east_c[BLOCK_SIZE][BLOCK_SIZE];

    __local float c_cuda_temp[BLOCK_SIZE][BLOCK_SIZE];
    __local float c_cuda_result[BLOCK_SIZE][BLOCK_SIZE];
    __local float temp[BLOCK_SIZE][BLOCK_SIZE];

    //load data to shared memory
	temp[ty][tx]      = J_cuda[index];

    barrier(CLK_LOCAL_MEM_FENCE);


	if ( by == get_num_groups(1) - 1 ){
	south_c[ty][tx] = C_cuda[cols * BLOCK_SIZE * (get_num_groups(1) - 1) + BLOCK_SIZE * bx + cols * ( BLOCK_SIZE - 1 ) + tx];
	}
    else
	south_c[ty][tx] = C_cuda[index_s];

	barrier(CLK_LOCAL_MEM_FENCE);



	if ( bx == get_num_groups(0) - 1 ){
	east_c[ty][tx] = C_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * ( get_num_groups(0) - 1) + cols * ty + BLOCK_SIZE-1];
	}
    else
	east_c[ty][tx] = C_cuda[index_e];

    barrier(CLK_LOCAL_MEM_FENCE);

    c_cuda_temp[ty][tx]      = C_cuda[index];

    barrier(CLK_LOCAL_MEM_FENCE);

	cc = c_cuda_temp[ty][tx];

   if ( ty == BLOCK_SIZE -1 && tx == BLOCK_SIZE - 1){ //se
	cn  = cc;
    cs  = south_c[ty][tx];
    cw  = cc;
    ce  = east_c[ty][tx];
   }
   else if ( tx == BLOCK_SIZE -1 ){ //e
	cn  = cc;
    cs  = c_cuda_temp[ty+1][tx];
    cw  = cc;
    ce  = east_c[ty][tx];
   }
   else if ( ty == BLOCK_SIZE -1){ //s
	cn  = cc;
    cs  = south_c[ty][tx];
    cw  = cc;
    ce  = c_cuda_temp[ty][tx+1];
   }
   else{ //the data elements which are not on the borders
	cn  = cc;
    cs  = c_cuda_temp[ty+1][tx];
    cw  = cc;
    ce  = c_cuda_temp[ty][tx+1];
   }

   // divergence (equ 58)
   d_sum = cn * N_C[index] + cs * S_C[index] + cw * W_C[index] + ce * E_C[index];

   // image update (equ 61)
   c_cuda_result[ty][tx] = temp[ty][tx] + 0.25f * lambda * d_sum;

   barrier(CLK_LOCAL_MEM_FENCE);

   J_cuda[index] = c_cuda_result[ty][tx];

}
