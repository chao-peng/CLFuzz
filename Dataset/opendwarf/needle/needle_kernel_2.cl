#define BLOCK_SIZE 16

int
maximum( int a,
		int b,
		int c){

	int k;
	if( a <= b )
		k = b;
	else
		k = a;

	if( k <=c )
		return(c);
	else
		return(k);

}



	__kernel void
needle_opencl_shared_2(  __global int* referrence,
		__global int* matrix_opencl,
		int cols,
		int penalty,
		int i,
		int block_width)
{

	int bx = get_group_id(0);
	int tx = get_local_id(0);

	int b_index_x = bx + block_width - i  ;
	int b_index_y = block_width - bx -1;
	int index   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( cols + 1 );

	for( int m = 0 ; m < BLOCK_SIZE ; m++){

		if ( tx <= m ){

			int ref_x=index+(m-tx)*cols;

			matrix_opencl[ref_x] = maximum( matrix_opencl[ref_x-(cols+1)] + referrence[ref_x],
					matrix_opencl[ref_x-1]  - penalty,
					matrix_opencl[ref_x-cols]  - penalty);

		}

		barrier(CLK_GLOBAL_MEM_FENCE);

	}


	for( int m = BLOCK_SIZE - 2 ; m >=0 ; m--){

		if ( tx <= m){

			int ref_x=index+(m-tx)*cols+(cols+1)*(BLOCK_SIZE-1-m);

			matrix_opencl[ref_x] = maximum( matrix_opencl[ref_x-(cols+1)] + referrence[ref_x],
					matrix_opencl[ref_x-1]  - penalty,
					matrix_opencl[ref_x-cols]  - penalty);

		}

		barrier(CLK_GLOBAL_MEM_FENCE);
	}

}
