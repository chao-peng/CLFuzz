#define OCL_NEW_BARRIER(barrierid,arg)\
{\
  atom_inc(&ocl_kernel_barrier_count[barrierid]);\
  barrier(arg);\
  if (ocl_kernel_barrier_count[barrierid]!=ocl_get_general_size()) {\
    ocl_barrier_divergence_recorder[barrierid]=1;\
  }\
  barrier(arg);\
  ocl_kernel_barrier_count[barrierid]=0;\
  barrier(arg);\
}
int ocl_get_general_size(){
  int result = 1;\
  for (int i=0; i<get_work_dim(); i++){
    result*=get_local_size(i);
  }
  return result;
}

#define BLOCK_SIZE 16

int
maximum( int a,
		int b,
		int c, __local int* my_ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder, __local int* ocl_kernel_barrier_count, __local int* my_ocl_kernel_loop_recorder){

	int k;
	if( a <= b )
		{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);
k = b;
}
	else
		{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
k = a;
}

	if( k <=c )
		{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);
return(c);
}
	else
		{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);
return(k);
}

}

	__kernel void
needle_opencl_shared_1(  __global int* referrence,
		__global int* matrix_opencl,
		int cols,
		int penalty,
		int i,
		int block_width, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder, __global int* ocl_kernel_loop_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[8];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 8; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int ocl_kernel_barrier_count[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    ocl_kernel_barrier_count[ocl_kernel_init_i] = 0;
}
__local int my_ocl_kernel_loop_recorder[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[2];
bool private_ocl_kernel_loop_boundary_not_reached[2];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	int bx = get_group_id(0);
	int tx = get_local_id(0);

	int b_index_x = bx;
	int b_index_y = i - 1 - bx;
	int index   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( cols + 1 );

	//This pattern at EACH loop step employes a variable number of threads to work in a wavefront pattern. I.e., first iteration only [1][1] is calculated
	//second iteration [2][1] and [1][2], third iteration [3][1] and [2][2] and [1][3], etc. Before each loop iteration starts the barrier ensures that
	//all values needed in the next step have been calculated and stored in local memory in temp.
	//This whole loop calculates up to the main anti-diagonal, i.e., [1][4] along [4][1].
	private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
for( int m = 0 ; m < BLOCK_SIZE || (private_ocl_kernel_loop_boundary_not_reached[0] = false) ; m++){
private_ocl_kernel_loop_iter_counter[0]++;


		if ( tx <= m ){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[4], 1);


			int ref_x=index+(m-tx)*cols;

			matrix_opencl[ref_x] = maximum( matrix_opencl[ref_x-(cols+1)] + referrence[ref_x],
					matrix_opencl[ref_x-1]  - penalty,
					matrix_opencl[ref_x-cols]  - penalty, my_ocl_kernel_branch_triggered_recorder, ocl_barrier_divergence_recorder, ocl_kernel_barrier_count, my_ocl_kernel_loop_recorder);

		}else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[5], 1);
}


		OCL_NEW_BARRIER(0,CLK_GLOBAL_MEM_FENCE);

	}
if (private_ocl_kernel_loop_iter_counter[0] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[0], 1);
}if (private_ocl_kernel_loop_iter_counter[0] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[0], 2);
}if (private_ocl_kernel_loop_iter_counter[0] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[0], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[0]) {
    atomic_or(&my_ocl_kernel_loop_recorder[0], 8);
}

	//Same as above, but for the lower right part of the matrix.
	private_ocl_kernel_loop_iter_counter[1] = 0;
private_ocl_kernel_loop_boundary_not_reached[1] = true;
for( int m = BLOCK_SIZE - 2 ; m >=0 || (private_ocl_kernel_loop_boundary_not_reached[1] = false) ; m--){
private_ocl_kernel_loop_iter_counter[1]++;


		if ( tx <= m){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[6], 1);


			int ref_x=index+(m-tx)*cols+(cols+1)*(BLOCK_SIZE-1-m);

			matrix_opencl[ref_x] = maximum( matrix_opencl[ref_x-(cols+1)] + referrence[ref_x],
					matrix_opencl[ref_x-1]  - penalty,
					matrix_opencl[ref_x-cols]  - penalty, my_ocl_kernel_branch_triggered_recorder, ocl_barrier_divergence_recorder, ocl_kernel_barrier_count, my_ocl_kernel_loop_recorder);

		}else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[7], 1);
}


		OCL_NEW_BARRIER(1,CLK_GLOBAL_MEM_FENCE);
	}
if (private_ocl_kernel_loop_iter_counter[1] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[1], 1);
}if (private_ocl_kernel_loop_iter_counter[1] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[1], 2);
}if (private_ocl_kernel_loop_iter_counter[1] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[1], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[1]) {
    atomic_or(&my_ocl_kernel_loop_recorder[1], 8);
}

for (int update_recorder_i = 0; update_recorder_i < 8; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
for (int update_recorder_i = 0; update_recorder_i < 2; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}
