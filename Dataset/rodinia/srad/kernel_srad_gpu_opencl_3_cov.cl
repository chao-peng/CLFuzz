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
//	Reduce KERNEL
//========================================================================================================================================================================================================200

__kernel void
reduce_kernel(	long d_Ne,													// number of elements in array
				long d_no,													// number of sums to reduce
				int d_mul,													// increment
				__global fp* d_sums,										// pointer to partial sums variable (DEVICE GLOBAL MEMORY)
				__global fp* d_sums2,
				int gridDim, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder, __global int* ocl_kernel_loop_recorder){__local int my_ocl_kernel_branch_triggered_recorder[20];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 20; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int ocl_kernel_barrier_count[4];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 4; ++ocl_kernel_init_i) {
    ocl_kernel_barrier_count[ocl_kernel_init_i] = 0;
}
__local int my_ocl_kernel_loop_recorder[5];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 5; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[5];
bool private_ocl_kernel_loop_boundary_not_reached[5];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);


	// indexes
    int bx = get_group_id(0);												// get current horizontal block index
	int tx = get_local_id(0);												// get current horizontal thread index
	int ei = (bx*NUMBER_THREADS)+tx;										// unique thread id, more threads than actual elements !!!
	// int gridDim = (int)get_group_size(0)/(int)get_local_size(0);			// number of workgroups
	int nf = NUMBER_THREADS-(gridDim*NUMBER_THREADS-d_no);				// number of elements assigned to last block
	int df = 0;															// divisibility factor for the last block

	// statistical
	__local fp d_psum[NUMBER_THREADS];										// data for block calculations allocated by every block in its shared memory
	__local fp d_psum2[NUMBER_THREADS];

	// counters
	int i;

	// copy data to shared memory
	if(ei<d_no){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);
															// do only for the number of elements, omit extra threads

		d_psum[tx] = d_sums[ei*d_mul];
		d_psum2[tx] = d_sums2[ei*d_mul];

	}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}

    // Lingjie Zhang modificated at Nov 1, 2015
	//	barrier(CLK_LOCAL_MEM_FENCE);
	OCL_NEW_BARRIER(0,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); // Lukasz proposed, Ke modified 2015/12/12 22:31:00
    // end Lingjie Zhang modification

	// reduction of sums if all blocks are full (rare case)
	if(nf == NUMBER_THREADS){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);

		// sum of every 2, 4, ..., NUMBER_THREADS elements
		private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
for(i=2; i<=NUMBER_THREADS || (private_ocl_kernel_loop_boundary_not_reached[0] = false); i=2*i){
private_ocl_kernel_loop_iter_counter[0]++;

			// sum of elements
			if((tx+1) % i == 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[4], 1);
											// every ith
				d_psum[tx] = d_psum[tx] + d_psum[tx-i/2];
				d_psum2[tx] = d_psum2[tx] + d_psum2[tx-i/2];
			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[5], 1);
}
			// synchronization
			OCL_NEW_BARRIER(1,CLK_LOCAL_MEM_FENCE);
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
		// final sumation by last thread in every block
		if(tx==(NUMBER_THREADS-1)){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[6], 1);
											// block result stored in global memory
			d_sums[bx*d_mul*NUMBER_THREADS] = d_psum[tx];
			d_sums2[bx*d_mul*NUMBER_THREADS] = d_psum2[tx];
		}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[7], 1);
}
	}
	// reduction of sums if last block is not full (common case)
	else{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);

		// for full blocks (all except for last block)
		if(bx != (gridDim - 1)){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[8], 1);
											//
			// sum of every 2, 4, ..., NUMBER_THREADS elements
			private_ocl_kernel_loop_iter_counter[1] = 0;
private_ocl_kernel_loop_boundary_not_reached[1] = true;
for(i=2; i<=NUMBER_THREADS || (private_ocl_kernel_loop_boundary_not_reached[1] = false); i=2*i){
private_ocl_kernel_loop_iter_counter[1]++;
								//
				// sum of elements
				if((tx+1) % i == 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[10], 1);
										// every ith
					d_psum[tx] = d_psum[tx] + d_psum[tx-i/2];
					d_psum2[tx] = d_psum2[tx] + d_psum2[tx-i/2];
				}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[11], 1);
}
				// synchronization
				OCL_NEW_BARRIER(2,CLK_LOCAL_MEM_FENCE);
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
			// final sumation by last thread in every block
			if(tx==(NUMBER_THREADS-1)){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[12], 1);
										// block result stored in global memory
				d_sums[bx*d_mul*NUMBER_THREADS] = d_psum[tx];
				d_sums2[bx*d_mul*NUMBER_THREADS] = d_psum2[tx];
			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[13], 1);
}
		}
		// for not full block (last block)
		else{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[9], 1);
																//
			// figure out divisibility
			private_ocl_kernel_loop_iter_counter[2] = 0;
private_ocl_kernel_loop_boundary_not_reached[2] = true;
for(i=2; i<=NUMBER_THREADS || (private_ocl_kernel_loop_boundary_not_reached[2] = false); i=2*i){
private_ocl_kernel_loop_iter_counter[2]++;
								//
				if(nf >= i){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[14], 1);

					df = i;
				}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[15], 1);
}
			}
if (private_ocl_kernel_loop_iter_counter[2] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[2], 1);
}if (private_ocl_kernel_loop_iter_counter[2] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[2], 2);
}if (private_ocl_kernel_loop_iter_counter[2] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[2], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[2]) {
    atomic_or(&my_ocl_kernel_loop_recorder[2], 8);
}
			// sum of every 2, 4, ..., NUMBER_THREADS elements
			private_ocl_kernel_loop_iter_counter[3] = 0;
private_ocl_kernel_loop_boundary_not_reached[3] = true;
for(i=2; i<=df || (private_ocl_kernel_loop_boundary_not_reached[3] = false); i=2*i){
private_ocl_kernel_loop_iter_counter[3]++;
											//
				// sum of elements (only busy threads)
				if((tx+1) % i == 0 && tx<df){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[16], 1);
								// every ith
					d_psum[tx] = d_psum[tx] + d_psum[tx-i/2];
					d_psum2[tx] = d_psum2[tx] + d_psum2[tx-i/2];
				}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[17], 1);
}
				// synchronization (all threads)
				OCL_NEW_BARRIER(3,CLK_LOCAL_MEM_FENCE);
			}
if (private_ocl_kernel_loop_iter_counter[3] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[3], 1);
}if (private_ocl_kernel_loop_iter_counter[3] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[3], 2);
}if (private_ocl_kernel_loop_iter_counter[3] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[3], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[3]) {
    atomic_or(&my_ocl_kernel_loop_recorder[3], 8);
}
			// remainder / final summation by last thread
			if(tx==(df-1)){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[18], 1);
										//
				// compute the remainder and final summation by last busy thread
				private_ocl_kernel_loop_iter_counter[4] = 0;
private_ocl_kernel_loop_boundary_not_reached[4] = true;
for(i=(bx*NUMBER_THREADS)+df; i<(bx*NUMBER_THREADS)+nf || (private_ocl_kernel_loop_boundary_not_reached[4] = false); i++){
private_ocl_kernel_loop_iter_counter[4]++;
						//
					d_psum[tx] = d_psum[tx] + d_sums[i];
					d_psum2[tx] = d_psum2[tx] + d_sums2[i];
				}
if (private_ocl_kernel_loop_iter_counter[4] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[4], 1);
}if (private_ocl_kernel_loop_iter_counter[4] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[4], 2);
}if (private_ocl_kernel_loop_iter_counter[4] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[4], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[4]) {
    atomic_or(&my_ocl_kernel_loop_recorder[4], 8);
}
				// final sumation by last thread in every block
				d_sums[bx*d_mul*NUMBER_THREADS] = d_psum[tx];
				d_sums2[bx*d_mul*NUMBER_THREADS] = d_psum2[tx];
			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[19], 1);
}
		}
	}

for (int update_recorder_i = 0; update_recorder_i < 20; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
for (int update_recorder_i = 0; update_recorder_i < 5; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}

//========================================================================================================================================================================================================200
//	SRAD KERNEL
//========================================================================================================================================================================================================200
