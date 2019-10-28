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

/***************************************************************************
 *cr
 *cr            (C) Copyright 2012-2012 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

///////////////////////////////////////////////////////////////////////////////
// R-per-block approach.- Base version: 1 sub-histogram per block
//
// histo:	Final histogram in global memory
// data:	Input image. Pixels are stored in 4-byte unsigned int
// size:	Input image size (number of pixels)
// BINS:	Number of histogram bins
//
// This function was developed at the University of Córdoba and
// contributed by Juan Gómez-Luna.
///////////////////////////////////////////////////////////////////////////////
__kernel void histo_R_per_block_kernel(__global unsigned int* histo,
                                         __global unsigned int* data,
                                         __local unsigned int* Hs,
                                         int size, int BINS, __global int* ocl_barrier_divergence_recorder, __global int* ocl_kernel_loop_recorder)
{__local int ocl_kernel_barrier_count[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    ocl_kernel_barrier_count[ocl_kernel_init_i] = 0;
}
__local int my_ocl_kernel_loop_recorder[3];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 3; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[3];
bool private_ocl_kernel_loop_boundary_not_reached[3];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  // Work-group and work-item index
  const int bx = get_group_id(0);
  const int tx = get_local_id(0);

  // Constants for naive read access
  const int begin = bx * get_local_size(0) + tx;
  const int end = size;
  const int step = get_global_size(0);

  // Sub-histogram initialization
  private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
for(int pos = tx; pos < BINS || (private_ocl_kernel_loop_boundary_not_reached[0] = false); pos += get_local_size(0)) {

private_ocl_kernel_loop_iter_counter[0]++;
Hs[pos] = 0;
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


  OCL_NEW_BARRIER(0,CLK_LOCAL_MEM_FENCE);	// Intra-group synchronization

  // Main loop
  private_ocl_kernel_loop_iter_counter[1] = 0;
private_ocl_kernel_loop_boundary_not_reached[1] = true;
for(int i = begin; i < end || (private_ocl_kernel_loop_boundary_not_reached[1] = false); i += step){
private_ocl_kernel_loop_iter_counter[1]++;

    // Global memory read
    unsigned int d = data[i];

    // Atomic vote in local memory
    atom_inc(&Hs[(d * BINS) >> 12]);
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

  OCL_NEW_BARRIER(1,CLK_LOCAL_MEM_FENCE);	// Intra-group synchronization

  // Merge in global memory
  private_ocl_kernel_loop_iter_counter[2] = 0;
private_ocl_kernel_loop_boundary_not_reached[2] = true;
for(int pos = tx; pos < BINS || (private_ocl_kernel_loop_boundary_not_reached[2] = false); pos += get_local_size(0)){
private_ocl_kernel_loop_iter_counter[2]++;

    unsigned int sum = 0;
    sum = Hs[pos];
    // Atomic addition in global memory
    atom_add(histo + pos, sum);
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
for (int update_recorder_i = 0; update_recorder_i < 3; update_recorder_i++) {
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]);
}
}
