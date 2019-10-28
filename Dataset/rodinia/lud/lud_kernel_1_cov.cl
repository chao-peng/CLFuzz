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
__kernel void
lud_diagonal(__global float *m,
			 __local  float *shadow,
			 int   matrix_dim,
			 int   offset, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder, __global int* ocl_kernel_loop_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[4];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 4; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int ocl_kernel_barrier_count[3];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 3; ++ocl_kernel_init_i) {
    ocl_kernel_barrier_count[ocl_kernel_init_i] = 0;
}
__local int my_ocl_kernel_loop_recorder[5];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 5; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[5];
bool private_ocl_kernel_loop_boundary_not_reached[5];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	int i,j;
	int tx = get_local_id(0);

	int array_offset = offset*matrix_dim+offset;
	private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
for(i=0; i < BLOCK_SIZE || (private_ocl_kernel_loop_boundary_not_reached[0] = false); i++){
private_ocl_kernel_loop_iter_counter[0]++;

		shadow[i * BLOCK_SIZE + tx]=m[array_offset + tx];
		array_offset += matrix_dim;
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

	OCL_NEW_BARRIER(0,CLK_LOCAL_MEM_FENCE);

	private_ocl_kernel_loop_iter_counter[1] = 0;
private_ocl_kernel_loop_boundary_not_reached[1] = true;
for(i=0; i < BLOCK_SIZE-1 || (private_ocl_kernel_loop_boundary_not_reached[1] = false); i++) {
private_ocl_kernel_loop_iter_counter[1]++;


    if (tx>i){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

      private_ocl_kernel_loop_iter_counter[2] = 0;
private_ocl_kernel_loop_boundary_not_reached[2] = true;
for(j=0; j < i || (private_ocl_kernel_loop_boundary_not_reached[2] = false); j++)
        {

private_ocl_kernel_loop_iter_counter[2]++;
shadow[tx * BLOCK_SIZE + i] -= shadow[tx * BLOCK_SIZE + j] * shadow[j * BLOCK_SIZE + i];
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

		shadow[tx * BLOCK_SIZE + i] /= shadow[i * BLOCK_SIZE + i];
    }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}

	OCL_NEW_BARRIER(1,CLK_LOCAL_MEM_FENCE);
    if (tx>i){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);


      private_ocl_kernel_loop_iter_counter[3] = 0;
private_ocl_kernel_loop_boundary_not_reached[3] = true;
for(j=0; j < i+1 || (private_ocl_kernel_loop_boundary_not_reached[3] = false); j++)
        {

private_ocl_kernel_loop_iter_counter[3]++;
shadow[(i+1) * BLOCK_SIZE + tx] -= shadow[(i+1) * BLOCK_SIZE + j]*shadow[j * BLOCK_SIZE + tx];
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

    }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);
}

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

    array_offset = (offset+1)*matrix_dim+offset;
    private_ocl_kernel_loop_iter_counter[4] = 0;
private_ocl_kernel_loop_boundary_not_reached[4] = true;
for(i=1; i < BLOCK_SIZE || (private_ocl_kernel_loop_boundary_not_reached[4] = false); i++){
private_ocl_kernel_loop_iter_counter[4]++;

      m[array_offset+tx]=shadow[i * BLOCK_SIZE + tx];
      array_offset += matrix_dim;
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

for (int update_recorder_i = 0; update_recorder_i < 4; update_recorder_i++) {
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]);
}
for (int update_recorder_i = 0; update_recorder_i < 5; update_recorder_i++) {
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]);
}
}
