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
lud_perimeter(__global float *m,
			  __local  float *dia,
			  __local  float *peri_row,
			  __local  float *peri_col,
			  int matrix_dim,
			  int offset, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder, __global int* ocl_kernel_loop_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[6];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 6; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int ocl_kernel_barrier_count[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    ocl_kernel_barrier_count[ocl_kernel_init_i] = 0;
}
__local int my_ocl_kernel_loop_recorder[10];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 10; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[10];
bool private_ocl_kernel_loop_boundary_not_reached[10];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    int i,j, array_offset;
    int idx;

    int  bx = get_group_id(0);
    int  tx = get_local_id(0);

    if (tx < BLOCK_SIZE) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

      idx = tx;
      array_offset = offset*matrix_dim+offset;
      private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
for (i=0; i < BLOCK_SIZE/2 || (private_ocl_kernel_loop_boundary_not_reached[0] = false); i++){
private_ocl_kernel_loop_iter_counter[0]++;

      dia[i * BLOCK_SIZE + idx]=m[array_offset+idx];
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

    array_offset = offset*matrix_dim+offset;
    private_ocl_kernel_loop_iter_counter[1] = 0;
private_ocl_kernel_loop_boundary_not_reached[1] = true;
for (i=0; i < BLOCK_SIZE || (private_ocl_kernel_loop_boundary_not_reached[1] = false); i++) {
private_ocl_kernel_loop_iter_counter[1]++;

      peri_row[i * BLOCK_SIZE+ idx]=m[array_offset+(bx+1)*BLOCK_SIZE+idx];
      array_offset += matrix_dim;
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

    } else {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);

    idx = tx-BLOCK_SIZE;

    array_offset = (offset+BLOCK_SIZE/2)*matrix_dim+offset;
    private_ocl_kernel_loop_iter_counter[2] = 0;
private_ocl_kernel_loop_boundary_not_reached[2] = true;
for (i=BLOCK_SIZE/2; i < BLOCK_SIZE || (private_ocl_kernel_loop_boundary_not_reached[2] = false); i++){
private_ocl_kernel_loop_iter_counter[2]++;

      dia[i * BLOCK_SIZE + idx]=m[array_offset+idx];
      array_offset += matrix_dim;
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

    array_offset = (offset+(bx+1)*BLOCK_SIZE)*matrix_dim+offset;
    private_ocl_kernel_loop_iter_counter[3] = 0;
private_ocl_kernel_loop_boundary_not_reached[3] = true;
for (i=0; i < BLOCK_SIZE || (private_ocl_kernel_loop_boundary_not_reached[3] = false); i++) {
private_ocl_kernel_loop_iter_counter[3]++;

      peri_col[i * BLOCK_SIZE + idx] = m[array_offset+idx];
      array_offset += matrix_dim;
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
    OCL_NEW_BARRIER(0,CLK_LOCAL_MEM_FENCE);

    if (tx < BLOCK_SIZE) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);
 //peri-row
     idx=tx;
      private_ocl_kernel_loop_iter_counter[4] = 0;
private_ocl_kernel_loop_boundary_not_reached[4] = true;
for(i=1; i < BLOCK_SIZE || (private_ocl_kernel_loop_boundary_not_reached[4] = false); i++){
private_ocl_kernel_loop_iter_counter[4]++;

      private_ocl_kernel_loop_iter_counter[5] = 0;
private_ocl_kernel_loop_boundary_not_reached[5] = true;
for (j=0; j < i || (private_ocl_kernel_loop_boundary_not_reached[5] = false); j++)
        {

private_ocl_kernel_loop_iter_counter[5]++;
peri_row[i * BLOCK_SIZE + idx]-=dia[i * BLOCK_SIZE+ j]*peri_row[j * BLOCK_SIZE + idx];
}
if (private_ocl_kernel_loop_iter_counter[5] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[5], 1);
}if (private_ocl_kernel_loop_iter_counter[5] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[5], 2);
}if (private_ocl_kernel_loop_iter_counter[5] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[5], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[5]) {
    atomic_or(&my_ocl_kernel_loop_recorder[5], 8);
}

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
    } else {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);
 //peri-col
     idx=tx - BLOCK_SIZE;
     private_ocl_kernel_loop_iter_counter[6] = 0;
private_ocl_kernel_loop_boundary_not_reached[6] = true;
for(i=0; i < BLOCK_SIZE || (private_ocl_kernel_loop_boundary_not_reached[6] = false); i++){
private_ocl_kernel_loop_iter_counter[6]++;

      private_ocl_kernel_loop_iter_counter[7] = 0;
private_ocl_kernel_loop_boundary_not_reached[7] = true;
for(j=0; j < i || (private_ocl_kernel_loop_boundary_not_reached[7] = false); j++)
        {

private_ocl_kernel_loop_iter_counter[7]++;
peri_col[idx * BLOCK_SIZE + i]-=peri_col[idx * BLOCK_SIZE+ j]*dia[j * BLOCK_SIZE + i];
}
if (private_ocl_kernel_loop_iter_counter[7] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[7], 1);
}if (private_ocl_kernel_loop_iter_counter[7] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[7], 2);
}if (private_ocl_kernel_loop_iter_counter[7] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[7], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[7]) {
    atomic_or(&my_ocl_kernel_loop_recorder[7], 8);
}

      peri_col[idx * BLOCK_SIZE + i] /= dia[i * BLOCK_SIZE+ i];
     }
if (private_ocl_kernel_loop_iter_counter[6] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[6], 1);
}if (private_ocl_kernel_loop_iter_counter[6] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[6], 2);
}if (private_ocl_kernel_loop_iter_counter[6] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[6], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[6]) {
    atomic_or(&my_ocl_kernel_loop_recorder[6], 8);
}
   }

	OCL_NEW_BARRIER(1,CLK_LOCAL_MEM_FENCE);

  if (tx < BLOCK_SIZE) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[4], 1);
 //peri-row
    idx=tx;
    array_offset = (offset+1)*matrix_dim+offset;
    private_ocl_kernel_loop_iter_counter[8] = 0;
private_ocl_kernel_loop_boundary_not_reached[8] = true;
for(i=1; i < BLOCK_SIZE || (private_ocl_kernel_loop_boundary_not_reached[8] = false); i++){
private_ocl_kernel_loop_iter_counter[8]++;

      m[array_offset+(bx+1)*BLOCK_SIZE+idx] = peri_row[i*BLOCK_SIZE+idx];
      array_offset += matrix_dim;
    }
if (private_ocl_kernel_loop_iter_counter[8] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[8], 1);
}if (private_ocl_kernel_loop_iter_counter[8] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[8], 2);
}if (private_ocl_kernel_loop_iter_counter[8] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[8], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[8]) {
    atomic_or(&my_ocl_kernel_loop_recorder[8], 8);
}
  } else {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[5], 1);
 //peri-col
    idx=tx - BLOCK_SIZE;
    array_offset = (offset+(bx+1)*BLOCK_SIZE)*matrix_dim+offset;
    private_ocl_kernel_loop_iter_counter[9] = 0;
private_ocl_kernel_loop_boundary_not_reached[9] = true;
for(i=0; i < BLOCK_SIZE || (private_ocl_kernel_loop_boundary_not_reached[9] = false); i++){
private_ocl_kernel_loop_iter_counter[9]++;

      m[array_offset+idx] =  peri_col[i*BLOCK_SIZE+idx];
      array_offset += matrix_dim;
    }
if (private_ocl_kernel_loop_iter_counter[9] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[9], 1);
}if (private_ocl_kernel_loop_iter_counter[9] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[9], 2);
}if (private_ocl_kernel_loop_iter_counter[9] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[9], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[9]) {
    atomic_or(&my_ocl_kernel_loop_recorder[9], 8);
}
  }

for (int update_recorder_i = 0; update_recorder_i < 6; update_recorder_i++) {
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]);
}
for (int update_recorder_i = 0; update_recorder_i < 10; update_recorder_i++) {
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]);
}
}
