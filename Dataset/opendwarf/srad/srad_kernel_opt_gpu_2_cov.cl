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
, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[10];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 10; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int ocl_kernel_barrier_count[5];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 5; ++ocl_kernel_init_i) {
    ocl_kernel_barrier_count[ocl_kernel_init_i] = 0;
}
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

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

    OCL_NEW_BARRIER(0,CLK_LOCAL_MEM_FENCE);


	if ( by == get_num_groups(1) - 1 ){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

	south_c[ty][tx] = C_cuda[cols * BLOCK_SIZE * (get_num_groups(1) - 1) + BLOCK_SIZE * bx + cols * ( BLOCK_SIZE - 1 ) + tx];
	}
    else
	{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
south_c[ty][tx] = C_cuda[index_s];
}

	OCL_NEW_BARRIER(1,CLK_LOCAL_MEM_FENCE);



	if ( bx == get_num_groups(0) - 1 ){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);

	east_c[ty][tx] = C_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * ( get_num_groups(0) - 1) + cols * ty + BLOCK_SIZE-1];
	}
    else
	{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);
east_c[ty][tx] = C_cuda[index_e];
}

    OCL_NEW_BARRIER(2,CLK_LOCAL_MEM_FENCE);

    c_cuda_temp[ty][tx]      = C_cuda[index];

    OCL_NEW_BARRIER(3,CLK_LOCAL_MEM_FENCE);

	cc = c_cuda_temp[ty][tx];

   if ( ty == BLOCK_SIZE -1 && tx == BLOCK_SIZE - 1){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[4], 1);
 //se
	cn  = cc;
    cs  = south_c[ty][tx];
    cw  = cc;
    ce  = east_c[ty][tx];
   }
   else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[5], 1);

if ( tx == BLOCK_SIZE -1 ){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[6], 1);
 //e
	cn  = cc;
    cs  = c_cuda_temp[ty+1][tx];
    cw  = cc;
    ce  = east_c[ty][tx];
   }
   else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[7], 1);

if ( ty == BLOCK_SIZE -1){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[8], 1);
 //s
	cn  = cc;
    cs  = south_c[ty][tx];
    cw  = cc;
    ce  = c_cuda_temp[ty][tx+1];
   }
   else{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[9], 1);
 //the data elements which are not on the borders
	cn  = cc;
    cs  = c_cuda_temp[ty+1][tx];
    cw  = cc;
    ce  = c_cuda_temp[ty][tx+1];
   }}
}


   // divergence (equ 58)
   d_sum = cn * N_C[index] + cs * S_C[index] + cw * W_C[index] + ce * E_C[index];

   // image update (equ 61)
   c_cuda_result[ty][tx] = temp[ty][tx] + 0.25f * lambda * d_sum;

   OCL_NEW_BARRIER(4,CLK_LOCAL_MEM_FENCE);

   J_cuda[index] = c_cuda_result[ty][tx];

for (int update_recorder_i = 0; update_recorder_i < 10; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
}
