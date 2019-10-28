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
srad_cuda_1(
		  __global float *E_C,
		  __global float *W_C,
		  __global float *N_C,
		  __global float *S_C,
		  __global float * J_cuda,
		  __global float * C_cuda,
		  int cols,
		  int rows,
		  float q0sqr
, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[28];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 28; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int ocl_kernel_barrier_count[4];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 4; ++ocl_kernel_init_i) {
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
  int index_n = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + tx - cols;
  int index_s = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
  int index_w = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty - 1;
  int index_e = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;

  float n, w, e, s, jc, g2, l, num, den, qsqr, c;

  //shared memory allocation
  __local float temp[BLOCK_SIZE][BLOCK_SIZE];
  __local float temp_result[BLOCK_SIZE][BLOCK_SIZE];

  __local float north[BLOCK_SIZE][BLOCK_SIZE];
  __local float south[BLOCK_SIZE][BLOCK_SIZE];
  __local float  east[BLOCK_SIZE][BLOCK_SIZE];
  __local float  west[BLOCK_SIZE][BLOCK_SIZE];

  //load data to shared memory
  north[ty][tx] = J_cuda[index_n];
  south[ty][tx] = J_cuda[index_s];
  if ( by == 0 ){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

  north[ty][tx] = J_cuda[BLOCK_SIZE * bx + tx];
  }
  else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);

if ( by == get_num_groups(1) - 1 ){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);

  south[ty][tx] = J_cuda[cols * BLOCK_SIZE * (get_num_groups(1) - 1) + BLOCK_SIZE * bx + cols * ( BLOCK_SIZE - 1 ) + tx];
  }else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);
}
}

   OCL_NEW_BARRIER(0,CLK_LOCAL_MEM_FENCE);

  west[ty][tx] = J_cuda[index_w];
  east[ty][tx] = J_cuda[index_e];

  if ( bx == 0 ){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[4], 1);

  west[ty][tx] = J_cuda[cols * BLOCK_SIZE * by + cols * ty];
  }
  else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[5], 1);

if ( bx == get_num_groups(0) - 1 ){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[6], 1);

  east[ty][tx] = J_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * ( get_num_groups(0) - 1) + cols * ty + BLOCK_SIZE-1];
  }else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[7], 1);
}
}


  OCL_NEW_BARRIER(1,CLK_LOCAL_MEM_FENCE);



  temp[ty][tx]      = J_cuda[index];

  OCL_NEW_BARRIER(2,CLK_LOCAL_MEM_FENCE);

   jc = temp[ty][tx];

   if ( ty == 0 && tx == 0 ){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[8], 1);
 //nw
	n  = north[ty][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = west[ty][tx]  - jc;
    e  = temp[ty][tx+1] - jc;
   }
   else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[9], 1);

if ( ty == 0 && tx == BLOCK_SIZE-1 ){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[10], 1);
 //ne
	n  = north[ty][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc;
    e  = east[ty][tx] - jc;
   }
   else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[11], 1);

if ( ty == BLOCK_SIZE -1 && tx == BLOCK_SIZE - 1){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[12], 1);
 //se
	n  = temp[ty-1][tx] - jc;
    s  = south[ty][tx] - jc;
    w  = temp[ty][tx-1] - jc;
    e  = east[ty][tx]  - jc;
   }
   else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[13], 1);

if ( ty == BLOCK_SIZE -1 && tx == 0 ){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[14], 1);
//sw
	n  = temp[ty-1][tx] - jc;
    s  = south[ty][tx] - jc;
    w  = west[ty][tx]  - jc;
    e  = temp[ty][tx+1] - jc;
   }

   else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[15], 1);

if ( ty == 0 ){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[16], 1);
 //n
	n  = north[ty][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc;
    e  = temp[ty][tx+1] - jc;
   }
   else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[17], 1);

if ( tx == BLOCK_SIZE -1 ){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[18], 1);
 //e
	n  = temp[ty-1][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc;
    e  = east[ty][tx] - jc;
   }
   else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[19], 1);

if ( ty == BLOCK_SIZE -1){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[20], 1);
 //s
	n  = temp[ty-1][tx] - jc;
    s  = south[ty][tx] - jc;
    w  = temp[ty][tx-1] - jc;
    e  = temp[ty][tx+1] - jc;
   }
   else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[21], 1);

if ( tx == 0 ){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[22], 1);
 //w
	n  = temp[ty-1][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = west[ty][tx] - jc;
    e  = temp[ty][tx+1] - jc;
   }
   else{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[23], 1);
  //the data elements which are not on the borders
	n  = temp[ty-1][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc;
    e  = temp[ty][tx+1] - jc;
   }}
}
}
}
}
}
}



    g2 = ( n * n + s * s + w * w + e * e ) / (jc * jc);

    l = ( n + s + w + e ) / jc;

	num  = (0.5f*g2) - ((1.0f/16.0f)*(l*l)) ;
	den  = 1 + (.25f*l);
	qsqr = num/(den*den);

	// diffusion coefficent (equ 33)
	den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
	c = 1.0f / (1.0f+den) ;

    // saturate diffusion coefficent
	if (c < 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[24], 1);
temp_result[ty][tx] = 0;}
	else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[25], 1);

if (c > 1) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[26], 1);
temp_result[ty][tx] = 1;}
	else {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[27], 1);
temp_result[ty][tx] = c;}}


    OCL_NEW_BARRIER(3,CLK_LOCAL_MEM_FENCE);

    C_cuda[index] = temp_result[ty][tx];
	E_C[index] = e;
	W_C[index] = w;
	S_C[index] = s;
	N_C[index] = n;

for (int update_recorder_i = 0; update_recorder_i < 28; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
}
