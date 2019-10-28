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

#define THREADS 256
#define WIDTH 16
#define HEIGHT 16
#define ETA 0.3f
#define MOMENTUM 0.3f

#ifndef _BACKPROP_CUDA_KERNEL_H_
#define _BACKPROP_CUDA_KERNEL_H_
#define WM(i, j)   weight_matrix[(j) + (i) * WIDTH]




__kernel void  bpnn_adjust_weights_ocl( __global float * delta,
										 int hid,
										__global float * ly,
										 int in,
										__global float * w,
										__global float * oldw, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int ocl_kernel_barrier_count[1];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 1; ++ocl_kernel_init_i) {
    ocl_kernel_barrier_count[ocl_kernel_init_i] = 0;
}
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);


   int by = get_group_id(1);
   int tx = get_local_id(0);
   int ty = get_local_id(1);

   int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;
   int index_y = HEIGHT * by + ty + 1;
   int index_x = tx + 1;

   w[index] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
   oldw[index] = ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));

   OCL_NEW_BARRIER(0,CLK_LOCAL_MEM_FENCE);

   if (ty == 0 && by ==0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

	w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
	oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
   }else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}


for (int update_recorder_i = 0; update_recorder_i < 2; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
}
#endif
