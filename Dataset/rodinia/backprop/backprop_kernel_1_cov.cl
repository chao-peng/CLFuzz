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

__kernel void
bpnn_layerforward_ocl(__global float *input_cuda,
	                  __global float *output_hidden_cuda,
					  __global float *input_hidden_cuda,
					  __global float *hidden_partial_sum,
					  __local float *input_node,
					  __local float *weight_matrix,
					  int in,
					  int hid, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder, __global int* ocl_kernel_loop_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[6];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 6; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int ocl_kernel_barrier_count[5];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 5; ++ocl_kernel_init_i) {
    ocl_kernel_barrier_count[ocl_kernel_init_i] = 0;
}
__local int my_ocl_kernel_loop_recorder[1];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 1; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[1];
bool private_ocl_kernel_loop_boundary_not_reached[1];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);


   int by = get_group_id(1);
   int tx = get_local_id(0);
   int ty = get_local_id(1);

   int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;

   int index_in = HEIGHT * by + ty + 1;

	if ( tx == 0 )
		{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);
input_node[ty] = input_cuda[index_in];
} else { 
atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}
;
		OCL_NEW_BARRIER(0,CLK_LOCAL_MEM_FENCE);

		weight_matrix[ty * WIDTH + tx] =  input_hidden_cuda[index];
		OCL_NEW_BARRIER(1,CLK_LOCAL_MEM_FENCE);

		weight_matrix[ty * WIDTH + tx]= weight_matrix[ty * WIDTH + tx] * input_node[ty];
		OCL_NEW_BARRIER(2,CLK_LOCAL_MEM_FENCE);

		private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
for ( int i = 1 ; i <= HEIGHT || (private_ocl_kernel_loop_boundary_not_reached[0] = false) ; i=i*2){
private_ocl_kernel_loop_iter_counter[0]++;

	//for ( int i = 1 ; i <= 4 ; i++){
      int power_two = i;
		//int power_two = 2 << (i - 1);

	    if( ty % power_two == 0 )
		  {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);
weight_matrix[ty * WIDTH + tx]= weight_matrix[ty * WIDTH + tx] + weight_matrix[(ty + power_two/2)* WIDTH + tx];
} else { 
atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);
}


		OCL_NEW_BARRIER(3,CLK_LOCAL_MEM_FENCE);

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

    input_hidden_cuda[index] =  weight_matrix[ty * WIDTH + tx];

	OCL_NEW_BARRIER(4,CLK_LOCAL_MEM_FENCE);

    if ( tx == 0 ) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[4], 1);

	  hidden_partial_sum[by * hid + ty] = weight_matrix[tx* WIDTH + ty];
    }else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[5], 1);
}


for (int update_recorder_i = 0; update_recorder_i < 6; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
for (int update_recorder_i = 0; update_recorder_i < 1; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}


#endif
