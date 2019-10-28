#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif



__kernel void
kmeans_swap(__global float  *feature,
			__global float  *feature_swap,
			int     npoints,
			int     nfeatures
, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_kernel_loop_recorder){__local int my_ocl_kernel_branch_triggered_recorder[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int my_ocl_kernel_loop_recorder[1];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 1; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[1];
bool private_ocl_kernel_loop_boundary_not_reached[1];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);


	unsigned int tid = get_global_id(0);
	//for(int i = 0; i <  nfeatures; i++)
	//	feature_swap[i * npoints + tid] = feature[tid * nfeatures + i];
    //Lingjie Zhang modificated at 11/05/2015
    if (tid < npoints){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

	    private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
for(int i = 0; i <  nfeatures || (private_ocl_kernel_loop_boundary_not_reached[0] = false); i++)
		    {

private_ocl_kernel_loop_iter_counter[0]++;
feature_swap[i * npoints + tid] = feature[tid * nfeatures + i];
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

    }else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}

    // end of Lingjie Zhang's modification
for (int update_recorder_i = 0; update_recorder_i < 2; update_recorder_i++) {
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]);
}
for (int update_recorder_i = 0; update_recorder_i < 1; update_recorder_i++) {
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]);
}
}
