#pragma OPENCL EXTENSION cl_khr_fp64: enable

/*****************************
* OpenCL Kernel Function to replace FindIndex
* param1: arrayX
* param2: arrayY
* param3: CDF
* param4: u
* param5: xj
* param6: yj
* param7: Nparticles
*****************************/
__kernel void particle_kernel(__global double * arrayX, __global double * arrayY, __global double * CDF, __global double * u, __global double * xj, __global double * yj, int Nparticles, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_kernel_loop_recorder){__local int my_ocl_kernel_branch_triggered_recorder[6];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 6; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int my_ocl_kernel_loop_recorder[1];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 1; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[1];
bool private_ocl_kernel_loop_boundary_not_reached[1];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	int i = get_global_id(0);

	if(i < Nparticles){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);


		int index = -1;
		int x;

		private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
for(x = 0; x < Nparticles || (private_ocl_kernel_loop_boundary_not_reached[0] = false); x++){
private_ocl_kernel_loop_iter_counter[0]++;

			if(CDF[x] >= u[i]){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);

				index = x;
				break;
			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);
}
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
		if(index == -1){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[4], 1);

			index = Nparticles-1;
		}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[5], 1);
}

		xj[i] = arrayX[index];
		yj[i] = arrayY[index];

	}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}
for (int update_recorder_i = 0; update_recorder_i < 6; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
for (int update_recorder_i = 0; update_recorder_i < 1; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}
