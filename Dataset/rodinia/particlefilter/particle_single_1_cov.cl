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

// #if defined(cl_amd_fp64) || defined(cl_khr_fp64)

// #if defined(cl_amd_fp64)
// #pragma OPENCL EXTENSION cl_amd_fp64 : enable
// #elif defined(cl_khr_fp64)
// #pragma OPENCL EXTENSION cl_khr_fp64 : enable
// #endif

#define SCALE_FACTOR 300

/** added this function. was missing in original float version.
 * Takes in a float and returns an integer that approximates to that float
 * @return if the mantissa < .5 => return value < input value; else return value > input value
 */

__kernel void find_index_kernel(__global float * arrayX, __global float * arrayY,
	__global float * CDF, __global float * u, __global float * xj,
	__global float * yj, __global float * weights, int Nparticles
	, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder, __global int* ocl_kernel_loop_recorder){__local int my_ocl_kernel_branch_triggered_recorder[6];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 6; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int ocl_kernel_barrier_count[1];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 1; ++ocl_kernel_init_i) {
    ocl_kernel_barrier_count[ocl_kernel_init_i] = 0;
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

			//weights[i] = 1 / ((float) (Nparticles)); //moved this code to the beginning of likelihood kernel

		}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}
		OCL_NEW_BARRIER(0,CLK_GLOBAL_MEM_FENCE);
for (int update_recorder_i = 0; update_recorder_i < 6; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
for (int update_recorder_i = 0; update_recorder_i < 1; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}



/*****************************
* OpenCL Likelihood Kernel Function to replace FindIndex
* param1: arrayX
* param2: arrayY
* param2.5: CDF
* param3: ind
* param4: objxy
* param5: likelihood
* param6: I
* param6.5: u
* param6.75: weights
* param7: Nparticles
* param8: countOnes
* param9: max_size
* param10: k
* param11: IszY
* param12: Nfr
*****************************/


//#endif
