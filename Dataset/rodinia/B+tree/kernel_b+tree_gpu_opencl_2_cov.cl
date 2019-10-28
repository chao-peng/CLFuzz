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

//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	DEFINE
//======================================================================================================================================================150

// double precision support (switch between as needed for NVIDIA/AMD)
#ifdef AMDAPP
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

// clBuildProgram compiler cannot link this file for some reason, so had to redefine constants and structures below
// #include ../common.h						// (in directory specified to compiler)			main function header

//======================================================================================================================================================150
//	DEFINE (had to bring from ../common.h here because feature of including headers in clBuildProgram does not work for some reason)
//======================================================================================================================================================150

// change to double if double precision needed
#define fp float

#define DEFAULT_ORDER_2 256

//======================================================================================================================================================150
//	STRUCTURES (had to bring from ../common.h here because feature of including headers in clBuildProgram does not work for some reason)
//======================================================================================================================================================150

// ???
typedef struct knode {
	int location;
	int indices [DEFAULT_ORDER_2 + 1];
	int  keys [DEFAULT_ORDER_2 + 1];
	bool is_leaf;
	int num_keys;
} knode;

//========================================================================================================================================================================================================200
//	findRangeK function
//========================================================================================================================================================================================================200

__kernel void
findRangeK(	long height,
			__global knode *knodesD,
			long knodes_elem,

			__global long *currKnodeD,
			__global long *offsetD,
			__global long *lastKnodeD,
			__global long *offset_2D,
			__global int *startD,
			__global int *endD,
			__global int *RecstartD,
			__global int *ReclenD, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder, __global int* ocl_kernel_loop_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[14];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 14; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int ocl_kernel_barrier_count[3];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 3; ++ocl_kernel_init_i) {
    ocl_kernel_barrier_count[ocl_kernel_init_i] = 0;
}
__local int my_ocl_kernel_loop_recorder[1];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 1; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[1];
bool private_ocl_kernel_loop_boundary_not_reached[1];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);


	// private thread IDs
	int thid = get_local_id(0);
	int bid = get_group_id(0);

	// ???
	int i;
	private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
for(i = 0; i < height || (private_ocl_kernel_loop_boundary_not_reached[0] = false); i++){
private_ocl_kernel_loop_iter_counter[0]++;


		if((knodesD[currKnodeD[bid]].keys[thid] <= startD[bid]) && (knodesD[currKnodeD[bid]].keys[thid+1] > startD[bid])){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

			// this conditional statement is inserted to avoid crush due to but in original code
			// "offset[bid]" calculated below that later addresses part of knodes goes outside of its bounds cause segmentation fault
			// more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
			if(knodesD[currKnodeD[bid]].indices[thid] < knodes_elem){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);

				offsetD[bid] = knodesD[currKnodeD[bid]].indices[thid];
			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);
}
		}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}
		if((knodesD[lastKnodeD[bid]].keys[thid] <= endD[bid]) && (knodesD[lastKnodeD[bid]].keys[thid+1] > endD[bid])){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[4], 1);

			// this conditional statement is inserted to avoid crush due to but in original code
			// "offset_2[bid]" calculated below that later addresses part of knodes goes outside of its bounds cause segmentation fault
			// more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
			if(knodesD[lastKnodeD[bid]].indices[thid] < knodes_elem){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[6], 1);

				offset_2D[bid] = knodesD[lastKnodeD[bid]].indices[thid];
			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[7], 1);
}
		}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[5], 1);
}
		//__syncthreads();
		OCL_NEW_BARRIER(0,CLK_LOCAL_MEM_FENCE);
		// set for next tree level
		if(thid==0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[8], 1);

			currKnodeD[bid] = offsetD[bid];
			lastKnodeD[bid] = offset_2D[bid];
		}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[9], 1);
}
		//	__syncthreads();
		OCL_NEW_BARRIER(1,CLK_LOCAL_MEM_FENCE);
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

	// Find the index of the starting record
	if(knodesD[currKnodeD[bid]].keys[thid] == startD[bid]){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[10], 1);

		RecstartD[bid] = knodesD[currKnodeD[bid]].indices[thid];
	}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[11], 1);
}
	//	__syncthreads();
	OCL_NEW_BARRIER(2,CLK_LOCAL_MEM_FENCE);

	// Find the index of the ending record
	if(knodesD[lastKnodeD[bid]].keys[thid] == endD[bid]){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[12], 1);

		ReclenD[bid] = knodesD[lastKnodeD[bid]].indices[thid] - RecstartD[bid]+1;
	}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[13], 1);
}

for (int update_recorder_i = 0; update_recorder_i < 14; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
for (int update_recorder_i = 0; update_recorder_i < 1; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}

//========================================================================================================================================================================================================200
//	End
//========================================================================================================================================================================================================200
