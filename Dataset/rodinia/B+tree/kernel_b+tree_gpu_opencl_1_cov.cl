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

// #ifdef __cplusplus
// extern "C" {
// #endif

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

#define DEFAULT_ORDER 256

//======================================================================================================================================================150
//	STRUCTURES (had to bring from ../common.h here because feature of including headers in clBuildProgram does not work for some reason)
//======================================================================================================================================================150

// Type representing the record to which a given key refers. In a real B+ tree system, the record would hold data (in a database) or a file (in an operating system) or some other information.
// Users can rewrite this part of the code to change the type and content of the value field.
typedef struct record {
	int value;
} record;

// ???
typedef struct knode {
	int location;
	int indices [DEFAULT_ORDER + 1];
	int  keys [DEFAULT_ORDER + 1];
	bool is_leaf;
	int num_keys;
} knode;

//========================================================================================================================================================================================================200
//	findK function
//========================================================================================================================================================================================================200

__kernel void
findK(	long height,
		__global knode *knodesD,
		long knodes_elem,
		__global record *recordsD,

		__global long *currKnodeD,
		__global long *offsetD,
		__global int *keysD,
		__global record *ansD, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder, __global int* ocl_kernel_loop_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[8];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 8; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int ocl_kernel_barrier_count[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
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

	// processtree levels
	int i;
	private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
for(i = 0; i < height || (private_ocl_kernel_loop_boundary_not_reached[0] = false); i++){
private_ocl_kernel_loop_iter_counter[0]++;


		// if value is between the two keys
		if((knodesD[currKnodeD[bid]].keys[thid]) <= keysD[bid] && (knodesD[currKnodeD[bid]].keys[thid+1] > keysD[bid])){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

			// this conditional statement is inserted to avoid crush due to but in original code
			// "offset[bid]" calculated below that addresses knodes[] in the next iteration goes outside of its bounds cause segmentation fault
			// more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
			if(knodesD[offsetD[bid]].indices[thid] < knodes_elem){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);

				offsetD[bid] = knodesD[offsetD[bid]].indices[thid];
			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);
}
		}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}
		//__syncthreads();
		OCL_NEW_BARRIER(0,CLK_LOCAL_MEM_FENCE);
		// set for next tree level
		if(thid==0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[4], 1);

			currKnodeD[bid] = offsetD[bid];
		}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[5], 1);
}
		//__syncthreads();
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

	//At this point, we have a candidate leaf node which may contain
	//the target record.  Check each key to hopefully find the record
	if(knodesD[currKnodeD[bid]].keys[thid] == keysD[bid]){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[6], 1);

		ansD[bid].value = recordsD[knodesD[currKnodeD[bid]].indices[thid]].value;
	}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[7], 1);
}

for (int update_recorder_i = 0; update_recorder_i < 8; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
for (int update_recorder_i = 0; update_recorder_i < 1; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}

//========================================================================================================================================================================================================200
//	End
//========================================================================================================================================================================================================200

// #ifdef __cplusplus
// }
// #endif
