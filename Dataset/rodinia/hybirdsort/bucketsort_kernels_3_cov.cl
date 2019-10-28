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

#define DIVISIONS               (1 << 10)
#define LOG_DIVISIONS	(10)
#define BUCKET_WARP_LOG_SIZE	(5)
#define BUCKET_WARP_N			(1)
#ifdef BUCKET_WG_SIZE_1
#define BUCKET_THREAD_N BUCKET_WG_SIZE_1
#else
#define BUCKET_THREAD_N			(BUCKET_WARP_N << BUCKET_WARP_LOG_SIZE)
#endif
#define BUCKET_BLOCK_MEMORY		(DIVISIONS * BUCKET_WARP_N)
#define BUCKET_BAND				(128)



__kernel void
bucketsort(global float *input, global int *indice, __global float *output, const int size, global uint *d_prefixoffsets,
		   global uint *l_offsets, __global int* ocl_barrier_divergence_recorder, __global int* ocl_kernel_loop_recorder)
{__local int ocl_kernel_barrier_count[1];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 1; ++ocl_kernel_init_i) {
    ocl_kernel_barrier_count[ocl_kernel_init_i] = 0;
}
__local int my_ocl_kernel_loop_recorder[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[2];
bool private_ocl_kernel_loop_boundary_not_reached[2];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	volatile __local unsigned int s_offset[BUCKET_BLOCK_MEMORY];

	int prefixBase = get_group_id(0) * BUCKET_BLOCK_MEMORY;
    const int warpBase = (get_local_id(0) >> BUCKET_WARP_LOG_SIZE) * DIVISIONS;
    const int numThreads = get_global_size(0);

	private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
for (int i = get_local_id(0); i < BUCKET_BLOCK_MEMORY || (private_ocl_kernel_loop_boundary_not_reached[0] = false); i += get_local_size(0)){
private_ocl_kernel_loop_iter_counter[0]++;

		s_offset[i] = l_offsets[i & (DIVISIONS - 1)] + d_prefixoffsets[prefixBase + i];
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

    OCL_NEW_BARRIER(0,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	private_ocl_kernel_loop_iter_counter[1] = 0;
private_ocl_kernel_loop_boundary_not_reached[1] = true;
for (int tid = get_global_id(0); tid < size || (private_ocl_kernel_loop_boundary_not_reached[1] = false); tid += numThreads) {
private_ocl_kernel_loop_iter_counter[1]++;


		float elem = input[tid];
		int id = indice[tid];
		output[s_offset[warpBase + (id & (DIVISIONS - 1))] + (id >> LOG_DIVISIONS)] = elem;
        int test = s_offset[warpBase + (id & (DIVISIONS - 1))] + (id >> LOG_DIVISIONS);
//        if(test == 2) {
//            printf("EDLLAWD %f", elem);
//        }
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
for (int update_recorder_i = 0; update_recorder_i < 2; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}
