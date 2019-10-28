// N-queen solver for OpenCL
// Ping-Che Chen


// define this to use predicated version in the rotation check part
#define PREDICATED


#ifdef USE_ATOMICS
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#endif

#ifdef ENABLE_CHAR
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#endif


inline int bit_scan(unsigned int x, __local int* my_ocl_kernel_branch_triggered_recorder, __local int* my_ocl_kernel_loop_recorder)
{
	int res = 0;
	res |= (x & 0xaaaaaaaa) ? 1 : 0;
	res |= (x & 0xcccccccc) ? 2 : 0;
	res |= (x & 0xf0f0f0f0) ? 4 : 0;
	res |= (x & 0xff00ff00) ? 8 : 0;
	res |= (x & 0xffff0000) ? 16 : 0;
	return res;
}




#define BOARD(x) ((x) < board_size - level ? params[idx + pitch * (4 + (x))] : ns[(x) - board_size + level])


__kernel void nqueen1(int board_size, int level, int threads, int pitch, __global uint* params, __global uint* results, __constant uint* forbidden, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_kernel_loop_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[6];
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

	int idx = get_global_id(0);
	int tid = get_local_id(0);
	uint mask = params[idx];
	uint left_mask = params[idx + pitch];
	uint right_mask = params[idx + pitch * 2];
	int second_row = params[idx + pitch * 3];
	uint board_mask = (1 << board_size) - 1;
	uint left_mask_big = 0;
	uint right_mask_big = 0;

	uint left_masks[32];
	uint right_masks[32];
	uint ms[32];
	uint ns[33];
	uint solutions = 0;
	int i = 0;

	ms[0] = mask | left_mask | right_mask | (i < second_row ? 2 : 0);
	ns[0] = ((ms[0] + 1) & ~ms[0]);
	left_masks[0] = left_mask;
	right_masks[0] = right_mask;

	results[idx + pitch * 2] = params[idx];	// for checking kernel exeuction completion

	private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
while(i >= 0 || (private_ocl_kernel_loop_boundary_not_reached[0] = false)) {
private_ocl_kernel_loop_iter_counter[0]++;

		if((ns[i] & board_mask) != 0) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

			mask |= ns[i];
			left_masks[i+1] = (left_masks[i] | ns[i]) << 1;
			right_masks[i+1] = (right_masks[i] | ns[i]) >> 1;
			ms[i+1] = mask | left_masks[i+1] | right_masks[i+1] | (i + 1 < second_row ? 2 : 0);
			ns[i + 1] = ((ms[i+1] + 1) & ~ms[i+1]);
			i++;
		}
		else {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);

			if(i == level) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);

				solutions++;
			}else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);
}


			i--;
			if(i >= 0) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[4], 1);

				mask &= ~ns[i];
				ms[i] |= ns[i];
				ns[i] = ((ms[i] + 1) & ~ms[i]);
			}else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[5], 1);
}

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

	results[idx] = solutions * 8;
	results[idx + pitch] = solutions;
	results[idx + pitch * 3] = params[idx];	// for checking kernel exeuction completion
for (int update_recorder_i = 0; update_recorder_i < 6; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
for (int update_recorder_i = 0; update_recorder_i < 1; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}
