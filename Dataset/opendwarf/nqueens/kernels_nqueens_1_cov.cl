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

__kernel void nqueen(int board_size, int level, int threads, int pitch, __global uint* params, __global uint* results, __constant uint* forbidden, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_kernel_loop_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[34];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 34; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int my_ocl_kernel_loop_recorder[9];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 9; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[9];
bool private_ocl_kernel_loop_boundary_not_reached[9];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	int idx = get_global_id(0);
	int tid = get_local_id(0);
	uint mask = params[idx];
	uint left_mask = params[idx + pitch];
	uint right_mask = params[idx + pitch * 2];
	uint index = params[idx + pitch * 3];
	uint board_mask = (1 << board_size) - 1;
	uint left_mask_big = 0;
	uint right_mask_big = 0;

	uint left_masks[32];
	uint right_masks[32];
	uint ms[32];
	uint ns[33];
	uint solutions = 0;
	uint unique_solutions = 0;
	int i = 0;
	int j;

	int t_array[32];
	int board_array[32];

	ms[0] = mask | left_mask | right_mask | forbidden[0];
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
			ms[i+1] = mask | left_masks[i+1] | right_masks[i+1] | forbidden[i + 1];
			ns[i+1] = ((ms[i+1] + 1) & ~ms[i+1]);
			i++;
		}
		else {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);

			if(i == level) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);

				int repeat_times = 8;
				bool repeat = false;
				bool equal = true;

				bool rotate1 = (BOARD(index) == (1 << (board_size - 1)));
				bool rotate2 = (BOARD(board_size - index - 1) == 1);
				bool rotate3 = (ns[level - 1] == (1 << (board_size - index - 1)));

				if(rotate1 || rotate2 || rotate3) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[4], 1);

					private_ocl_kernel_loop_iter_counter[1] = 0;
private_ocl_kernel_loop_boundary_not_reached[1] = true;
for(j = 0; j < board_size - level || (private_ocl_kernel_loop_boundary_not_reached[1] = false); j++) {
private_ocl_kernel_loop_iter_counter[1]++;

						board_array[j] = bit_scan(params[idx + pitch * (4 + j)], my_ocl_kernel_branch_triggered_recorder, my_ocl_kernel_loop_recorder);
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
					private_ocl_kernel_loop_iter_counter[2] = 0;
private_ocl_kernel_loop_boundary_not_reached[2] = true;
for(j = 0; j < level || (private_ocl_kernel_loop_boundary_not_reached[2] = false); j++) {
private_ocl_kernel_loop_iter_counter[2]++;

						board_array[j + board_size - level] = bit_scan(ns[j], my_ocl_kernel_branch_triggered_recorder, my_ocl_kernel_loop_recorder);
					}
if (private_ocl_kernel_loop_iter_counter[2] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[2], 1);
}if (private_ocl_kernel_loop_iter_counter[2] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[2], 2);
}if (private_ocl_kernel_loop_iter_counter[2] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[2], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[2]) {
    atomic_or(&my_ocl_kernel_loop_recorder[2], 8);
}

					if(rotate3) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[6], 1);

						// rotate 180
						equal = true;
						private_ocl_kernel_loop_iter_counter[3] = 0;
private_ocl_kernel_loop_boundary_not_reached[3] = true;
for(j = 0; j < board_size || (private_ocl_kernel_loop_boundary_not_reached[3] = false); j++) {
private_ocl_kernel_loop_iter_counter[3]++;

							t_array[board_size - j - 1] = (board_size - board_array[j] - 1);
						}
if (private_ocl_kernel_loop_iter_counter[3] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[3], 1);
}if (private_ocl_kernel_loop_iter_counter[3] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[3], 2);
}if (private_ocl_kernel_loop_iter_counter[3] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[3], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[3]) {
    atomic_or(&my_ocl_kernel_loop_recorder[3], 8);
}

						private_ocl_kernel_loop_iter_counter[4] = 0;
private_ocl_kernel_loop_boundary_not_reached[4] = true;
for(j = 0; j < board_size || (private_ocl_kernel_loop_boundary_not_reached[4] = false); j++) {
private_ocl_kernel_loop_iter_counter[4]++;

							if(t_array[j] < board_array[j]) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[8], 1);

								repeat = true;
								equal = false;
								break;
							}
							else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[9], 1);

if(t_array[j] > board_array[j]) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[10], 1);

								equal = false;
								break;
							}else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[11], 1);
}
}

						}
if (private_ocl_kernel_loop_iter_counter[4] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[4], 1);
}if (private_ocl_kernel_loop_iter_counter[4] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[4], 2);
}if (private_ocl_kernel_loop_iter_counter[4] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[4], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[4]) {
    atomic_or(&my_ocl_kernel_loop_recorder[4], 8);
}

						if(equal) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[12], 1);

							repeat_times = 4;
						}else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[13], 1);
}

					}else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[7], 1);
}


					// rotate cw
					if(!repeat && rotate1) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[14], 1);

						equal = true;
						private_ocl_kernel_loop_iter_counter[5] = 0;
private_ocl_kernel_loop_boundary_not_reached[5] = true;
for(j = 0; j < board_size || (private_ocl_kernel_loop_boundary_not_reached[5] = false); j++) {
private_ocl_kernel_loop_iter_counter[5]++;

							t_array[board_size - board_array[j] - 1] = j;
						}
if (private_ocl_kernel_loop_iter_counter[5] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[5], 1);
}if (private_ocl_kernel_loop_iter_counter[5] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[5], 2);
}if (private_ocl_kernel_loop_iter_counter[5] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[5], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[5]) {
    atomic_or(&my_ocl_kernel_loop_recorder[5], 8);
}

						private_ocl_kernel_loop_iter_counter[6] = 0;
private_ocl_kernel_loop_boundary_not_reached[6] = true;
for(j = 0; j < board_size || (private_ocl_kernel_loop_boundary_not_reached[6] = false); j++) {
private_ocl_kernel_loop_iter_counter[6]++;

							if(t_array[j] < board_array[j]) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[16], 1);

								repeat = true;
								equal = false;
								break;
							}
							else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[17], 1);

if(t_array[j] > board_array[j]) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[18], 1);

								equal = false;
								break;
							}else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[19], 1);
}
}

						}
if (private_ocl_kernel_loop_iter_counter[6] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[6], 1);
}if (private_ocl_kernel_loop_iter_counter[6] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[6], 2);
}if (private_ocl_kernel_loop_iter_counter[6] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[6], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[6]) {
    atomic_or(&my_ocl_kernel_loop_recorder[6], 8);
}

						if(equal) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[20], 1);

							repeat_times = 2;
						}else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[21], 1);
}

					}else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[15], 1);
}


					if(!repeat && rotate2) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[22], 1);

						// rotate ccw
						equal = true;
						private_ocl_kernel_loop_iter_counter[7] = 0;
private_ocl_kernel_loop_boundary_not_reached[7] = true;
for(j = 0; j < board_size || (private_ocl_kernel_loop_boundary_not_reached[7] = false); j++) {
private_ocl_kernel_loop_iter_counter[7]++;

							t_array[board_array[j]] = (board_size - j - 1);
						}
if (private_ocl_kernel_loop_iter_counter[7] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[7], 1);
}if (private_ocl_kernel_loop_iter_counter[7] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[7], 2);
}if (private_ocl_kernel_loop_iter_counter[7] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[7], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[7]) {
    atomic_or(&my_ocl_kernel_loop_recorder[7], 8);
}
						private_ocl_kernel_loop_iter_counter[8] = 0;
private_ocl_kernel_loop_boundary_not_reached[8] = true;
for(j = 0; j < board_size || (private_ocl_kernel_loop_boundary_not_reached[8] = false); j++) {
private_ocl_kernel_loop_iter_counter[8]++;

							if(t_array[j] < board_array[j]) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[24], 1);

								repeat = true;
								equal = false;
								break;
							}
							else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[25], 1);

if(t_array[j] > board_array[j]) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[26], 1);

								equal = false;
								break;
							}else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[27], 1);
}
}

						}
if (private_ocl_kernel_loop_iter_counter[8] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[8], 1);
}if (private_ocl_kernel_loop_iter_counter[8] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[8], 2);
}if (private_ocl_kernel_loop_iter_counter[8] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[8], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[8]) {
    atomic_or(&my_ocl_kernel_loop_recorder[8], 8);
}

						if(equal) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[28], 1);

							repeat_times = 2;
						}else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[29], 1);
}

					}else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[23], 1);
}


					if(!repeat) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[30], 1);

						solutions += repeat_times;
						unique_solutions++;
					}else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[31], 1);
}

				}
				else {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[5], 1);

					solutions += 8;
					unique_solutions ++;
				}
			}else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);
}


			i--;
			if(i >= 0) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[32], 1);

				mask &= ~ns[i];
				ms[i] |= ns[i];
				ns[i] = ((ms[i] + 1) & ~ms[i]);
			}else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[33], 1);
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

	results[idx] = solutions;
	results[idx + pitch] = unique_solutions;
	results[idx + pitch * 3] = params[idx];	// for checking kernel exeuction completion
for (int update_recorder_i = 0; update_recorder_i < 34; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
for (int update_recorder_i = 0; update_recorder_i < 9; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}
