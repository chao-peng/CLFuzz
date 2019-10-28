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


inline int bit_scan(unsigned int x)
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

__kernel void nqueen(int board_size, int level, int threads, int pitch, __global uint* params, __global uint* results, __constant uint* forbidden)
{
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

	while(i >= 0) {
		if((ns[i] & board_mask) != 0) {
			mask |= ns[i];
			left_masks[i+1] = (left_masks[i] | ns[i]) << 1;
			right_masks[i+1] = (right_masks[i] | ns[i]) >> 1;
			ms[i+1] = mask | left_masks[i+1] | right_masks[i+1] | forbidden[i + 1];
			ns[i+1] = ((ms[i+1] + 1) & ~ms[i+1]);
			i++;
		}
		else {
			if(i == level) {
				int repeat_times = 8;
				bool repeat = false;
				bool equal = true;

				bool rotate1 = (BOARD(index) == (1 << (board_size - 1)));
				bool rotate2 = (BOARD(board_size - index - 1) == 1);
				bool rotate3 = (ns[level - 1] == (1 << (board_size - index - 1)));

				if(rotate1 || rotate2 || rotate3) {
					for(j = 0; j < board_size - level; j++) {
						board_array[j] = bit_scan(params[idx + pitch * (4 + j)]);
					}
					for(j = 0; j < level; j++) {
						board_array[j + board_size - level] = bit_scan(ns[j]);
					}

					if(rotate3) {
						// rotate 180
						equal = true;
						for(j = 0; j < board_size; j++) {
							t_array[board_size - j - 1] = (board_size - board_array[j] - 1);
						}

						for(j = 0; j < board_size; j++) {
							if(t_array[j] < board_array[j]) {
								repeat = true;
								equal = false;
								break;
							}
							else if(t_array[j] > board_array[j]) {
								equal = false;
								break;
							}
						}

						if(equal) {
							repeat_times = 4;
						}
					}

					// rotate cw
					if(!repeat && rotate1) {
						equal = true;
						for(j = 0; j < board_size; j++) {
							t_array[board_size - board_array[j] - 1] = j;
						}

						for(j = 0; j < board_size; j++) {
							if(t_array[j] < board_array[j]) {
								repeat = true;
								equal = false;
								break;
							}
							else if(t_array[j] > board_array[j]) {
								equal = false;
								break;
							}
						}

						if(equal) {
							repeat_times = 2;
						}
					}

					if(!repeat && rotate2) {
						// rotate ccw
						equal = true;
						for(j = 0; j < board_size; j++) {
							t_array[board_array[j]] = (board_size - j - 1);
						}
						for(j = 0; j < board_size; j++) {
							if(t_array[j] < board_array[j]) {
								repeat = true;
								equal = false;
								break;
							}
							else if(t_array[j] > board_array[j]) {
								equal = false;
								break;
							}
						}

						if(equal) {
							repeat_times = 2;
						}
					}

					if(!repeat) {
						solutions += repeat_times;
						unique_solutions++;
					}
				}
				else {
					solutions += 8;
					unique_solutions ++;
				}
			}

			i--;
			if(i >= 0) {
				mask &= ~ns[i];
				ms[i] |= ns[i];
				ns[i] = ((ms[i] + 1) & ~ms[i]);
			}
		}
	}

	results[idx] = solutions;
	results[idx + pitch] = unique_solutions;
	results[idx + pitch * 3] = params[idx];	// for checking kernel exeuction completion
}
