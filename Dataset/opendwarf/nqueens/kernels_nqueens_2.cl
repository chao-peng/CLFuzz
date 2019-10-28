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


__kernel void nqueen1(int board_size, int level, int threads, int pitch, __global uint* params, __global uint* results, __constant uint* forbidden)
{
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

	while(i >= 0) {
		if((ns[i] & board_mask) != 0) {
			mask |= ns[i];
			left_masks[i+1] = (left_masks[i] | ns[i]) << 1;
			right_masks[i+1] = (right_masks[i] | ns[i]) >> 1;
			ms[i+1] = mask | left_masks[i+1] | right_masks[i+1] | (i + 1 < second_row ? 2 : 0);
			ns[i + 1] = ((ms[i+1] + 1) & ~ms[i+1]);
			i++;
		}
		else {
			if(i == level) {
				solutions++;
			}

			i--;
			if(i >= 0) {
				mask &= ~ns[i];
				ms[i] |= ns[i];
				ns[i] = ((ms[i] + 1) & ~ms[i]);
			}
		}
	}

	results[idx] = solutions * 8;
	results[idx + pitch] = solutions;
	results[idx + pitch * 3] = params[idx];	// for checking kernel exeuction completion
}
