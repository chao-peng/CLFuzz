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


__kernel void sum_kernel(__global float* partial_sums, int Nparticles)
{

	int i = get_global_id(0);
        size_t THREADS_PER_BLOCK = get_local_size(0);

	if(i == 0)
	{
		int x;
		float sum = 0;
		int num_blocks = ceil((float) Nparticles / (float) THREADS_PER_BLOCK);
		for (x = 0; x < num_blocks; x++) {
			sum += partial_sums[x];
		}
		partial_sums[0] = sum;
	}
}
