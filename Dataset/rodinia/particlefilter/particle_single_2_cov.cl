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

void cdfCalc(__global float * CDF, __global float * weights, int Nparticles){
	int x;
	CDF[0] = weights[0];
	for(x = 1; x < Nparticles; x++){
		CDF[x] = weights[x] + CDF[x-1];
	}
}
/*****************************
* RANDU
* GENERATES A UNIFORM DISTRIBUTION
* returns a float representing a randomily generated number from a uniform distribution with range [0, 1)
******************************/
float d_randu(__global int * seed, int index)
{

	int M = INT_MAX;
	int A = 1103515245;
	int C = 12345;
	int num = A*seed[index] + C;
	seed[index] = num % M;
	return fabs(seed[index] / ((float) M));
}

/**
* Generates a normally distributed random number using the Box-Muller transformation
* @note This function is thread-safe
* @param seed The seed array
* @param index The specific index of the seed to be advanced
* @return a float representing random number generated using the Box-Muller algorithm
* @see http://en.wikipedia.org/wiki/Normal_distribution, section computing value for normal random distribution
*/

/****************************
UPDATE WEIGHTS
UPDATES WEIGHTS
param1 weights
param2 likelihood
param3 Nparticles
****************************/



/*******************************************
* OpenCL helper function to read a single element from a 2d image
* param1: img
* param2: index
*******************************************/

/*****************************
* CUDA Find Index Kernel Function to replace FindIndex
* param1: arrayX
* param2: arrayY
* param3: CDF
* param4: u
* param5: xj
* param6: yj
* param7: weights
* param8: Nparticles
*****************************/

__kernel void normalize_weights_kernel(__global float * weights, int Nparticles, __global float * partial_sums, __global float * CDF, __global float * u, __global int * seed)
{
	int i = get_global_id(0);
	int local_id = get_local_id(0);
	__local float u1;
	__local float sumWeights;

	if(0 == local_id)
		sumWeights = partial_sums[0];

	barrier(CLK_LOCAL_MEM_FENCE);

	if(i < Nparticles) {
		weights[i] = weights[i]/sumWeights;
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if(i == 0) {
		cdfCalc(CDF, weights, Nparticles);
		u[0] = (1/((float)(Nparticles))) * d_randu(seed, i); // do this to allow all threads in all blocks to use the same u1
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if(0 == local_id)
		u1 = u[0];

	barrier(CLK_LOCAL_MEM_FENCE);

	if(i < Nparticles)
	{
		u[i] = u1 + i/((float)(Nparticles));
	}
}
