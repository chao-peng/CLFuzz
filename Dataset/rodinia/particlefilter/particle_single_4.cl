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
float dev_round_float(float value) {
    int newValue = (int) (value);
    if (value - newValue < .5f)
        return newValue;
    else
        return newValue++;
}


/********************************
* CALC LIKELIHOOD SUM
* DETERMINES THE LIKELIHOOD SUM BASED ON THE FORMULA: SUM( (IK[IND] - 100)^2 - (IK[IND] - 228)^2)/ 100
* param 1 I 3D matrix
* param 2 current ind array
* param 3 length of ind array
* returns a float representing the sum
********************************/
float calcLikelihoodSum(__global unsigned char * I, __global int * ind, int numOnes, int index){
	float likelihoodSum = 0.0;
	int x;
	for(x = 0; x < numOnes; x++)
		likelihoodSum += (pow((float)(I[ind[index*numOnes + x]] - 100),2) - pow((float)(I[ind[index*numOnes + x]]-228),2))/50.0;
	return likelihoodSum;
}
/****************************
CDF CALCULATE
CALCULATES CDF
param1 CDF
param2 weights
param3 Nparticles
*****************************/

/**
* Generates a normally distributed random number using the Box-Muller transformation
* @note This function is thread-safe
* @param seed The seed array
* @param index The specific index of the seed to be advanced
* @return a float representing random number generated using the Box-Muller algorithm
* @see http://en.wikipedia.org/wiki/Normal_distribution, section computing value for normal random distribution
*/


float d_randu(__global int * seed, int index)
{

	int M = INT_MAX;
	int A = 1103515245;
	int C = 12345;
	int num = A*seed[index] + C;
	seed[index] = num % M;
	return fabs(seed[index] / ((float) M));
}

float d_randn(__global int * seed, int index){
	//Box-Muller algortihm
	float pi = 3.14159265358979323846;
	float u = d_randu(seed, index);
	float v = d_randu(seed, index);
	float cosine = cos(2*pi*v);
	float rt = -2*log(u);
	return sqrt(rt)*cosine;
}




/****************************
UPDATE WEIGHTS
UPDATES WEIGHTS
param1 weights
param2 likelihood
param3 Nparticles
****************************/

__kernel void likelihood_kernel(__global float * arrayX, __global float * arrayY,__global float * xj, __global float * yj, __global float * CDF, __global int * ind, __global int * objxy, __global float * likelihood, __global unsigned char * I, __global float * u, __global float * weights, const int Nparticles, const int countOnes, const int max_size, int k, const int IszY, const int Nfr, __global int *seed, __global float * partial_sums, __local float* buffer){
	int block_id = get_group_id(0);
	int thread_id = get_local_id(0);
	int i = get_global_id(0);
        size_t THREADS_PER_BLOCK = get_local_size(0);
	int y;
	int indX, indY;


	if(i < Nparticles){
		arrayX[i] = xj[i];
		arrayY[i] = yj[i];

		weights[i] = 1 / ((float) (Nparticles)); //Donnie - moved this line from end of find_index_kernel to prevent all weights from being reset before calculating position on final iteration.


		arrayX[i] = arrayX[i] + 1.0 + 5.0*d_randn(seed, i);
		arrayY[i] = arrayY[i] - 2.0 + 2.0*d_randn(seed, i);

	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if(i < Nparticles)
	{
		for(y = 0; y < countOnes; y++){

			indX = dev_round_float(arrayX[i]) + objxy[y*2 + 1];
			indY = dev_round_float(arrayY[i]) + objxy[y*2];

			ind[i*countOnes + y] = abs(indX*IszY*Nfr + indY*Nfr + k);
			if(ind[i*countOnes + y] >= max_size)
				ind[i*countOnes + y] = 0;
		}
		likelihood[i] = calcLikelihoodSum(I, ind, countOnes, i);

		likelihood[i] = likelihood[i]/countOnes-SCALE_FACTOR;

		weights[i] = weights[i] * exp(likelihood[i]); //Donnie Newell - added the missing exponential function call

	}

	buffer[thread_id] = 0.0; // DEBUG!!!!!!!!!!!!!!!!!!!!!!!!
	//buffer[thread_id] = i;

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);


	if(i < Nparticles){
		buffer[thread_id] = weights[i];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	/* for some reason the get_local_size(0) call was not returning 512. */
	//for(unsigned int s=get_local_size(0)/2; s>0; s>>=1)
	for(unsigned int s=THREADS_PER_BLOCK/2; s>0; s>>=1)
	{
		if(thread_id < s)
		{
			buffer[thread_id] += buffer[thread_id + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if(thread_id == 0)
	{
		partial_sums[block_id] = buffer[0];
	}

}//*/

//#endif
