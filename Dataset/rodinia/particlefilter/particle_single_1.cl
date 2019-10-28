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

__kernel void find_index_kernel(__global float * arrayX, __global float * arrayY,
	__global float * CDF, __global float * u, __global float * xj,
	__global float * yj, __global float * weights, int Nparticles
	){
		int i = get_global_id(0);

		if(i < Nparticles){

			int index = -1;
			int x;

			for(x = 0; x < Nparticles; x++){
				if(CDF[x] >= u[i]){
					index = x;
					break;
				}
			}
			if(index == -1){
				index = Nparticles-1;
			}

			xj[i] = arrayX[index];
			yj[i] = arrayY[index];

			//weights[i] = 1 / ((float) (Nparticles)); //moved this code to the beginning of likelihood kernel

		}
		barrier(CLK_GLOBAL_MEM_FENCE);
}



/*****************************
* OpenCL Likelihood Kernel Function to replace FindIndex
* param1: arrayX
* param2: arrayY
* param2.5: CDF
* param3: ind
* param4: objxy
* param5: likelihood
* param6: I
* param6.5: u
* param6.75: weights
* param7: Nparticles
* param8: countOnes
* param9: max_size
* param10: k
* param11: IszY
* param12: Nfr
*****************************/


//#endif
