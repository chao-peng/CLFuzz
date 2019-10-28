/* ============================================================
//--functions: 	kernel funtion
//--programmer:	Jianbin Fang
//--date:		24/03/2011
============================================================ */
#ifndef _KERNEL_
#define _KERNEL_

#define GAMMA (1.4f)


#define NDIM 3
#define NNB 4

#define RK 3	// 3rd order RK
#define ff_mach 1.2f
#define deg_angle_of_attack 0.0f

#define VAR_DENSITY 0
#define VAR_MOMENTUM  1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM+NDIM)
#define NVAR (VAR_DENSITY_ENERGY+1)
//#pragma OPENCL EXTENSION CL_MAD : enable

//self-defined user type
typedef struct{
	float x;
	float y;
	float z;
} FLOAT3;
/*------------------------------------------------------------
	@function:	set memory
	@params:
		mem_d: 		target memory to be set;
		val:		set the target memory to value 'val'
		num_bytes:	the number of bytes all together
	@return:	through mem_d
------------------------------------------------------------*/


//--cambine: omit &
inline void compute_velocity(float  density, FLOAT3 momentum, FLOAT3* velocity, __local int* my_ocl_kernel_branch_triggered_recorder){
	velocity->x = momentum.x / density;
	velocity->y = momentum.y / density;
	velocity->z = momentum.z / density;
}

inline float compute_speed_sqd(FLOAT3 velocity, __local int* my_ocl_kernel_branch_triggered_recorder){
	return velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z;
}

inline float compute_pressure(float density, float density_energy, float speed_sqd, __local int* my_ocl_kernel_branch_triggered_recorder){
	return ((float)(GAMMA) - (float)(1.0f))*(density_energy - (float)(0.5f)*density*speed_sqd);
}
inline float compute_speed_of_sound(float density, float pressure, __local int* my_ocl_kernel_branch_triggered_recorder){
	//return sqrtf(float(GAMMA)*pressure/density);
	return sqrt((float)(GAMMA)*pressure/density);
}
inline void compute_flux_contribution(float density, FLOAT3 momentum, float density_energy, float pressure, FLOAT3 velocity, FLOAT3* fc_momentum_x, FLOAT3* fc_momentum_y, FLOAT3* fc_momentum_z, FLOAT3* fc_density_energy, __local int* my_ocl_kernel_branch_triggered_recorder)
{
	fc_momentum_x->x = velocity.x*momentum.x + pressure;
	fc_momentum_x->y = velocity.x*momentum.y;
	fc_momentum_x->z = velocity.x*momentum.z;


	fc_momentum_y->x = fc_momentum_x->y;
	fc_momentum_y->y = velocity.y*momentum.y + pressure;
	fc_momentum_y->z = velocity.y*momentum.z;

	fc_momentum_z->x = fc_momentum_x->z;
	fc_momentum_z->y = fc_momentum_y->z;
	fc_momentum_z->z = velocity.z*momentum.z + pressure;

	float de_p = density_energy+pressure;
	fc_density_energy->x = velocity.x*de_p;
	fc_density_energy->y = velocity.y*de_p;
	fc_density_energy->z = velocity.z*de_p;
}


__kernel void time_step(int j, int nelr,
				__global float* old_variables, 
				__global float* variables,
				__global float* step_factors,
				__global float* fluxes, __global int* ocl_kernel_branch_triggered_recorder){__local int my_ocl_kernel_branch_triggered_recorder[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	//const int i = (blockDim.x*blockIdx.x + threadIdx.x);
	const int i = get_global_id(0);
	if( i >= nelr) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);
return;
} else { 
atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}


	float factor = step_factors[i]/(float)(RK+1-j);

	variables[i + VAR_DENSITY*nelr] = old_variables[i + VAR_DENSITY*nelr] + factor*fluxes[i + VAR_DENSITY*nelr];
	variables[i + VAR_DENSITY_ENERGY*nelr] = old_variables[i + VAR_DENSITY_ENERGY*nelr] + factor*fluxes[i + VAR_DENSITY_ENERGY*nelr];
	variables[i + (VAR_MOMENTUM+0)*nelr] = old_variables[i + (VAR_MOMENTUM+0)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+0)*nelr];
	variables[i + (VAR_MOMENTUM+1)*nelr] = old_variables[i + (VAR_MOMENTUM+1)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+1)*nelr];
	variables[i + (VAR_MOMENTUM+2)*nelr] = old_variables[i + (VAR_MOMENTUM+2)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+2)*nelr];

for (int update_recorder_i = 0; update_recorder_i < 2; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
}

#endif
