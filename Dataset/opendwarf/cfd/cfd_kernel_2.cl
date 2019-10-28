#define NDIM 3
#define NNB 4

#define RK 3	// 3rd order RK

#define GAMMA 1.4f
#define VAR_DENSITY 0
#define VAR_MOMENTUM  1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM+NDIM)
#define NVAR (VAR_DENSITY_ENERGY+1)

//extern int printf(constant char *format, ...);

void compute_flux_contribution(__private float* density, __private float3* momentum, __private float* density_energy,
		float pressure, __private float3* velocity,
		__private float3* fc_momentum_x, __private float3* fc_momentum_y, __private float3* fc_momentum_z,
		__private float3* fc_density_energy)
{
	(*fc_momentum_x).x = (*velocity).x*(*momentum).x + pressure;
	(*fc_momentum_x).y = (*velocity).x*(*momentum).y;
	(*fc_momentum_x).z = (*velocity).x*(*momentum).z;


	(*fc_momentum_y).x = (*fc_momentum_x).y;
	(*fc_momentum_y).y = (*velocity).y*(*momentum).y + pressure;
	(*fc_momentum_y).z = (*velocity).y*(*momentum).z;

	(*fc_momentum_z).x = (*fc_momentum_x).z;
	(*fc_momentum_z).y = (*fc_momentum_y).z;
	(*fc_momentum_z).z = (*velocity).z*(*momentum).z + pressure;

	float de_p = *density_energy+pressure;
	(*fc_density_energy).x = (*velocity).x*de_p;
	(*fc_density_energy).y = (*velocity).y*de_p;
	(*fc_density_energy).z = (*velocity).z*de_p;
}

void compute_velocity(float density, float3 momentum, __private float3* velocity)
{
	(*velocity).x = momentum.x / density;
	(*velocity).y = momentum.y / density;
	(*velocity).z = momentum.z / density;
}

float compute_speed_of_sound(float density, float pressure)
{
	return sqrt(GAMMA*pressure/density);
}

float compute_speed_sqd(float3 velocity)
{
	return velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z;
}

float compute_pressure(float density, float density_energy, float speed_sqd)
{
	return (GAMMA-1.0f)*(density_energy - 0.5f*density*speed_sqd);
}


__kernel void compute_flux_contributions(int nelr, __global float* variables, __global float* fc_momentum_x,
		__global float* fc_momentum_y, __global float* fc_momentum_z, __global float* fc_density_energy)
{
	const int i = get_global_id(0);

	float density_i = variables[i + VAR_DENSITY*nelr];
	float3 momentum_i;
	momentum_i.x = variables[i + (VAR_MOMENTUM+0)*nelr];
	momentum_i.y = variables[i + (VAR_MOMENTUM+1)*nelr];
	momentum_i.z = variables[i + (VAR_MOMENTUM+2)*nelr];
	float density_energy_i = variables[i + VAR_DENSITY_ENERGY*nelr];

	float3 velocity_i;             				compute_velocity(density_i, momentum_i, &velocity_i);
	float speed_sqd_i                          = compute_speed_sqd(velocity_i);
	float speed_i                              = sqrt(speed_sqd_i);
	float pressure_i                           = compute_pressure(density_i, density_energy_i, speed_sqd_i);
	float speed_of_sound_i                     = compute_speed_of_sound(density_i, pressure_i);
	float3 fc_i_momentum_x, fc_i_momentum_y, fc_i_momentum_z;
	float3 fc_i_density_energy;
	compute_flux_contribution(&density_i, &momentum_i, &density_energy_i, pressure_i, &velocity_i, &fc_i_momentum_x, &fc_i_momentum_y, &fc_i_momentum_z, &fc_i_density_energy);

	fc_momentum_x[i + 0*nelr] = fc_i_momentum_x.x;
	fc_momentum_x[i + 1*nelr] = fc_i_momentum_x.y;
	fc_momentum_x[i + 2*nelr] = fc_i_momentum_x.z;

	fc_momentum_y[i + 0*nelr] = fc_i_momentum_y.x;
	fc_momentum_y[i + 1*nelr] = fc_i_momentum_y.y;
	fc_momentum_y[i + 2*nelr] = fc_i_momentum_y.z;


	fc_momentum_z[i + 0*nelr] = fc_i_momentum_z.x;
	fc_momentum_z[i + 1*nelr] = fc_i_momentum_z.y;
	fc_momentum_z[i + 2*nelr] = fc_i_momentum_z.z;

	fc_density_energy[i + 0*nelr] = fc_i_density_energy.x;
	fc_density_energy[i + 1*nelr] = fc_i_density_energy.y;
	fc_density_energy[i + 2*nelr] = fc_i_density_energy.z;
}
