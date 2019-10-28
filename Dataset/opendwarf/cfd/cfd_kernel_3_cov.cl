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


__kernel void compute_step_factor(int nelr, __global float* variables, __global float* areas, __global float* step_factors)
{
	const int i = get_global_id(0);

	float density = variables[i + VAR_DENSITY*nelr];
	float3 momentum;
	momentum.x = variables[i + (VAR_MOMENTUM+0)*nelr];
	momentum.y = variables[i + (VAR_MOMENTUM+1)*nelr];
	momentum.z = variables[i + (VAR_MOMENTUM+2)*nelr];

	float density_energy = variables[i + VAR_DENSITY_ENERGY*nelr];

	float3 velocity;       compute_velocity(density, momentum, &velocity);
	float speed_sqd      = compute_speed_sqd(velocity);
	float pressure       = compute_pressure(density, density_energy, speed_sqd);
	float speed_of_sound = compute_speed_of_sound(density, pressure);

	// dt = float(0.5) * sqrt(areas[i]) /  (||v|| + c).... but when we do time stepping, this later would need to be divided by the area, so we just do it all at once
	step_factors[i] = 0.5f / (sqrt(areas[i]) * (sqrt(speed_sqd) + speed_of_sound));
}
