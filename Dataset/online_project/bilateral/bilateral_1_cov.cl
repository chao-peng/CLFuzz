
/**
 * This kernel performs bilateral filtering of the depth map and outputs the vertex map and normal map
*/

__kernel void measurement_vertices(	__global const ushort* src,
								__global const int* width,
								__global const int* height,
								__global const float* fl,		// focal length
								//__global ushort* dst,
								__global float3* vmap,
								__global const float* sigma_s,
								__global const float* sigma_r
						, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_kernel_loop_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int my_ocl_kernel_loop_recorder[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[2];
bool private_ocl_kernel_loop_boundary_not_reached[2];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

   /* get_global_id(0) returns the ID of the thread in execution.
   As many threads are launched at the same time, executing the same kernel,
   each one will receive a different ID, and consequently perform a different computation.*/

   const int x = get_global_id(0);
   const int y = get_global_id(1);
   const float s = *sigma_s;
   const float r = *sigma_r;

   const int radius = 8;
   int w = *width;	// buffer into shared memory?
   int h = *height;
   float focal_length = *fl;

   int tlx = max(x-radius, 0);
   int tly = max(y-radius, 0);
   int brx = min(x+radius, w);
   int bry = min(y+radius, h);

   int idx = w * y + x;
   //depth_l[idx] = src[idx];
   float sum = 0;
   float wp = 0;	// normalizing constant

	//barrier(CLK_LOCAL_MEM_FENCE);
   float src_depth = src[idx];

   float s2 = s*s;
   float r2 = r*r;

   if(src_depth != 0)
   {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

	   private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
for(int i=tlx; i< brx || (private_ocl_kernel_loop_boundary_not_reached[0] = false); i++)
	   {
private_ocl_kernel_loop_iter_counter[0]++;

			private_ocl_kernel_loop_iter_counter[1] = 0;
private_ocl_kernel_loop_boundary_not_reached[1] = true;
for(int j=tly; j<bry || (private_ocl_kernel_loop_boundary_not_reached[1] = false); j++)
			{
private_ocl_kernel_loop_iter_counter[1]++;

			// cost:
				float delta_dist = (float)((x - i) * (x - i) + (y - j) * (y - j));

				int idx2 = w * j + i;
				float d = src[idx2]; // cost: 0.013s	// TODO : use shared memory?
				float delta_depth = (src_depth - d) * (src_depth - d);
				float weight = native_exp( -(delta_dist / s2 + delta_depth / r2) ); //cost :
				sum += weight * d;
				wp += weight;
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
	   float res = sum / wp;
	   //dst[idx] = res;
	   vmap[idx].x = res*(x-w/2)/focal_length;
	   vmap[idx].y = res*(y-h/2)/focal_length;
	   vmap[idx].z = res;
	   //vmap[idx].x = res*(x-(*pptu))/(*fl);
		//vmap[idx].y = res*(y-(*pptv))/(*fl);
   }
   else
   {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);

		//dst[idx] = NAN;
	   vmap[idx].x = NAN;
	   vmap[idx].y = NAN;
	   vmap[idx].z = NAN;
   }
for (int update_recorder_i = 0; update_recorder_i < 2; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
for (int update_recorder_i = 0; update_recorder_i < 2; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}
