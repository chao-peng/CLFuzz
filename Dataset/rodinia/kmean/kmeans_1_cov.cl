#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

__kernel void
kmeans_kernel_c(__global float  *feature,
			  __global float  *clusters,
			  __global int    *membership,
			    int     npoints,
				int     nclusters,
				int     nfeatures,
				int		offset,
				int		size
			  , __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_kernel_loop_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[4];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 4; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int my_ocl_kernel_loop_recorder[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[2];
bool private_ocl_kernel_loop_boundary_not_reached[2];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	unsigned int point_id = get_global_id(0);
    int index = 0;
    //const unsigned int point_id = get_global_id(0);
		if (point_id < npoints)
		{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

			float min_dist=FLT_MAX;
			private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
for (int i=0; i < nclusters || (private_ocl_kernel_loop_boundary_not_reached[0] = false); i++) {
private_ocl_kernel_loop_iter_counter[0]++;


				float dist = 0;
				float ans  = 0;
				private_ocl_kernel_loop_iter_counter[1] = 0;
private_ocl_kernel_loop_boundary_not_reached[1] = true;
for (int l=0; l<nfeatures || (private_ocl_kernel_loop_boundary_not_reached[1] = false); l++){
private_ocl_kernel_loop_iter_counter[1]++;

						ans += (feature[l * npoints + point_id]-clusters[i*nfeatures+l])*
							   (feature[l * npoints + point_id]-clusters[i*nfeatures+l]);
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

				dist = ans;
				if (dist < min_dist) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);

					min_dist = dist;
					index    = i;

				}else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);
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
		  //printf("%d\n", index);
		  membership[point_id] = index;
		}else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}


	return;
for (int update_recorder_i = 0; update_recorder_i < 4; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
for (int update_recorder_i = 0; update_recorder_i < 2; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}
