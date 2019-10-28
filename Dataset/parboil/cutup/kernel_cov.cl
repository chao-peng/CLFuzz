/*
 * potential lattice is decomposed into size 8^3 lattice point "regions"
 *
 * THIS IMPLEMENTATION:  one thread per lattice point
 * thread block size 128 gives 4 thread blocks per region
 * kernel is invoked for each x-y plane of regions,
 * where gridDim.x is 4*(x region dimension) so that blockIdx.x 
 * can absorb the z sub-region index in its 2 lowest order bits
 *
 * Regions are stored contiguously in memory in row-major order
 *
 * The bins have to not only cover the region, but they need to surround
 * the outer edges so that region sides and corners can still use
 * neighbor list stencil.  The binZeroAddr is actually a shifted pointer into
 * the bin array (binZeroAddr = binBaseAddr + (c*binDim_y + c)*binDim_x + c)
 * where c = ceil(cutoff / binsize).  This allows for negative offsets to
 * be added to myBinIndex.
 *
 * The (0,0,0) spatial origin corresponds to lower left corner of both
 * regionZeroAddr and binZeroAddr.  The atom coordinates are translated
 * during binning to enforce this assumption.
 */

#include "macros.h"

// OpenCL 1.1 support for int3 is not uniform on all implementations, so
// we use int4 instead.  Only the 'x', 'y', and 'z' fields of xyz are used.
typedef int4 xyz;

__kernel void opencl_cutoff_potential_lattice(
    int binDim_x,
    int binDim_y,
    __global float4 *binBaseAddr,
    int offset,
    float h,                /* lattice spacing */
    float cutoff2,          /* square of cutoff distance */
    float inv_cutoff2,
    __global float *regionZeroAddr,  /* address of lattice regions starting at origin */
    int zRegionIndex,
    __constant int *NbrListLen,
    __constant xyz *NbrList
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

  __global float4* binZeroAddr = binBaseAddr + offset;

  __global float *myRegionAddr;
  int Bx, By, Bz;

  /* thread id */
  const int tid = (get_local_id(2)*get_local_size(1) + 
                      get_local_id(1))*get_local_size(0) + get_local_id(0);

  /* this is the start of the sub-region indexed by tid */
  myRegionAddr = regionZeroAddr + ((zRegionIndex*get_num_groups(1)
	+ get_group_id(1))*(get_num_groups(0)>>2) + (get_group_id(0)>>2))*REGION_SIZE
	+ (get_group_id(0)&3)*SUB_REGION_SIZE;

  /* spatial coordinate of this lattice point */
  float x = (8 * (get_group_id(0) >> 2) + get_local_id(0)) * h;
  float y = (8 * get_group_id(1) + get_local_id(1)) * h;
  float z = (8 * zRegionIndex + 2*(get_group_id(0)&3) + get_local_id(2)) * h;

  float dx;
  float dy;
  float dz;
  float r2;
  float s;

  int totalbins = 0;

  /* bin number determined by center of region */
  Bx = (int) floor((8 * (get_group_id(0) >> 2) + 4) * h * BIN_INVLEN);
  By = (int) floor((8 * get_group_id(1) + 4) * h * BIN_INVLEN);
  Bz = (int) floor((8 * zRegionIndex + 4) * h * BIN_INVLEN);

  float energy = 0.f;
  int bincnt;
  private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
for (bincnt = 0; bincnt < *NbrListLen || (private_ocl_kernel_loop_boundary_not_reached[0] = false);  bincnt++) {
private_ocl_kernel_loop_iter_counter[0]++;

    int i = Bx + NbrList[bincnt].x;
    int j = By + NbrList[bincnt].y;
    int k = Bz + NbrList[bincnt].z;

  	__global float4* p_global = binZeroAddr + 
  	                       (((k*binDim_y + j)*binDim_x + i) * BIN_DEPTH);

    int m;
    private_ocl_kernel_loop_iter_counter[1] = 0;
private_ocl_kernel_loop_boundary_not_reached[1] = true;
for (m = 0;  m < BIN_DEPTH || (private_ocl_kernel_loop_boundary_not_reached[1] = false);  m++) {
private_ocl_kernel_loop_iter_counter[1]++;

    	float aq = p_global[m].w;
      if (0.f != aq) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

        dx = p_global[m].x - x;
        dy = p_global[m].y - y;
        dz = p_global[m].z - z;
        r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < cutoff2) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);

          s = (1.f - r2 * inv_cutoff2);
          energy += aq * rsqrt(r2) * s * s;
        }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);
}
      }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}
    }
if (private_ocl_kernel_loop_iter_counter[1] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[1], 1);
}if (private_ocl_kernel_loop_iter_counter[1] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[1], 2);
}if (private_ocl_kernel_loop_iter_counter[1] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[1], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[1]) {
    atomic_or(&my_ocl_kernel_loop_recorder[1], 8);
} /* end loop over atoms in bin */
  }
if (private_ocl_kernel_loop_iter_counter[0] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[0], 1);
}if (private_ocl_kernel_loop_iter_counter[0] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[0], 2);
}if (private_ocl_kernel_loop_iter_counter[0] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[0], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[0]) {
    atomic_or(&my_ocl_kernel_loop_recorder[0], 8);
} /* end loop over neighbor list */

  /* store into global memory */
  myRegionAddr[tid+0] = energy;
for (int update_recorder_i = 0; update_recorder_i < 4; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
for (int update_recorder_i = 0; update_recorder_i < 2; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}
