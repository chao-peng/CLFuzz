#ifdef SINGLE_PRECISION
#define POSVECTYPE float4
#define FORCEVECTYPE float4
#define FPTYPE float
#elif K_DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define POSVECTYPE double4
#define FORCEVECTYPE double4
#define FPTYPE double
#elif AMD_DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_amd_fp64: enable
#define POSVECTYPE double4
#define FORCEVECTYPE double4
#define FPTYPE double
#endif

#define POSVECTYPE float4
#define FORCEVECTYPE float4
#define FPTYPE float

__kernel void compute_lj_force(__global FORCEVECTYPE *force,
                               __global POSVECTYPE *position,
                               const int neighCount,
                               __global int* neighList,
                               const FPTYPE cutsq,
                               const FPTYPE lj1,
                               const FPTYPE lj2,
                               const int inum, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_kernel_loop_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int my_ocl_kernel_loop_recorder[1];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 1; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[1];
bool private_ocl_kernel_loop_boundary_not_reached[1];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    uint idx = get_global_id(0);

    POSVECTYPE ipos = position[idx];
    FORCEVECTYPE f = {0.0f, 0.0f, 0.0f, 0.0f};

    int j = 0;
    private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
while (j < neighCount || (private_ocl_kernel_loop_boundary_not_reached[0] = false))
    {
private_ocl_kernel_loop_iter_counter[0]++;

        int jidx = neighList[j*inum + idx];

        // Uncoalesced read
        POSVECTYPE jpos = position[jidx];

        // Calculate distance
        FPTYPE delx = ipos.x - jpos.x;
        FPTYPE dely = ipos.y - jpos.y;
        FPTYPE delz = ipos.z - jpos.z;
        FPTYPE r2inv = delx*delx + dely*dely + delz*delz;

        // If distance is less than cutoff, calculate force
        if (r2inv < cutsq)
        {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

            r2inv = 1.0f/r2inv;
            FPTYPE r6inv = r2inv * r2inv * r2inv;
            FPTYPE forceC = r2inv*r6inv*(lj1*r6inv - lj2);

            f.x += delx * forceC;
            f.y += dely * forceC;
            f.z += delz * forceC;
        }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}
        j++;
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
    // store the results
    force[idx] = f;
for (int update_recorder_i = 0; update_recorder_i < 2; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
for (int update_recorder_i = 0; update_recorder_i < 1; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}
