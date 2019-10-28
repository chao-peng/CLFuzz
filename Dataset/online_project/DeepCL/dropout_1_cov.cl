// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

__kernel void forwardNaive(
        const int N,
        global const unsigned char *mask,
        global const float *input,
        global float *output, __global int* ocl_kernel_branch_triggered_recorder) {__local int my_ocl_kernel_branch_triggered_recorder[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    const int globalId = get_global_id(0);
    if (globalId >= N) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

        return;
    }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);
}
    output[globalId] = mask[globalId] == 1 ? input[globalId] : 0.0f;
for (int update_recorder_i = 0; update_recorder_i < 2; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
}
