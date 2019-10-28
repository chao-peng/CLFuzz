__kernel void convolute(
  __global float * output,
  const __global float * input,
  const __global float * filter,
  int HALF_FILTER_SIZE,
  int IMAGE_H,
  int IMAGE_W
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


  int row = get_global_id(1);
  int col = get_global_id(0);
  int idx = col + row * IMAGE_W;

  if (
    col < HALF_FILTER_SIZE ||
    col > IMAGE_W - HALF_FILTER_SIZE - 1 ||
    row < HALF_FILTER_SIZE ||
    row > IMAGE_H - HALF_FILTER_SIZE - 1
  ) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

    if (row < IMAGE_W && col < IMAGE_H) {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);

      output[idx] = 0.0f;
    }
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);
}
  } else {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);

    // perform convolution
    int fIndex = 0;
    float result = 0.0f;

    private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE || (private_ocl_kernel_loop_boundary_not_reached[0] = false); r++) {
private_ocl_kernel_loop_iter_counter[0]++;

      private_ocl_kernel_loop_iter_counter[1] = 0;
private_ocl_kernel_loop_boundary_not_reached[1] = true;
for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE || (private_ocl_kernel_loop_boundary_not_reached[1] = false); c++) {
private_ocl_kernel_loop_iter_counter[1]++;

        int offset = c + r * IMAGE_W;
        result += input[ idx + offset ] * filter[fIndex];
        fIndex++;
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
    output[idx] = result;
  }
for (int update_recorder_i = 0; update_recorder_i < 4; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
for (int update_recorder_i = 0; update_recorder_i < 2; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}

