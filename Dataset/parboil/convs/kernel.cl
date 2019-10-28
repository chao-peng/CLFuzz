__kernel void convolute(
  __global float * output,
  const __global float * input,
  const __global float * filter,
  int HALF_FILTER_SIZE,
  int IMAGE_H,
  int IMAGE_W
)
{

  int row = get_global_id(1);
  int col = get_global_id(0);
  int idx = col + row * IMAGE_W;

  if (
    col < HALF_FILTER_SIZE ||
    col > IMAGE_W - HALF_FILTER_SIZE - 1 ||
    row < HALF_FILTER_SIZE ||
    row > IMAGE_H - HALF_FILTER_SIZE - 1
  ) {
    if (row < IMAGE_W && col < IMAGE_H) {
      output[idx] = 0.0f;
    }
  } else {
    // perform convolution
    int fIndex = 0;
    float result = 0.0f;

    for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; r++) {
      for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; c++) {
        int offset = c + r * IMAGE_W;
        result += input[ idx + offset ] * filter[fIndex];
        fIndex++;
      }
    }
    output[idx] = result;
  }
}
