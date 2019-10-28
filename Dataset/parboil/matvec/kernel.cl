__kernel void matvec(
  __global float * output,
  const __global float * mat,
  const __global float * vec,
  int nRows,
  int nCols
)
{
  int tid = get_global_id(0);
  if (tid < nRows) {
    float result = 0.f;
    for (int i = 0; i < nCols; i++) {
      result += mat[tid + nRows * i] * vec[i];
    }
    output[tid] = result;
  }
}
