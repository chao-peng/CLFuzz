// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

// expected defines:
// one of: [ TANH | RELU | LINEAR ]
// BIASED (or not)

#define ACTIVATION_FUNCTION(output) (1.7159f * tanh(0.66667f * output))


__kernel void convolve_floats(global const int *p_imageSize, global const int *p_filterSize,
      global const float *image, global const float *filter, global float *result, __global int* ocl_kernel_loop_recorder) {__local int my_ocl_kernel_loop_recorder[2];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 2; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[2];
bool private_ocl_kernel_loop_boundary_not_reached[2];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    int id = get_global_id(0);
    int imageSize = p_imageSize[0];
    int filterSize = p_filterSize[0];
    int imageOffset = id / (imageSize * imageSize) * (imageSize * imageSize);
    int localid = id % (imageSize * imageSize);
    int row = localid / imageSize;
    int col = localid % imageSize;
    int halfFilterSize = filterSize >> 1;
    float sum = 0;
    int minm = max(-halfFilterSize, -row);
    int maxm = min(halfFilterSize, imageSize - 1 - row);
    int minn = max(-halfFilterSize, -col);
    int maxn = min(halfFilterSize, imageSize - 1 - col);
    int m = minm;
    private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
while(m <= maxm || (private_ocl_kernel_loop_boundary_not_reached[0] = false)) {
private_ocl_kernel_loop_iter_counter[0]++;

        int x = (row + m);
        int ximage = imageOffset + x * imageSize;
        int filterrowoffset = (m+halfFilterSize) * filterSize + halfFilterSize;
        int n = minn;
        private_ocl_kernel_loop_iter_counter[1] = 0;
private_ocl_kernel_loop_boundary_not_reached[1] = true;
while(n <= maxn || (private_ocl_kernel_loop_boundary_not_reached[1] = false)) {
private_ocl_kernel_loop_iter_counter[1]++;

            int y = col + n;
            sum += image[ ximage + y] * filter[ filterrowoffset + n ];
            n++;
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
        m++;
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
    result[id] = sum;
for (int update_recorder_i = 0; update_recorder_i < 2; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}

// receive images as a stack of images
// globalid = n * numfilters * imagesize * imagesize + filter * imagesize * imagesize + imagerow * imagesize + imagecol
//                                 globalid              globalid
//  inputimage3 1 inputimage2 1----filter 1             -> outputimage2 1   outputimage3 1
//                inputimage2 2_/\_filter 2             -> outputimage2 2
//  inputimage3 2 inputimage2 3    filter 1             -> outputimage2 3   outputimage3 2
//                inputimage2 4    filter 2             -> outputimage2 4
//
// each outputimage is only written once, by a combination of:
// - one inputimage3
// - one filter
// each inputimage3 is mapped to each filter once, each time writing to one outputimage
//
// images is:
//       numimages * numinputplanes * imagesizesquared
// filters is:
//       numfilters * numinputplanes * filtersizesquared
// outputs is:
//       numimages * numfilters * outputimagesizesquared

// images are organized like [imageId][plane][row][col]
// filters are organized like [filterid][plane][filterrow][filtercol]
// output are organized like [imageid][filterid][row][col]
