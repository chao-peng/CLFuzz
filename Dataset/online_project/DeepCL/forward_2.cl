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
      global const float *image, global const float *filter, global float *result) {
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
    while(m <= maxm) {
        int x = (row + m);
        int ximage = imageOffset + x * imageSize;
        int filterrowoffset = (m+halfFilterSize) * filterSize + halfFilterSize;
        int n = minn;
        while(n <= maxn) {
            int y = col + n;
            sum += image[ ximage + y] * filter[ filterrowoffset + n ];
            n++;
        }
        m++;
    }
    result[id] = sum;
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
