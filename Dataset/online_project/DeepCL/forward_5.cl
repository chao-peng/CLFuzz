// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

// expected defines:
// one of: [ TANH | RELU | LINEAR ]
// BIASED (or not)

#ifdef TANH
    #define ACTIVATION_FUNCTION(output) (tanh(output))
#elif defined SCALEDTANH
    #define ACTIVATION_FUNCTION(output) (1.7159f * tanh(0.66667f * output))
#elif SIGMOID
    #define ACTIVATION_FUNCTION(output) (1.0f / (1 + exp(-output)))
#elif defined RELU
    #define ACTIVATION_FUNCTION(output) (output> 0 ? output : 0)
#elif defined LINEAR
    #define ACTIVATION_FUNCTION(output) (output)
#endif


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

__kernel void convolve_imagecubes_float_nopadzeros(
      const int numInputPlanes, const int numFilters,
      const int inputSize, const int filterSize,
      global const float *images, global const float *filters, global float *output) {
    int globalId = get_global_id(0);

    int inputSizeSquared = inputSize * inputSize;
    int outputSize = inputSize - filterSize + 1;
    int outputSizeSquared = outputSize * outputSize;

    int outputImage2Id = globalId / outputSizeSquared;
    int filterId = outputImage2Id % numFilters;
    int inputImage3Id = outputImage2Id / numFilters;

    int filterOffset = filterId * filterSize * filterSize;
    int inputImage3Offset = inputImage3Id * numInputPlanes * inputSizeSquared;

    // intraimage coords
    int localid = globalId % outputSizeSquared;
    int outputRow = localid / outputSize;
    int outputCol = localid % outputSize;

    int halfFilterSize = filterSize >> 1;
    float sum = 0;
    int minm = -halfFilterSize;
    int maxm = halfFilterSize;
    int minn = -halfFilterSize;
    int maxn = halfFilterSize;
    int inputPlane = 0;
    while(inputPlane < numInputPlanes) {
        int inputImageOffset = inputImage3Offset + inputPlane * inputSizeSquared;
        int m = minm;
        while(m <= maxm) {
            int inputRow = outputRow + m + halfFilterSize;
            int inputimagerowoffset = inputImageOffset + inputRow * inputSize;
            int filterrowoffset = filterOffset + (m+halfFilterSize) * filterSize + halfFilterSize;
            int n = minn;
            while(n <= maxn) {
                int inputCol = outputCol + n + halfFilterSize;
                sum += images[ inputimagerowoffset + inputCol] * filters[ filterrowoffset + n ];
                n++;
            }
            m++;
        }
        inputPlane++;
    }
    output[globalId] = sum;
}
