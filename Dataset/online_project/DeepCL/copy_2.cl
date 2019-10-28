// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

// simply copies from one to the other...
// there might be something built-in to opencl for this
// anyway... :-)

__kernel void copy_with_offset(
        const int N,
        global const float *in,
        const int inoffset,
        global float *out,
        const int outoffset) {
    const int globalId = get_global_id(0);
    if (globalId >= N) {
        return;
    }
    out[globalId + outoffset] = in[globalId + inoffset];
}
