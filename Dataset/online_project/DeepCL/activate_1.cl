// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

// expected defines:
// one of: [ TANH | RELU | LINEAR | SIGMOID | SCALEDTANH | ELU ]

#define ACTIVATION_FUNCTION(output) (1.0f / (1 + exp(-output)))


__kernel void activate(const int N, global float *inout) {
    const int globalId = get_global_id(0);
    if (globalId >= N) {
        return;
    }
    inout[globalId] = ACTIVATION_FUNCTION(inout[globalId]);
}
