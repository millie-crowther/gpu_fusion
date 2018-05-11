#ifndef MIN_PARAMS_H
#define MIN_PARAMS_H

#include "function.h"

struct min_params_t {
    // learning rates for rigid and non-rigid alignment
    float eta;

    // relative weighting of killing condition
    float omega_k;

    // relative weighting of level set condition
    float omega_s;

    // killing condition purity
    float gamma;

    // prevention of division by zero in level set gradient
    float epsilon;

    // threshold for terminating registration in mm
    float threshold;

    bool is_multithreaded;
    int frames;

    point_t size;
    float voxel_length;
    float near_clip;

    float delta;
    float sdf_eta;
};

#endif
