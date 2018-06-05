#ifndef FUSION_H
#define FUSION_H

#include <string>
#include <vector>
#include "min_params.hpp"
#include "cutil_math.hpp"

class fusion_t {
public:
    // constructors and destructors
    fusion_t(min_params_t * ps);
    ~fusion_t();
    
    // main public method
    void fusion();

private:
    // N.B. pointers are device pointers
    int * device_phi;
    float2 * phi_global;
    float3 * psi;

    // hyper parameters
    min_params_t * ps;

    // private methods
    void get_sdf(int i, int * phi, int3 dim);
};

#endif
