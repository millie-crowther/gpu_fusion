#ifndef FUSION_H
#define FUSION_H

#include <string>
#include <vector>
#include "min_params.h"

class fusion_t {
public:
    // constructors and destructors
    fusion_t(min_params_t * ps);
    ~fusion_t();
    
    // main public method
    void fusion();

private:
    // deformation field
    // N.B. pointers are device pointers
    float * u;
    float * v;
    float * w;

    // canonical sdf
    // N.B. pointers are device pointers
    float * device_phi;
    float * phi_global;

    // hyper parameters
    min_params_t * ps;

    // private methods
    void get_sdf(int i, int * phi);
    void update(bool is_rigid, int * phi);
};

#endif
