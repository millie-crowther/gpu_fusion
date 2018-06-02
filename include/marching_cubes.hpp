#ifndef MARCHING_CUBES_HPP
#define MARCHING_CUBES_HPP

#include <string>
#include "cutil_math.hpp"

struct triangle_t {
    float3 vertices[3];
};

typedef std::vector<triangle_t> mesh_t;

void save_mesh(std::string model_name, int frame);

#endif
