#ifndef MARCHING_CUBES_HPP
#define MARCHING_CUBES_HPP

#include <string>
#include "cutil_math.hpp"
#include <vector>

namespace mc {
    struct triangle_t {
        float3 vertices[3];
    };

    typedef std::vector<triangle_t> mesh_t;

    void save_mesh(float2 * data, int3 dim, float l, std::string model_name, int frame);

    float phi(float2 * data, int3 v, int3 dim);
    float3 normal(float2 * data, float3 v, int3 dim, float l);
    void create_mesh(float isolevel, mesh_t * mesh, float2 * data, int3 dim, float l);
    void create_mesh_at(int3 pos, float isolevel, mesh_t * mesh, float2 * data, int3 dim, float l);

    float3 interpolate(float isolevel, float3 a, float3 b, float alpha, float beta);
}
#endif
