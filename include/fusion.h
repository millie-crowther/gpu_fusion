#ifndef FUSION_H
#define FUSION_H

#include <string>
#include <vector>

class fusion_t {
public:
    // constructors and destructors
    fusion_t();
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
    // N.B. pointer is a device pointer
    float * phi_global;

    // properties of volume
    int width, height, depth;
    float voxel_length;

    // private methods
    void load_filenames(std::vector<std::string> * fns, int frames);
    sdf_t get_sdf(std::string filename);
    void update(bool is_rigid, sdf_t sdf);
};

#endif
