#include "fusion.hpp"

int 
main(){
    //declare hyper-parameters
    min_params_t ps;
    ps.frames = 31;
    ps.width = 640;
    ps.height = 480;
    ps.depth = 1500;
    ps.voxel_length = 8;

    fusion_t f(&ps);
    f.fusion();

    return 0;
}
