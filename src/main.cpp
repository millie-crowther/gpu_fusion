#include "fusion.h"

int 
main(){
    //declare hyper-parameters
    min_params_t ps;
    ps.frames = 10;
    ps.width = 640;
    ps.height = 480;

    fusion_t f(&ps);
    f.fusion();

    return 0;
}
