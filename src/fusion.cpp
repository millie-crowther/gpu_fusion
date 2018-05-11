#include "fusion.h"

#include <iostream>
#include <chrono>
#include <unistd.h>
#include "CImg.h"
#include <string>

using namespace cimg_library;

// declare cuda functions
void initialise(
    float * phi, 
    float ** phi_global, 
    float ** u, float ** v, float ** w,
    int width, int height, int depth
);

void update_rigid(
    float * phi, 
    float * phi_global, 
    float * u, float * v, float * w,
    int width, int height, int depth
);

void update_nonrigid(
    float * phi, 
    float * phi_global, 
    float * u, float * v, float * w,
    int width, int height, int depth
);

fusion_t::fusion_t(min_params_t * ps){
    this->ps = ps;
}

fusion_t::~fusion_t(){

}

void 
fusion_t::get_sdf(int i, int * phi){
    // get filename
    std::string padding = i < 10 ? "0" : "";
    if (i < 100){
       padding = "0" + padding;
    }
    std::string filename = "../data/umbrella/depth/frame-000" + 
	                   padding + std::to_string(i) + ".depth.png";

    std::cout << "loading depth map: " << filename << std::endl; 

    // get image data
    CImg<unsigned short> image(filename.c_str());

    int j = 0;
    for (int x = 0; x < image.width(); x++){
        for (int y = 0; y < image.height(); y++){
            phi[j] = (int) *image.data(x, y);
	    j++;
        }
    }
}

void 
fusion_t::update(bool is_rigid, int * phi){
    std::string msg = is_rigid ? "Rigid" : "Non-rigid";

    bool should_update = true;
    for (int i = 1; should_update; i++){
        std::cout << msg << " transformation, iteration " << i << std::endl;

	should_update = false;
	if (is_rigid){
	    //rigid update
	} else {
	    //non-rigid update
	}
    }

    std::cout << msg << " transformation converged." << std::endl;
}

void
fusion_t::fusion(){
    // initalise deform field and canonical sdf
    int phi[ps->width * ps->height];
    get_sdf(0, phi);

    // perform main fusion
    auto start = std::chrono::system_clock::now();
    for (int i = 1; i < ps->frames; i++){
        std::cout << "Frame number: " << i << std::endl;     

        get_sdf(i, phi); 
        update(true, phi);
	update(false, phi);

        std::cout << std::endl;
    } 
    auto end = std::chrono::system_clock::now();

    // output FPS
    std::chrono::duration<float> elapsed_seconds = end - start;    
    float t = elapsed_seconds.count();
    std::cout << "Total time elapsed: " << t << " seconds." << std::endl;
    std::cout << "Average framerate: " << ps->frames / t << " frames per second." << std::endl;
}
