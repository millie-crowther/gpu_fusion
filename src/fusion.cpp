#include "fusion.hpp"

#include <iostream>
#include <chrono>
#include <unistd.h>
#include "CImg.h"

#include "cutil_math.hpp"

using namespace cimg_library;

// declare cuda functions
void initialise(int * phi, int ** device_phi, float2 ** phi_global, float3 ** psi, int3 dim);
void update_rigid(int * phi, int * device_phi, float2 * phi_global, float3 * psi, int3 dim);
void update_nonrigid(int * device_phi, float2 * phi_global, float3 * psi, int3 dim);
void cleanup(float2 * phi_global, float3 * psi, int * device_phi);
void get_canon(float2 * host_phi_global, float2 * device_phi_global, int size);
 

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
fusion_t::fusion(){
    // initalise deform field and canonical sdf
    int phi[ps->width * ps->height];
    get_sdf(0, phi);
 
    // device pointers
    int3 dim = make_int3(ps->width, ps->height, ps->depth) / ps->voxel_length; 

    // store canon sdf data
    float2 host_phi_global[dim.x * dim.y * dim.z];

    initialise(phi, &device_phi, &phi_global, &psi, dim);

    // perform main fusion
    auto start = std::chrono::system_clock::now();
    for (int i = 1; i < ps->frames; i++){
        std::cout << "Frame number: " << i << std::endl;     

        get_sdf(i, phi); 
	std::cout << "Performing rigid deformation..." << std::endl;
	update_rigid(phi, device_phi, phi_global, psi, dim);
	std::cout << "Rigid deformation converged." << std::endl;

	std::cout << "Performing non-rigid deformation..." << std::endl;
	update_nonrigid(device_phi, phi_global, psi, dim);
	std::cout << "Non-rigid deformation converged." << std::endl;

	if (i % 30 == 0){
            std::cout << "Storing canonical SDF data..." << std::endl;
	    get_canon(host_phi_global, phi_global, dim.x * dim.y * dim.z);
	}

        std::cout << std::endl;
    } 
    auto end = std::chrono::system_clock::now();

    // output FPS
    std::chrono::duration<float> elapsed_seconds = end - start;    
    float t = elapsed_seconds.count();
    std::cout << "Total time elapsed: " << t << " seconds." << std::endl;
    std::cout << "Average framerate: " << ps->frames / t << " frames per second." << std::endl;

    cleanup(phi_global, psi, device_phi);
}
