#include "fusion.h"

#include "min_params.h"
#include <iostream>
#include <chrono>
#include <unistd.h>
#include "CImg.h"

using namespace cimg_library;

fusion_t::fusion_t(){
    
}

fusion_t::~fusion_t(){
    delete canon;
}

sdf_t 
fusion_t::get_sdf(std::string filename){
    std::cout << "loading depth map: " << filename << std::endl; 

    CImg<unsigned short> image(filename.c_str());

    std::vector<int> * ds = new std::vector<int>(image.width() * image.height());
    for (int x = 0; x < image.width(); x++){
        
        for (int y = 0; y < image.height(); y++){
            int d = *image.data(x, y, 0, 0);
            ds->push_back(d);
        }
    }

    return sdf_t(depths, image.width(), image.height());
}

void
fusion_t::load_filenames(std::vector<std::string> * fns, int frames){
    for (int i = 0; i < frames; i++){
        std::string padding = i < 10 ? "0" : "";
        if (i < 100){
            padding = "0" + padding;
        }
        fns->push_back("../data/umbrella/depth/frame-000" + padding + std::to_string(i) + ".depth.png");
    }
}

void 
fusion_t::update(bool is_rigid, sdf_t sdf){
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
fusion_t::initialise(sdf_t sdf){

}

void
fusion_t::fusion(){
    // load filenames
    std::vector<std::string> filenames;
    load_filenames(&filenames, ps->frames);

    // initalise deform field and canonical sdf
    sdf_t initial = get_sdf(filenames[0]);
    initialise(initial);

    // perform main fusion
    auto start = std::chrono::system_clock::now();
    for (int i = 1; i < filenames.size(); i++){
        std::cout << "Frame number: " << i << std::endl;     

        sdf_t sdf = get_sdf(filenames[i]); 
        update(true, sdf);
	update(false, sdf);

        std::cout << std::endl;
    } 
    auto end = std::chrono::system_clock::now();

    // output FPS
    std::chrono::duration<float> elapsed_seconds = end - start;    
    float t = elapsed_seconds.count();
    std::cout << "Total time elapsed: " << t << " seconds." << std::endl;
    std::cout << "Average framerate: " << ps->frames / t << " frames per second." << std::endl;
}
