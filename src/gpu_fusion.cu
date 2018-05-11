#include <cuda.h>

__global__ 
void
init_kernel(){

}

void
initialise(
    float * phi,  
    float ** phi_global,
    float ** u, float ** v, float ** w,
    int width, int height, int depth
){

}

void 
update_rigid(
    float * phi, 
    float * phi_global, 
    float * u, float * v, float * w, 
    int width, int height, int depth
){

}

void 
update_nonrigid(
    float * phi, 
    float * phi_global, 
    float * u, float * v, float * w,
    int width, int height, int depth
    ){

}
         
void cleanup(float * phi_global, float * u, float * v, float * w){
    cudaFree(phi_global);
    cudaFree(u);
    cudaFree(v);
    cudaFree(w);
}

