#include <cuda.h>
#include "cutil_math.h"

/*
   Device code
 */

__device__ float voxel_length = 8.0f;
__device__ float eta = 0.1f;
__device__ float delta = voxel_length * 10;
__device__ float omega_k = 0.5f;
__device__ float omega_s = 0.2f;
__device__ float gamma = 0.1f;
__device__ float epsilon = 0.00001f;
__device__ float threshold = 0.1f;
__device__ float sdf_eta = 100.0f;

__device__ int3
get_global_id(){
    //TODO
}

__device__ float
canon_distance(float2 * phi_global, int3 v, int3 dim){
    if (
        v.x < 0 || v.y < 0 || v.z < 0 ||
        v.x >= dim.x || v.y >= dim.y || v.z >= dim.z
    ){
        return 1.0f;
    } else {
        float2 pair = phi_global[v.x + v.y * dim.x + v.z * dim.x * dim.y];
        return pair.x / pair.y;
    }
}

__device__ float
phi_true(float * phi, int3 v, int3 dim){
    // TODO: add deformation, and interpolation

    int x = v.x;
    int y = v.y;
   
    if (x < 0 || y < 0 || x >= dim.x || y >= dim.y){
        return delta; 
    } else {
        return phi[x + y * dim.x] - v.z;
    } 
}

__device__ float
distance(float * phi, float3 * psi, int3 v, int3 dim){
    // divide by delta
    float result = phi_true(phi, v, dim) / delta;
    
    // clamp to range [-1..1]
    if (result > 1.0f){
	return 1.0f;
    } else if (result < -1.0f){
	return -1.0f;
    } else {
	return result;
    }
}

__device__ float3
distance_gradient(float * phi, float3 * psi, int3 v, int3 dim){
    int d = 1;
    return make_float3(
        ( 
            distance(phi, psi, v + make_int3(l, 0, 0), dim) - 
            distance(phi, psi, v - make_int3(l, 0, 0), dim)
        ) / (2 * d),
        
        (
            distance(phi, psi, v + make_int3(0, l, 0), dim) - 
            distance(phi, psi, v - make_int3(0, l, 0), dim)
        ) / (2 * d),
        
        (
            distance(phi, psi, v + make_int3(0, 0, l), dim) - 
            distance(phi, psi, v - make_int3(0, 0, l), dim)
        ) / (2 * d)
    );
}

__device__ void
sample_phi(float * phi, float2 ** phi_global, int3 v, int3 dim){
    float phi_t = phi_true(phi, v, dim);
    if (phi_t > -sdf_eta){
        int index = (v.x + v.y * dim.x + v.z * dim.x * dim.y);
        phi_global[index] += make_float2(distance(phi, psi, v, dim), 1.0f);
    }
}

__device__ float3
data_energy(float * phi, float3 * psi, float2 * phi_global, int3 dim, float3 grad){
    return grad * (distance(phi, psi, v, dim) - canon_distance(phi_global, v, dim));
}

__device__ float3
killing_energy(float3 * psi, int3 v, int3 dim){
    float3 result = make_float3(0.0f);
    for (int i = 0; i < 9; i++){

    }
    return result * 2.0f;
}

__device__ float3
level_set_energy(float * phi, float3 * psi, int3 v, int3 dim, float3 grad){
    float s = (length(grad) - 1) / (length(grad) + epsilon);
    
    float3 h;
    h.x = grad.x;
    h.y = grad.y;
    h.z = grad.z;
    return h * s;
}

__device__ float3
energy(float * phi, float * phi_global, float3 * psi, int3 v, dim){
    // pre-calculate grad, since used by both data and level set
    float3 grad = distance_gradient(phi, v, dim);
    return
        data_energy(phi, phi_global, v, dim, grad) +
        killing_energy(psi, v, dim) * omega_k + 
        level_set_energy(phi, psi, v, dim, grad) * omega_s;
}

__global__ void
init_kernel(float * phi, float * phi_global, int3 dim){
    int3 id = get_global_id();
    if (
        id.x < 0 || id.y < 0 || id.z < 0 ||
        id.x >= dim.x || id.y >= dim.y || id.z >= dim.z 
    ){
        return;
    }
    
    sample_phi(phi, phi_global, id, dim); 
}

__global__ void
rigid_kernel(float * phi, float * phi_global, float3 * psi, int3 dim){
    //TODO: sync

    int3 id = get_global_id();
    if (
        id.x < 2 || id.y < 2 || id.z < 2 ||
        id.x >= dim.x - 2 || id.y >= dim.y - 2 || id.z >= dim.z - 2
    ){
        return;
    }

    bool quit = false;
    while (!quit){ 
        // pre-calculate grad to comply with signature which is adapted from non-rigid term
        float3 grad = distance_gradient(phi, id, dim);
        float3 e = data_energy(phi, psi, phi_global, id, dim);
        
        if (length(e) <= threshold){
            quit = true;
        }
         
        int index = id.x + id.y * dim.x + id.z * dim.x * dim.y;
        psi[index] -= e * eta;        
    }

    sample_phi(phi, phi_global, id, dim);
}

__global__ void
nonrigid_kernel(float * phi, float * phi_global, float3 * psi, int3 dim){
    // TODO: sync

    int3 id = get_global_id();
    if (
        id.x < 2 || id.y < 2 || id.z < 2 ||
        id.x >= dim.x - 2 || id.y >= dim.y - 2 || id.z >= dim.z - 2
    ){
        return;
    }
     
    bool quit = false;
    while (!quit){
        float3 e = energy(phi, phi_global, id, dim);
        
        if (length(e) <= threshold){
            quit = true;
        }
         
        int index = id.x + id.y * dim.x + id.z * dim.x * dim.y;
        psi[index] -= e * eta;        
    }

    sample_phi(phi, phi_global, id, dim);
}

/*
   Host code
 */
int grid_size = 512; // TODO
int block_size = 512; // TODO

void
initialise(float * phi, float ** device_phi, float2 ** phi_global, float3 ** psi, int3 dim){
    int img_size   = sizeof(float)  * dim.x * dim.y;
    int phi_g_size = sizeof(float2) * dim.x * dim.y * dim.z;
    int psi_size   = sizeof(float3) * dim.x * dim.y * dim.z;

    // allocate memory
    cudaMalloc(phi_global, phi_g_size);
    cudaMalloc(psi, psi_size);
    cudaMalloc(device_phi, img_size);

    // sample sdf into canon sdf
    cudaMemcpy(*device_phi, phi, img_size, cudaMemcpyHostToDevice);
    init_kernel<<<grid_size, block_size>>>(*device_phi, *phi_global, dim);

    // set deform field to zero
    cudaMemset(*psi_size, 0, psi_size);
}

void 
update_rigid(float * phi, float * device_phi, float * phi_global, float3 * psi, int3 dim){
    cudaMemcpy(device_phi, phi, sizeof(float) * width * height, cudaMemcpyHostToDevice);
    rigid_kernel<<<grid_size, block_size>>>(device_phi, phi_global, psi, dim);
}

void 
update_nonrigid(float * phi, float * device_phi, float * phi_global, float3 * psi, int3 dim){
    cudaMemcpy(device_phi, phi, sizeof(float) * width * height, cudaMemcpyHostToDevice);
    nonrigid_kernel<<<grid_size, block_size>>>(device_phi, phi_global, psi, dim);
}

void
get_canon(float * host_phi_global, float * device_phi_global, int size){
    cudaMemcpy(host_phi_global, device_phi_global, size, cudaMemcpyDeviceToHost);
}
         
void cleanup(float * phi_global, float * u, float * v, float * w, float * device_phi){
    cudaFree(phi_global);
    cudaFree(device_phi);
    cudaFree(psi);
}
