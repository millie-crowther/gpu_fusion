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
    return make_int3(blockIdx.x, blockIdx.y, blockIdx.z);
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

__device__ float3
deformation(float3 * psi, int3 p, int3 dim){
    if (
        p.x < 0 || p.y < 0 || p.z < 0 ||
        p.x >= dim.x || p.y >= dim.y || p.z >= dim.z
    ){
        return make_float3(0);
    } else {
        int index = p.x + p.y * dim.x + p.z * dim.x * dim.y;
        return psi[index];
    }
}

__device__ float
phi_true(float * phi, float3 * psi, int3 x, int3 dim){
    // TODO: interpolation
    float3 x1 = make_float3(x.x, x.y, x.z) + make_float3(0.5f);
    float3 v = x1 + deformation(psi, x1, dim); 

    int x = v.x;
    int y = v.y;
    int z = v.z
   
    if (x < 0 || y < 0 || x >= dim.x || y >= dim.y){
        return delta; 
    } else {
        return phi[x + y * dim.x] - z;
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
    return make_float3(
        ( distance(phi, psi, v + make_int3(1, 0, 0), dim) - 
          distance(phi, psi, v - make_int3(1, 0, 0), dim)
        ) / 2.0f,
        
        ( distance(phi, psi, v + make_int3(0, 1, 0), dim) - 
          distance(phi, psi, v - make_int3(0, 1, 0), dim)
        ) / 2.0f,
        
        ( distance(phi, psi, v + make_int3(0, 0, 1), dim) - 
          distance(phi, psi, v - make_int3(0, 0, 1), dim)
        ) / 2.0f
    );
}

__device__ void
sample_phi(float * phi, float2 ** phi_global, int3 v, int3 dim){
    if (phi_true(phi, v, dim) > -sdf_eta){
        int index = (v.x + v.y * dim.x + v.z * dim.x * dim.y);
        phi_global[index] += make_float2(distance(phi, psi, v, dim), 1.0f);
    }
}

__device__ float3
data_energy(float * phi, float3 * psi, float2 * phi_global, int3 dim, float3 grad){
    return grad * (distance(phi, psi, v, dim) - canon_distance(phi_global, v, dim));
}

__device__ float
hessian_uvw(float3 * psi, int3 v, int3 dim, int x, int y, int elem){
    int3 axes[3];
    axes[0] = make_int3(1, 0, 0);
    axes[1] = make_int3(0, 1, 0);
    axes[2] = make_int3(0, 0, 1);

    float3 p1 = deformation(psi, v + axes[x] + axes[y], dim); 
    float3 p2 = deformation(psi, v + axes[x] - axes[y], dim); 
    float3 p3 = deformation(psi, v - axes[x] + axes[y], dim); 
    float3 p4 = deformation(psi, v - axes[x] - axes[y], dim); 

    float p = elem == 0 ? p1.x : (elem == 1 ? p1.y : p1.z);
    float q = elem == 0 ? p2.x : (elem == 1 ? p2.y : p2.z);
    float r = elem == 0 ? p3.x : (elem == 1 ? p3.y : p3.z);
    float s = elem == 0 ? p4.x : (elem == 1 ? p4.y : p4.z);
    
    float t = (p - q) / (2 * voxel_length);
    float u = (r - s) / (2 * voxel_length);
    return (t - u) / (2 * voxel_length); 
}

__device__ float
jacobian(float3 * psi, int3 v, int3 dim, int x, int y){
    int3 axes[3];
    axes[0] = make_int3(1, 0, 0);
    axes[1] = make_int3(0, 1, 0);
    axes[2] = make_int3(0, 0, 1);

    float3 p = deformation(psi, v + axes[x], dim); 
    float3 q = deformation(psi, v - axes[x], dim); 
    float3 result = (p - q) / (2.0f * voxel_length);
    if (y == 0){
        return result.x;
    } else if (y == 1){
        return result.y;
    } else {
        return result.z;
    }
}

__device__ float3
killing_energy(float3 * psi, int3 v, int3 dim){
    float3 result = make_float3(0.0f);
    for (int i = 0; i < 9; i++){
         int jx = i / 3;
         int jy = i % 3;
         float j = jacobian(psi, v, dim, jy, jx) + gamma * jacobian(psi, v, dim, kx, jy);
         
         int hx = i % 3;
         int elem = i / 3;
         result += make_float3(
             hessian_uvw(psi, v, dim, hx, 0, elem) * j, 
             hessian_uvw(psi, v, dim, hx, 1, elem) * j, 
             hessian_uvw(psi, v, dim, hx, 2, elem) * j
         );
    }
    return result * 2.0f;
}

__device__ float
hessian_d(float * phi, float3 * psi, int3 v, int3 dim, int x, int y){
    int3 axes[3];
    axes[0] = make_int3(1, 0, 0);
    axes[1] = make_int3(0, 1, 0);
    axes[2] = make_int3(0, 0, 1);

    float p = distance(phi, psi, v + axes[x] + axes[y], dim);
    float q = distance(phi, psi, v + axes[x] - axes[y], dim);
    float r = distance(phi, psi, v - axes[x] + axes[y], dim);
    float s = distance(phi, psi, v - axes[x] - axes[y], dim);

    float t = (p - q) / 2.0f;
    float u = (r - s) / 2.0f;
    return (t - u) / 2.0f; 
}

__device__ float3
level_set_energy(float * phi, float3 * psi, int3 v, int3 dim, float3 grad){
    float s = (length(grad) - 1) / (length(grad) + epsilon);
    
    float3 h;
    h.x = hessian_d(phi, psi, v, dim, 0, 0) * grad.x +
          hessian_d(phi, psi, v, dim, 1, 0) * grad.y +
          hessian_d(phi, psi, v, dim, 2, 0) * grad.z +
    h.y = hessian_d(phi, psi, v, dim, 0, 1) * grad.x +
          hessian_d(phi, psi, v, dim, 1, 1) * grad.y +
          hessian_d(phi, psi, v, dim, 2, 1) * grad.z +
    h.z = hessian_d(phi, psi, v, dim, 0, 2) * grad.x +
          hessian_d(phi, psi, v, dim, 1, 2) * grad.y +
          hessian_d(phi, psi, v, dim, 2, 2) * grad.z +
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
dim3 grid_size = (1, 1, 1); 
dim3 block_size = (80, 60, 200);

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
