#include <cuda.h>
#include "cutil_math.hpp"
#include "fusion.hpp"

/*
   Device code
 */

__constant__ float voxel_length = 8.0f;
__constant__ float eta = 0.1f;
__constant__ float delta = 80; // voxel length times 10
__constant__ float omega_k = 0.5f;
__constant__ float omega_s = 0.2f;
__constant__ float killing_gamma = 0.1f;
__constant__ float epsilon = 0.00001f;
__constant__ float threshold = 0.1f;
__constant__ float sdf_eta = 100.0f;

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
	if (pair.y == 0.0f){
	    return 1.0f;
	} else {
            return pair.x / pair.y;
	}
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
        return psi[p.x + p.y * dim.x + p.z * dim.x * dim.y];
    }
}

__device__ int
phi_true(int * phi, float3 * psi, int3 v_grid, int3 dim){
    // TODO: interpolate
    float3 v = make_float3(v_grid.x, v_grid.y, v_grid.z) + make_float3(0.5f);
    v += deformation(psi, v_grid, dim); 

    int x = v.x;
    int y = v.y;
    int z = v.z;
   
    if (x < 0 || y < 0 || x >= dim.x || y >= dim.y){
        return delta; 
    } else {
        return phi[x + y * dim.x] - z;
    } 
}

__device__ float
distance(int * phi, float3 * psi, int3 v, int3 dim){
    // divide by delta
    float phi_t = phi_true(phi, psi, v, dim);
    float result = phi_t / delta;
    
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
distance_gradient(int * phi, float3 * psi, int3 v, int3 dim){
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
sample_phi(int * phi, float3 * psi, float2 * phi_global, int3 v, int3 dim){
    if (phi_true(phi, psi, v, dim) > -sdf_eta){
        int index = (v.x + v.y * dim.x + v.z * dim.x * dim.y);
        phi_global[index] += make_float2(distance(phi, psi, v, dim), 1.0f);
    }
}

__device__ float3
data_energy(int * phi, float3 * psi, float2 * phi_global, int3 v, int3 dim, float3 grad){
    return grad * (distance(phi, psi, v, dim) - canon_distance(phi_global, v, dim));
}

__device__ float
get_elem(float3 v, int i){
    return i == 0 ? v.x : (i == 1 ? v.y : v.z);
}

__device__ float
hessian_uvw(float3 * psi, int3 v, int3 dim, int x, int y, int elem){
    int3 axes[3];
    axes[0] = make_int3(1, 0, 0);
    axes[1] = make_int3(0, 1, 0);
    axes[2] = make_int3(0, 0, 1);

    // sample deformation field
    float3 p1 = deformation(psi, v + axes[x] + axes[y], dim); 
    float3 p2 = deformation(psi, v + axes[x] - axes[y], dim); 
    float3 p3 = deformation(psi, v - axes[x] + axes[y], dim); 
    float3 p4 = deformation(psi, v - axes[x] - axes[y], dim); 

    // select u, v, w element of deformation as appropriate
    float p = get_elem(p1, elem);
    float q = get_elem(p2, elem);
    float r = get_elem(p3, elem);
    float s = get_elem(p4, elem);
    
    // differentiate
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
         int x = i / 3;
         int y = i % 3;
         float j = jacobian(psi, v, dim, y, x) + killing_gamma * jacobian(psi, v, dim, x, y);
         
         result += make_float3(
             hessian_uvw(psi, v, dim, y, 0, x) * j, 
             hessian_uvw(psi, v, dim, y, 1, x) * j, 
             hessian_uvw(psi, v, dim, y, 2, x) * j
         );
    }
    return result * 2.0f;
}

__device__ float
hessian_d(int * phi, float3 * psi, int3 v, int3 dim, int x, int y){
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
level_set_energy(int * phi, float3 * psi, int3 v, int3 dim, float3 grad){
    float s = (length(grad) - 1) / (length(grad) + epsilon);
    
    float3 h;
    h.x = hessian_d(phi, psi, v, dim, 0, 0) * grad.x +
          hessian_d(phi, psi, v, dim, 1, 0) * grad.y +
          hessian_d(phi, psi, v, dim, 2, 0) * grad.z;
    h.y = hessian_d(phi, psi, v, dim, 0, 1) * grad.x +
          hessian_d(phi, psi, v, dim, 1, 1) * grad.y +
          hessian_d(phi, psi, v, dim, 2, 1) * grad.z;
    h.z = hessian_d(phi, psi, v, dim, 0, 2) * grad.x +
          hessian_d(phi, psi, v, dim, 1, 2) * grad.y +
          hessian_d(phi, psi, v, dim, 2, 2) * grad.z;
    return h * s;
}

__device__ float3
energy(int * phi, float2 * phi_global, float3 * psi, int3 v, int3 dim){
    // pre-calculate grad, since used by both data and level set
    float3 grad = distance_gradient(phi, psi, v, dim);
    return
        data_energy(phi, psi, phi_global, v, dim, grad) +
        killing_energy(psi, v, dim) * omega_k + 
        level_set_energy(phi, psi, v, dim, grad) * omega_s;
}

__global__ void
init_kernel(int * phi, float3 * psi, float2 * phi_global, int3 dim){
    int3 id = get_global_id();
    if (
        id.x < 0 || id.y < 0 || id.z < 0 ||
        id.x >= dim.x || id.y >= dim.y || id.z >= dim.z 
    ){
        return;
    }
   
    phi_global[id.x + id.y * dim.x + id.z * dim.x * dim.y] = make_float2(0.0f); 
    sample_phi(phi, psi, phi_global, id, dim); 
}

__global__ void
estimate_psi_kernel(int * phi, float2 * phi_global, float3 * psi, int3 dim){
    int3 id = get_global_id();
    if (
        id.x < 2 || id.y < 2 || id.z < 2 ||
        id.x >= dim.x - 2 || id.y >= dim.y - 2 || id.z >= dim.z - 2
    ){
        return;
    }
    
 
    bool quit = false;
    while (!quit){
        float3 e = energy(phi, phi_global, psi, id, dim);

        if (length(e) <= threshold){
            quit = true;
        }
         
        int index = id.x + id.y * dim.x + id.z * dim.x * dim.y;
        psi[index] -= e * eta;        
    }

    sample_phi(phi, psi, phi_global, id, dim);
}

/*
   Host code
 */
dim3 grid_size = (1, 1, 1); 
dim3 block_size = (80, 60, 200);

void
initialise(int * phi, int ** device_phi, float2 ** phi_global, float3 ** psi, int3 dim){
    int img_size   = sizeof(int)  * dim.x * dim.y;
    int phi_g_size = sizeof(float2) * dim.x * dim.y * dim.z;
    int psi_size   = sizeof(float3) * dim.x * dim.y * dim.z;

    // allocate memory
    cudaMalloc(phi_global, phi_g_size);
    cudaMalloc(psi, psi_size);
    cudaMalloc(device_phi, img_size);

    // sample sdf into canon sdf
    cudaMemcpy(*device_phi, phi, img_size, cudaMemcpyHostToDevice);
    init_kernel<<<grid_size, block_size>>>(*device_phi, *psi, *phi_global, dim);

    cudaError e = cudaGetLastError();
    if (e != cudaSuccess){
        printf("error: %d\n", e);
    }

    // set deform field to zero
    cudaMemset(*psi, 0, psi_size);
}

void 
estimate_psi(int * phi, int * device_phi, float2 * phi_global, float3 * psi, int3 dim){
    int img_size = sizeof(int) * dim.x * dim.y;
    cudaMemcpy(device_phi, phi, img_size, cudaMemcpyHostToDevice);
    estimate_psi_kernel<<<grid_size, block_size>>>(device_phi, phi_global, psi, dim);
}

void
get_canon(float2 * host_phi_global, float2 * device_phi_global, int size){
    cudaMemcpy(host_phi_global, device_phi_global, size, cudaMemcpyDeviceToHost);
}
         
void 
cleanup(float2 * phi_global, float3 * psi, int * device_phi){
    cudaFree(phi_global);
    cudaFree(device_phi);
    cudaFree(psi);
}
