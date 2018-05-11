#include <cuda.h>

/*
   Device code
 */

__device__ float voxel_length = 20.0f;

__device__ int
get_global_id(){
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void
init_kernel(
    float * phi, 
    float * phi_global, 
    int width, int height, int depth
){
    // calculate unique id
    int id = get_global_id();
    if (id >= width * height * depth){
	return;
    }

    // get coordinates from id
    int z  = id / (width * height);
    int xy = id - z * width * height;
    int y  = xy / width;
    int x  = xy % width;

    // sample sdf
    phi_global[id] = phi[x + y * width] - z * voxel_length;
}

__global__ void
rigid_kernel(
    float * phi, 
    float * phi_global, 
    float * u, float * v, float * w, 
    int width, int height, int depth
){
    // calculate unique id
    int id = get_global_id();
    if (id >= width * height * depth){
	return;
    }

    // get coordinates from id
    int z  = id / (width * height);
    int xy = id - z * width * height;
    int y  = xy / width;
    int x  = xy % width;
   
    // 
}

__global__ void
nonrigid_kernel(
    float * phi, 
    float * phi_global, 
    float * u, float * v, float * w, 
    int width, int height, int depth
){
    // calculate unique id
    int id = get_global_id();
    if (id >= width * height * depth){
	return;
    }

    // get coordinates from id
    int z  = id / (width * height);
    int xy = id - z * width * height;
    int y  = xy / width;
    int x  = xy % width;

}


/*
   Host code
 */
int grid_size = 512;
int block_size = 512;

void
initialise(
    float * phi,
    float ** device_phi,  
    float ** phi_global,
    float ** u, float ** v, float ** w,
    int width, int height, int depth,
){
    int vol_size = sizeof(float) * width * height * depth;
    int img_size = sizeof(float) * width * height;

    // allocate memory
    cudaMalloc(phi_global, vol_size);
    cudaMalloc(u, vol_size);
    cudaMalloc(v, vol_size);
    cudaMalloc(w, vol_size);

    // sample sdf into canon sdf
    cudaMalloc(device_phi, img_size);
    cudaMemcpy(*device_phi, phi, img_size, cudaMemcpyHostToDevice);
    init_kernel<<<grid_size, block_size>>>(
        *device_phi, *phi_global, width, height, depth
    );

    // set deform field to zero
    cudaMemset(*u, 0, vol_size);
    cudaMemset(*v, 0, vol_size);
    cudaMemset(*w, 0, vol_size);
}

void 
update_rigid(
    float * phi,
    float * device_phi, 
    float * phi_global, 
    float * u, float * v, float * w, 
    int width, int height, int depth,
){
    cudaMemcpy(device_phi, phi, sizeof(float) * width * height, cudaMemcpyHostToDevice);

    rigid_kernel<<<grid_size, block_size>>>(
        device_phi, phi_global, u, v, w, width, height, depth
    );
}

void 
update_nonrigid(
    float * phi, 
    float * device_phi, 
    float * phi_global, 
    float * u, float * v, float * w,
    int width, int height, int depth,
){
    cudaMemcpy(device_phi, phi, sizeof(float) * width * height, cudaMemcpyHostToDevice);

    nonrigid_kernel<<<grid_size, block_size>>>(
	device_phi, phi_global, u, v, w, width, height, depth
    );
}

void
get_canon(float * host_phi_global, float * device_phi_global, int size){
    cudaMemcpy(host_phi_global, device_phi_global, size, cudaMemcpyDeviceToHost);
}
         
void cleanup(float * phi_global, float * u, float * v, float * w, float * device_phi){
    cudaFree(phi_global);
    cudaFree(u);
    cudaFree(v);
    cudaFree(w);
    cudaFree(device_phi);
}

