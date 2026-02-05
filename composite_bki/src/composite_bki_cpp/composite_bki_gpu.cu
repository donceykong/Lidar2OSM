/*
 * GPU-Accelerated functions for Composite BKI using CUDA
 * Optional module - requires NVIDIA GPU and CUDA toolkit
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iostream>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

// Kernel: Compute pairwise squared distances
__global__ void computeDistancesKernel(
    const float* points_x, const float* points_y, const float* points_z,
    int* neighbor_counts,
    int* neighbor_indices,
    int n_points,
    float radius_sq,
    int max_neighbors) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) return;
    
    float px = points_x[i];
    float py = points_y[i];
    float pz = points_z[i];
    
    int count = 0;
    int base_idx = i * max_neighbors;
    
    for (int j = 0; j < n_points && count < max_neighbors; j++) {
        float dx = px - points_x[j];
        float dy = py - points_y[j];
        float dz = pz - points_z[j];
        float dist_sq = dx*dx + dy*dy + dz*dz;
        
        if (dist_sq <= radius_sq) {
            neighbor_indices[base_idx + count] = j;
            count++;
        }
    }
    
    neighbor_counts[i] = count;
}

// Kernel: Compute semantic weights in parallel
__global__ void computeSemanticWeightsKernel(
    const float* points_x, const float* points_y,
    const float* osm_distances,  // Pre-computed OSM distances
    float* k_sem_all,
    int n_points,
    float delta) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) return;
    
    // Simplified semantic weight computation
    // In full implementation, this would use OSM prior
    float dist = osm_distances[i];
    k_sem_all[i] = 1.0f / (1.0f + expf(dist / delta));
}

// Kernel: Aggregate neighbor weights for BKI update
__global__ void aggregateWeightsKernel(
    const int* neighbor_indices,
    const int* neighbor_counts,
    const float* k_sem_all,
    const int* dense_labels,
    float* alpha_star,
    int n_points,
    int n_classes,
    int max_neighbors,
    float alpha_0) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) return;
    
    int base_idx = i * max_neighbors;
    int count = neighbor_counts[i];
    int alpha_base = i * n_classes;
    
    // Initialize alpha_star
    for (int c = 0; c < n_classes; c++) {
        alpha_star[alpha_base + c] = alpha_0;
    }
    
    // Accumulate neighbor weights
    for (int j = 0; j < count; j++) {
        int neighbor_idx = neighbor_indices[base_idx + j];
        int dense_class = dense_labels[neighbor_idx];
        
        if (dense_class >= 0 && dense_class < n_classes) {
            float weight = k_sem_all[neighbor_idx];
            atomicAdd(&alpha_star[alpha_base + dense_class], weight);
        }
    }
}

extern "C" {

// GPU neighbor search
bool gpu_find_neighbors(
    const float* h_points_x, const float* h_points_y, const float* h_points_z,
    int n_points,
    float radius,
    int max_neighbors,
    int* h_neighbor_counts,
    int* h_neighbor_indices) {
    
    float radius_sq = radius * radius;
    
    // Allocate device memory
    float *d_points_x, *d_points_y, *d_points_z;
    int *d_neighbor_counts, *d_neighbor_indices;
    
    size_t points_size = n_points * sizeof(float);
    size_t counts_size = n_points * sizeof(int);
    size_t indices_size = n_points * max_neighbors * sizeof(int);
    
    CUDA_CHECK(cudaMalloc(&d_points_x, points_size));
    CUDA_CHECK(cudaMalloc(&d_points_y, points_size));
    CUDA_CHECK(cudaMalloc(&d_points_z, points_size));
    CUDA_CHECK(cudaMalloc(&d_neighbor_counts, counts_size));
    CUDA_CHECK(cudaMalloc(&d_neighbor_indices, indices_size));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_points_x, h_points_x, points_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_points_y, h_points_y, points_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_points_z, h_points_z, points_size, cudaMemcpyHostToDevice));
    
    // Launch kernel
    int block_size = 256;
    int grid_size = (n_points + block_size - 1) / block_size;
    
    computeDistancesKernel<<<grid_size, block_size>>>(
        d_points_x, d_points_y, d_points_z,
        d_neighbor_counts, d_neighbor_indices,
        n_points, radius_sq, max_neighbors);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_neighbor_counts, d_neighbor_counts, counts_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_neighbor_indices, d_neighbor_indices, indices_size, cudaMemcpyDeviceToHost));
    
    // Free device memory
    cudaFree(d_points_x);
    cudaFree(d_points_y);
    cudaFree(d_points_z);
    cudaFree(d_neighbor_counts);
    cudaFree(d_neighbor_indices);
    
    return true;
}

// Check if CUDA is available
bool cuda_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

// Get GPU info
void print_gpu_info() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    
    if (device_count == 0) {
        std::cout << "No CUDA-capable GPU found" << std::endl;
        return;
    }
    
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        std::cout << "GPU " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
    }
}

} // extern "C"
