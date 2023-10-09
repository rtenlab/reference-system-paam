#ifndef CUDACC
#define CUDACC
#endif

//#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
//#include <device_functions.h>
#include <cuda_runtime_api.h>
#define TILE_SIZE 16
#define BLOCK_SIZE 512

#include "gpu_operations.hpp"

#define CUDA_CHECK_ERRORS(result)                    \
    {                                                \
        checkCudaErrors(result, __FILE__, __LINE__); \
    }
inline void checkCudaErrors(cudaError_t result, const char *filename, int line_number)
{
    if (result != cudaSuccess)
    {
        throw std::runtime_error(
            "CUDA Error: " + std::string(cudaGetErrorString(result)) +
            " (error code: " + std::to_string(result) + ") at " +
            std::string(filename) + " in line " + std::to_string(line_number));
    }
}

__global__ void shared_hist_kernel(unsigned int *input, unsigned int *bins,
                                   unsigned int num_elements, unsigned int num_bins)
{

    extern __shared__ unsigned int shared_hist[];

    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;
    unsigned int i = threadIdx.x;
    while (i < num_bins)
    {
        shared_hist[i] = 0;
        i += blockDim.x;
    }
    i = threadIdx.x;
    __syncthreads();

    while (tid < num_elements)
    {
        atomicAdd(&shared_hist[input[tid] % num_bins], 1);
        tid += stride;
    }
    __syncthreads();

    while (i < num_bins)
    {
        atomicAdd(&bins[i], shared_hist[i]);
        i += blockDim.x;
    }
}

__global__ void global_hist_kernel(unsigned int *input, unsigned int *bins, unsigned int num_elements, unsigned int num_bins)
{

    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;

    while (tid < num_elements)
    {
        atomicAdd(&bins[input[tid] % num_bins], 1);
        tid += stride;
    }
}

void histogram(unsigned int *input, unsigned int *bins, unsigned int num_elements,
               unsigned int num_bins)
{
    dim3 block, grid;
    block.x = BLOCK_SIZE;
    grid.x = (num_elements - 1) / BLOCK_SIZE + 1;

    if (num_bins * sizeof(unsigned int) <= 49152)
    {
        // printf("\nUsing shared memory\n");
        shared_hist_kernel<<<grid, block, num_bins * sizeof(unsigned int)>>>(input, bins, num_elements, num_bins);
    }
    else
    {
        // printf("\nUsing global memory\n");
        global_hist_kernel<<<grid, block>>>(input, bins, num_elements, num_bins);
    }
}

void run_hist(unsigned int* in_h, unsigned int* bins_h, unsigned int num_elements, unsigned int num_bins)
{
    std::string home_dir = std::getenv("HOME");

    cudaError_t cuda_ret;
    unsigned int *in_d;
    unsigned int *bins_d;
    // Allocate device variables ----------------------------------------------
#ifdef EX_TIME_DEBUG
    gettimeofday(&ex_ctime, NULL);
    log_time("hist_bench", &ex_ctime, home_dir);
#endif
    cuda_ret = cudaMalloc((void **)&in_d, num_elements * sizeof(unsigned int));
    if (cuda_ret != cudaSuccess)
    {
        printf("Unable to allocate device memory\n");
        exit(EXIT_FAILURE);
    }
    cuda_ret = cudaMalloc((void **)&bins_d, num_bins * sizeof(unsigned int));
    if (cuda_ret != cudaSuccess)
    {
        printf("Unable to allocate device memory\n");
        exit(EXIT_FAILURE);
    }
    
    cuda_ret = cudaMemcpy(in_d, in_h, num_elements * sizeof(unsigned int),
                          cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess)
    {
        printf("Unable to copy memory to the device\n");
        exit(EXIT_FAILURE);
    }
    cuda_ret = cudaMemset(bins_d, 0, num_bins * sizeof(unsigned int));
    if (cuda_ret != cudaSuccess)
    {
        printf("Unable to set device memory\n");
        exit(EXIT_FAILURE);
    }
    
    histogram(in_d, bins_d, num_elements, num_bins);
    cuda_ret = cudaMemcpy(bins_h, bins_d, num_bins * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess)
    {
        printf("Unable to copy memory to host\n");
        exit(EXIT_FAILURE);
    }
    
#ifdef EX_TIME_DEBUG
    gettimeofday(&ex_ctime, NULL);
    log_time("hist_bench", &ex_ctime, home_dir);
#endif
    cudaFree(in_d);
    cudaFree(bins_d);
}

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float *C)
{

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     ********************************************************************/
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    int row = block_y * blockDim.y + thread_y;
    int col = block_x * blockDim.x + thread_x;

    float out = 0;

    for (int i = 0; i < (n - 1) / TILE_SIZE + 1; i++)
    {
        if (row < m && i * TILE_SIZE + thread_x < n)
        {
            shared_A[thread_y][thread_x] = A[row * n + i * TILE_SIZE + thread_x];
        }
        else
        {
            shared_A[thread_y][thread_x] = 0.0;
        }

        if (i * TILE_SIZE + thread_y < n && col < k)
        {
            shared_B[thread_y][thread_x] = B[(i * TILE_SIZE + thread_y) * k + col];
        }
        else
        {
            shared_B[thread_y][thread_x] = 0.0;
        }
        __syncthreads();
        for (int j = 0; j < TILE_SIZE; j++)
        {
            out += shared_A[thread_y][j] * shared_B[j][thread_x];
        }
        __syncthreads();
    }
    if (row < m && col < k)
    {
        C[row * k + col] = out;
    }
}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    dim3 block, grid;
    if ((transa != 'N') && (transa != 'n'))
    {
        printf("unsupported value of 'transa'\n");
        return;
    }

    if ((transb != 'N') && (transb != 'n'))
    {
        printf("unsupported value of 'transb'\n");
        return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10))
    {
        printf("unsupported value of alpha\n");
        return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10))
    {
        printf("unsupported value of beta\n");
        return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------
    block.x = TILE_SIZE;
    block.y = TILE_SIZE;
    grid.x = (n - 1) / TILE_SIZE + 1;
    grid.y = (m - 1) / TILE_SIZE + 1;
    // Invoke CUDA kernel -----------------------------------------------------
    mysgemm<<<grid, block, 0>>>(m, k, n, A, B, C);
}

void gemm_operator::gemm_wrapper(void);
{
     cudaError_t cuda_ret;
    float *A_d, *B_d, *C_d;

    cuda_ret = cudaMalloc((void **)&A_d, A_sz * sizeof(float));
    if (cuda_ret != cudaSuccess)
    {
        printf("Unable to allocate device memory for A\n");
        exit(EXIT_FAILURE);
    }

    cuda_ret = cudaMalloc((void **)&B_d, B_sz * sizeof(float));
    if (cuda_ret != cudaSuccess)
    {
        printf("Unable to allocate device memory for B\n");
        exit(EXIT_FAILURE);
    }

    cuda_ret = cudaMalloc((void **)&C_d, C_sz * sizeof(float));
    if (cuda_ret != cudaSuccess)
    {
        printf("Unable to allocate device memory for C\n");
        exit(EXIT_FAILURE);
    }

    cuda_ret = cudaMemcpy(A_d, A_h, A_sz * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess)
    {
        printf("Unable to copy memory to the device\n");
        exit(EXIT_FAILURE);
    }

    cuda_ret = cudaMemcpy(B_d, B_h, B_sz * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess)
    {
        printf("Unable to copy memory to the device\n");
        exit(EXIT_FAILURE);
    }

    cuda_ret = cudaMemset(C_d, 0, C_sz * sizeof(float));
    if (cuda_ret != cudaSuccess)
    {
        printf("Unable to copy memory to the device");
        exit(EXIT_FAILURE);
    }

    basicSgemm('N', 'N', matArow, matBcol, matBrow, 1.0f,
               A_d, matArow, B_d, matBrow, 0.0f, C_d, matBrow);

    cuda_ret = cudaMemcpy(C_h, C_d, C_sz * sizeof(float), cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess)
    {
        printf("Unable to copy memory to host");
        exit(EXIT_FAILURE);
    }
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}


void run_sgemm(float *A_h, float *B_h, float *C_h, size_t A_sz, size_t B_sz, size_t C_sz, unsigned int matArow, unsigned int matAcol, unsigned int matBrow, unsigned int matBcol)
{
    cudaError_t cuda_ret;
    float *A_d, *B_d, *C_d;

    cuda_ret = cudaMalloc((void **)&A_d, A_sz * sizeof(float));
    if (cuda_ret != cudaSuccess)
    {
        printf("Unable to allocate device memory for A\n");
        exit(EXIT_FAILURE);
    }

    cuda_ret = cudaMalloc((void **)&B_d, B_sz * sizeof(float));
    if (cuda_ret != cudaSuccess)
    {
        printf("Unable to allocate device memory for B\n");
        exit(EXIT_FAILURE);
    }

    cuda_ret = cudaMalloc((void **)&C_d, C_sz * sizeof(float));
    if (cuda_ret != cudaSuccess)
    {
        printf("Unable to allocate device memory for C\n");
        exit(EXIT_FAILURE);
    }

    cuda_ret = cudaMemcpy(A_d, A_h, A_sz * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess)
    {
        printf("Unable to copy memory to the device\n");
        exit(EXIT_FAILURE);
    }

    cuda_ret = cudaMemcpy(B_d, B_h, B_sz * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess)
    {
        printf("Unable to copy memory to the device\n");
        exit(EXIT_FAILURE);
    }

    cuda_ret = cudaMemset(C_d, 0, C_sz * sizeof(float));
    if (cuda_ret != cudaSuccess)
    {
        printf("Unable to copy memory to the device");
        exit(EXIT_FAILURE);
    }

    basicSgemm('N', 'N', matArow, matBcol, matBrow, 1.0f,
               A_d, matArow, B_d, matBrow, 0.0f, C_d, matBrow);

    cuda_ret = cudaMemcpy(C_h, C_d, C_sz * sizeof(float), cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess)
    {
        printf("Unable to copy memory to host");
        exit(EXIT_FAILURE);
    }
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}
