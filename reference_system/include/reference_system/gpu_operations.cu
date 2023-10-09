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

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float *C)
{
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

void di_gemm(gemm_operator* op)
{
     cudaError_t cuda_ret;
    float *A_d, *B_d, *C_d;

    cuda_ret = cudaMalloc((void **)&A_d, op->A_sz * sizeof(float));
    if (cuda_ret != cudaSuccess)
    {
        printf("Unable to allocate device memory for A\n");
        exit(EXIT_FAILURE);
    }

    cuda_ret = cudaMalloc((void **)&B_d, op->B_sz * sizeof(float));
    if (cuda_ret != cudaSuccess)
    {
        printf("Unable to allocate device memory for B\n");
        exit(EXIT_FAILURE);
    }

    cuda_ret = cudaMalloc((void **)&C_d, op->C_sz * sizeof(float));
    if (cuda_ret != cudaSuccess)
    {
        printf("Unable to allocate device memory for C\n");
        exit(EXIT_FAILURE);
    }

    cuda_ret = cudaMemcpy(A_d, op->A_h, op->A_sz * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess)
    {
        printf("Unable to copy memory to the device\n");
        exit(EXIT_FAILURE);
    }

    cuda_ret = cudaMemcpy(B_d, op->B_h, op->B_sz * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess)
    {
        printf("Unable to copy memory to the device\n");
        exit(EXIT_FAILURE);
    }

    cuda_ret = cudaMemset(C_d, 0, op->C_sz * sizeof(float));
    if (cuda_ret != cudaSuccess)
    {
        printf("Unable to copy memory to the device");
        exit(EXIT_FAILURE);
    }

    basicSgemm('N', 'N', op->matArow, op->matBcol, op->matBrow, 1.0f,
               A_d, op->matArow, B_d, op->matBrow, 0.0f, C_d, op->matBrow);

    cuda_ret = cudaMemcpy(op->C_h, C_d, op->C_sz * sizeof(float), cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess)
    {
        printf("Unable to copy memory to host");
        exit(EXIT_FAILURE);
    }
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}