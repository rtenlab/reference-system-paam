#ifndef CUDACC
#define CUDACC
#endif

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <cuda_device_runtime_api.h>
#include <cuda_profiler_api.h>
#include <thrust/count.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <iostream>
#include "gpu_operations.h"
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <eigen3/Eigen/QR>
#include <eigen3/Eigen/Core>
#include <stdio.h>
#include <sys/time.h>
using namespace std;
using namespace cv;

#define TILE_SIZE 32
#define BLOCK_SIZE 128

#include "gpu_operations.h"
void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc, int bucket);
__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float *C);
cudaStream_t streams[6];
cudaMemPool_t memPools[6];

int numcudastreams = 0;
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
void destroy_cuda_streams()
{

    for (int i = 0; i < numcudastreams; i++)
    {
        CUDA_CHECK_ERRORS(cudaStreamDestroy(streams[i]));
        CUDA_CHECK_ERRORS(cudaMemPoolDestroy(memPools[i]));
    }
    std::printf("Cuda Streams Destroyed\n");

    // delete [] streams;
    //   delete &numcudastreams;
}
int init_cuda_streams()
{
    // cudaSetDeviceFlags(cudaDeviceScheduleSpin);
    // cudaSetDeviceFlags(cudaDeviceDefaultStreamPerThread);
    cudaSetDeviceFlags(cudaDeviceScheduleSpin);
    int leastPriority, greatestPriority;
    int thresholdVal = ULLONG_MAX;
    CUDA_CHECK_ERRORS(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    std::printf("Minimum Priority of Stream: %i \nMaximum Priority of Stream: %i \n", leastPriority, greatestPriority);
    int j = 0;
    int device = 0;
    cudaMemPoolProps poolProps = {};
    poolProps.allocType = cudaMemAllocationTypePinned;
    poolProps.location.id = device;
    poolProps.location.type = cudaMemLocationTypeDevice;
    for (int i = leastPriority; i >= greatestPriority; --i)
    {
        CUDA_CHECK_ERRORS(cudaStreamCreateWithPriority(&streams[j], cudaStreamNonBlocking, i));
        CUDA_CHECK_ERRORS(cudaMemPoolCreate(&memPools[j], &poolProps));
        CUDA_CHECK_ERRORS(cudaMemPoolSetAttribute(memPools[j], cudaMemPoolAttrReleaseThreshold, (void *)&thresholdVal));
        j++;
    }
    numcudastreams = j;
    return j;
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
               unsigned int num_bins, int bucket)
{
    dim3 block, grid;
    block.x = BLOCK_SIZE;
    grid.x = (num_elements - 1) / BLOCK_SIZE + 1;

    if (num_bins * sizeof(unsigned int) <= 49152)
    {
        // printf("\nUsing shared memory\n");
        shared_hist_kernel<<<grid, block, num_bins * sizeof(unsigned int), streams[bucket]>>>(input, bins, num_elements, num_bins);
    }
    else
    {
        // printf("\nUsing global memory\n");
        global_hist_kernel<<<grid, block, 0, streams[bucket]>>>(input, bins, num_elements, num_bins);
    }
}
unsigned int *run_remote_hist(int in_h, int num_elements, int num_bins, int bucket, bool *in_use)
{

    printf("Running Remote HIST Kernel on Stream %i\n", bucket);
    cudaError_t cuda_ret;
    unsigned int *bins_d, *in_d;
    unsigned int *bins_h;
    // unsigned int num_elements = 1000000, num_bins = 4096;
    bins_h = (unsigned int *)malloc(num_bins * sizeof(unsigned int));
    cuda_ret = cudaMallocAsync((void **)&in_d, num_elements * sizeof(unsigned int), streams[bucket]);
    cuda_ret = cudaMallocAsync((void **)&bins_d, num_bins * sizeof(unsigned int), streams[bucket]);
    cuda_ret = cudaMemcpyAsync(in_d, &(in_h), num_elements * sizeof(unsigned int),
                               cudaMemcpyHostToDevice, streams[bucket]);
    cuda_ret = cudaMemset(bins_d, 0, num_bins * sizeof(unsigned int));
    histogram(in_d, bins_d, num_elements, num_bins, bucket);
    cuda_ret = cudaMemcpyAsync(bins_h, bins_d, num_bins * sizeof(unsigned int),
                               cudaMemcpyDeviceToHost, streams[bucket]);
    cudaStreamSynchronize(streams[bucket]);
    *in_use = false;
    cudaFreeAsync(in_d, streams[bucket]);
    cudaFreeAsync(bins_d, streams[bucket]);
    printf("Finshed Remote HIST Kernel on Stream %i\n", bucket);
    return bins_h;
}

void run_hist(struct hist_struct *shared_request, int bucket, bool *in_use)
{
    printf("Running HIST Kernel on Stream %i\n", bucket);
    pthread_mutex_lock(&shared_request->pthread_mutex);
    cudaError_t cuda_ret;
    unsigned int *bins_d, *in_d;
    //unsigned int num_elements = 1000000, num_bins = 4096;
    unsigned int num_elements = shared_request->request.num_elements;
    unsigned int num_bins = shared_request->request.nbins;
    cuda_ret = cudaMallocAsync((void **)&in_d, num_elements * sizeof(unsigned int), streams[bucket]);
    if (cuda_ret != cudaSuccess)
    {
        printf("Unable to allocate device memory");
        exit(EXIT_FAILURE);
    }
    cuda_ret = cudaMallocAsync((void **)&bins_d, num_bins * sizeof(unsigned int), streams[bucket]);
    if (cuda_ret != cudaSuccess)
    {
        printf("Unable to allocate device memory");
        exit(EXIT_FAILURE);
    }
    // cudaStreamSynchronize(streams[bucket]);
    cuda_ret = cudaMemcpyAsync(in_d, &(shared_request->request.in_h), num_elements * sizeof(unsigned int),
                               cudaMemcpyHostToDevice, streams[bucket]);
    if (cuda_ret != cudaSuccess)
    {
        printf("Unable to copy memory to the device");
        exit(EXIT_FAILURE);
    }
    cuda_ret = cudaMemset(bins_d, 0, num_bins * sizeof(unsigned int));
    if (cuda_ret != cudaSuccess)
    {
        printf("Unable to set device memory");
        exit(EXIT_FAILURE);
    }
    // cudaStreamSynchronize(streams[bucket]);
    histogram(in_d, bins_d, num_elements, num_bins, bucket);
    // cuda_ret = cudaStreamSynchronize(streams[bucket]);
    if (cuda_ret != cudaSuccess)
    {
        printf("Unable to launch/execute kernel");
        exit(EXIT_FAILURE);
    }
    cuda_ret = cudaMemcpyAsync(&shared_request->response.bins_h, bins_d, num_bins * sizeof(unsigned int),
                               cudaMemcpyDeviceToHost, streams[bucket]);
    if (cuda_ret != cudaSuccess)
    {
        printf("Unable to copy memory to host");
        exit(EXIT_FAILURE);
    }
    cudaStreamSynchronize(streams[bucket]);
    shared_request->ready = true;
    *in_use = false;
    pthread_cond_signal(&shared_request->pthread_cv);
    pthread_mutex_unlock(&shared_request->pthread_mutex);
    cudaFreeAsync(in_d, streams[bucket]);
    cudaFreeAsync(bins_d, streams[bucket]);
    printf("Finshed HIST Kernel on Stream %i\n", bucket);
}

void run_sgemm(struct gemm_struct *shared_request, int bucket, bool *in_use)
{
    int prio = 0;
    //uint64_t thresholdVal = ULONG_MAX;
    // it does not contain any active suballocations.
    // checkCudaErrors(cudaDeviceGetDefaultMemPool(&memPool, dev));

    // cudaStreamGetPriority(streams[bucket], &prio);
    // printf("Running GEMM Kernel on Stream %i with priority %i\n", bucket, prio);
    struct gemm_request *active_request = &(shared_request->request);
    size_t A_sz = active_request->matArow * active_request->matAcol;
    size_t B_sz = active_request->matBrow * active_request->matBcol;
    size_t C_sz = active_request->matArow * active_request->matBcol;
    float *A_d, *B_d, *C_d;
    // float *output = (float*) malloc( sizeof(float)*C_sz );
    CUDA_CHECK_ERRORS(cudaMallocFromPoolAsync((void **)&A_d, A_sz * sizeof(float), memPools[bucket], streams[bucket]));
    CUDA_CHECK_ERRORS(cudaMallocFromPoolAsync((void **)&B_d, B_sz * sizeof(float), memPools[bucket], streams[bucket]));
    CUDA_CHECK_ERRORS(cudaMallocFromPoolAsync((void **)&C_d, C_sz * sizeof(float), memPools[bucket], streams[bucket]));
    CUDA_CHECK_ERRORS(cudaMemcpyAsync(A_d, active_request->A_h, A_sz * sizeof(float), cudaMemcpyHostToDevice, streams[bucket]));
    CUDA_CHECK_ERRORS(cudaMemcpyAsync(B_d, active_request->B_h, B_sz * sizeof(float), cudaMemcpyHostToDevice, streams[bucket]));
    CUDA_CHECK_ERRORS(cudaMemsetAsync(C_d, 0, C_sz * sizeof(float), streams[bucket]));
    // cudaStreamSynchronize(streams[bucket]);
    basicSgemm('N', 'N', active_request->matArow, active_request->matBcol, active_request->matBrow, 1.0f, A_d, active_request->matArow, B_d, active_request->matBrow, 0.0f, C_d, active_request->matBrow, bucket);
    // CUDA_CHECK_ERRORS(cudaStreamSynchronize(streams[bucket]));
    CUDA_CHECK_ERRORS(cudaMemcpyAsync(&(shared_request->response.C_h), C_d, C_sz * sizeof(float), cudaMemcpyDeviceToHost, streams[bucket]));
    //&(shared_request->response.C_h) = output;
#ifdef OVERHEAD_DEBUG
    logger->set_start();
    // log_time("Waking Client");
    // flush_buffer();
#endif

    /*
     echo 100 > /sys/kernel/debug/tegra_mce/rt_window_us
     echo 20 > /sys/kernel/debug/tegra_mce/rt_fwd_progress_us
     echo 0x7f > /sys/kernel/debug/tegra_mce/rt_safe_mask
     echo -1 > /proc/sys/kernel/sched_rt_runtime_us

     */
    shared_request->ready = true; // notify here before cudaFreeAsync
    pthread_mutex_lock(&shared_request->pthread_mutex);
    pthread_cond_signal(&shared_request->pthread_cv);
    pthread_mutex_unlock(&shared_request->pthread_mutex);

    *in_use = false;
    cudaFreeAsync(A_d, streams[bucket]);
    cudaFreeAsync(B_d, streams[bucket]);
    cudaFreeAsync(C_d, streams[bucket]);
    // printf("Finshed GEMM Kernel on Stream %i\n", bucket);
}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc, int bucket)
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
    mysgemm<<<grid, block, 0, streams[bucket]>>>(m, k, n, A, B, C);
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
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.4;
const float CONFIDENCE_THRESHOLD = 0.4;

void yolo_detect(struct yolo_struct *shared_request, bool *in_use, cv::dnn::Net net, const std::vector<std::string> className)
{
    // void detect(cv::Mat &image, cv::dnn::Net &net, std::vector<Detection> &output, const std::vector<std::string> &className) {
    printf("Running YOLO Kernel default Stream \n");
    pthread_mutex_lock(&shared_request->pthread_mutex);

    // cv::Mat *image = &shared_request->request.frame;
    cv::Mat image(shared_request->request.rows, shared_request->request.cols, shared_request->request.type, shared_request->request.frame);
    int col = image.cols;
    int row = image.rows;
    int _max = MAX(col, row);
    cv::Mat input_image = cv::Mat::zeros(_max, _max, CV_8UC3);
    image.copyTo(input_image(cv::Rect(0, 0, col, row)));
    cv::Mat blob;
    cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    *in_use = false;

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    float *data = (float *)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i)
    {

        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD)
        {

            float *classes_scores = data + 5;
            cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > SCORE_THRESHOLD)
            {

                confidences.push_back(confidence);

                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }

        data += 85;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    shared_request->response.size = nms_result.size();
    for (int i = 0; i < nms_result.size() && i < 25; i++)
    {
        int idx = nms_result[i];
        // Detection result;
        // result.class_id = class_ids[idx];
        // result.confidence = confidences[idx];
        // result.box = boxes[idx];
        shared_request->response.detections[i].class_id = class_ids[idx];
        shared_request->response.detections[i].confidence = confidences[idx];
        shared_request->response.detections[i].box = boxes[idx];

        // output.push_back(result);
    }
    pthread_cond_signal(&shared_request->pthread_cv);
    pthread_mutex_unlock(&shared_request->pthread_mutex);
    printf("Running YOLO Kernel completed\n");
}

struct isNonZeroIndex
{
    __host__ __device__ bool operator()(const int &idx)
    {
        return (idx != -1);
    }
};

__global__ void kernel_find_indices(const uint8_t *input, int width, int height, int step, float *indicesx, float *indicesy)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        const int tidPixel = y * step + x;
        const int tidIndex = y * width + x;

        int value = input[tidPixel];
        if (value)
        {
            float X = float(x);
            float Y = float(y);
            indicesx[tidIndex] = X;
            indicesy[tidIndex] = Y;
        }
        else
        {
            indicesx[tidIndex] = -1;
            indicesy[tidIndex] = -1;
        }
    }
}

__global__ void processing_next(float *PointX_n, float *PointY_n, const float margin, float *left_n, float *right_n, const int N_n, float *LPoint_x, float *LPoint_y, float *RPoint_x, float *RPoint_y)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N_n)
    {
        float good_left_inds_n = ((PointX_n[i] > (left_n[0] * pow(float(PointY_n[i]), 2) + left_n[1] * PointY_n[i] + left_n[2] - margin)) & (PointX_n[i] < (left_n[0] * (pow(float(PointY_n[i]), 2)) + left_n[1] * PointY_n[i] + left_n[2] + margin)));
        float good_right_inds_n = ((PointX_n[i] > (right_n[0] * pow(float(PointY_n[i]), 2) + right_n[1] * PointY_n[i] + right_n[2] - margin)) & (PointX_n[i] < (right_n[0] * (pow(float(PointY_n[i]), 2)) + right_n[1] * PointY_n[i] + right_n[2] + margin)));

        if (good_left_inds_n != 0)
        {
            LPoint_x[i] = PointX_n[i];
            LPoint_y[i] = PointY_n[i];
        }
        else
        {
            LPoint_x[i] = -1;
            LPoint_y[i] = -1;
        }
        if (good_right_inds_n != 0)
        {
            RPoint_x[i] = PointX_n[i];
            RPoint_y[i] = PointY_n[i];
        }
        else
        {
            RPoint_x[i] = -1;
            RPoint_y[i] = -1;
        }
    }
}

void indices_point(cuda::GpuMat &src, thrust::device_vector<float> &outx, thrust::device_vector<float> &outy)
{
    int Array_Size = cuda::countNonZero(src);
    thrust::device_vector<float> Point_x(src.rows * src.step);
    thrust::device_vector<float> Point_y(src.rows * src.step);
    uint8_t *imgPtr;
    cudaMalloc((void **)&imgPtr, src.rows * src.step);
    cudaMemcpyAsync(imgPtr, src.ptr<uint8_t>(), src.rows * src.step, cudaMemcpyDeviceToDevice);
    dim3 block(16, 16);
    dim3 grid;
    grid.x = (src.cols + block.x - 1) / block.x;
    grid.y = (src.rows + block.y - 1) / block.y;
    kernel_find_indices<<<grid, block>>>(imgPtr, int(src.cols), int(src.rows), int(src.step), thrust::raw_pointer_cast(Point_x.data()), thrust::raw_pointer_cast(Point_y.data()));
    cudaDeviceSynchronize();
    thrust::copy_if(Point_x.begin(), Point_x.end(), outx.begin(), isNonZeroIndex());
    thrust::copy_if(Point_y.begin(), Point_y.end(), outy.begin(), isNonZeroIndex());
    cudaFree(imgPtr);
}

void getIndicesOfNonZeroPixels(cuda::GpuMat &src, vector<float> &output_hx, vector<float> &output_hy)
{

    int array_size = cuda::countNonZero(src);
    thrust::device_vector<float> Point_X(array_size);
    thrust::device_vector<float> Point_Y(array_size);
    indices_point(src, Point_X, Point_Y);
    output_hx.resize(array_size);
    output_hy.resize(array_size);
    thrust::copy(Point_X.begin(), Point_X.end(), output_hx.begin());
    thrust::copy(Point_Y.begin(), Point_Y.end(), output_hy.begin());
}

void getIndicesOfNonZeroPixelsnext(cuda::GpuMat &src, vector<float> &Loutput_hx, vector<float> &Loutput_hy, vector<float> &Routput_hx, vector<float> &Routput_hy, struct last_fit *last_fit)
{

    vector<float> polyright_out_n;
    vector<float> polyleft_out_n;

    polyright_out_n = last_fit->polyright_last;
    polyleft_out_n = last_fit->polyleft_last;
    size_t SIZE_T = 3 * sizeof(float);
    float *right_fit_last = (float *)malloc(SIZE_T);
    float *right_fit_last_d;
    cudaMalloc(&right_fit_last_d, SIZE_T);
    float *left_fit_last = (float *)malloc(SIZE_T);
    float *left_fit_last_d;
    cudaMalloc(&left_fit_last_d, SIZE_T);

    right_fit_last[2] = polyright_out_n[0];
    left_fit_last[2] = polyleft_out_n[0];
    right_fit_last[1] = polyright_out_n[1];
    left_fit_last[1] = polyleft_out_n[1];
    right_fit_last[0] = polyright_out_n[2];
    left_fit_last[0] = polyleft_out_n[2];

    cudaMemcpy(right_fit_last_d, right_fit_last, SIZE_T, cudaMemcpyHostToDevice);
    cudaMemcpy(left_fit_last_d, left_fit_last, SIZE_T, cudaMemcpyHostToDevice);

    const float margin = 10;
    const int Size_array = cuda::countNonZero(src);
    thrust::device_vector<float> Point_X(Size_array);
    thrust::device_vector<float> Point_Y(Size_array);
    indices_point(src, Point_X, Point_Y);

    float *arrayx = thrust::raw_pointer_cast(&Point_X[0]);
    float *arrayy = thrust::raw_pointer_cast(&Point_Y[0]);

    thrust::device_vector<float> LPoint_x(Size_array);
    thrust::device_vector<float> LPoint_y(Size_array);
    thrust::device_vector<float> RPoint_x(Size_array);
    thrust::device_vector<float> RPoint_y(Size_array);

    processing_next<<<Size_array, 1>>>(arrayx, arrayy, margin, left_fit_last_d, right_fit_last_d, Size_array, thrust::raw_pointer_cast(LPoint_x.data()),
                                       thrust::raw_pointer_cast(LPoint_y.data()), thrust::raw_pointer_cast(RPoint_x.data()), thrust::raw_pointer_cast(RPoint_y.data()));
    cudaDeviceSynchronize();

    int nonZeroCountL = int(thrust::count_if(LPoint_x.begin(), LPoint_x.end(), isNonZeroIndex()));
    int nonZeroCountR = int(thrust::count_if(RPoint_x.begin(), RPoint_x.end(), isNonZeroIndex()));

    thrust::device_vector<float> Loutx(nonZeroCountL);
    thrust::copy_if(LPoint_x.begin(), LPoint_x.end(), Loutx.begin(), isNonZeroIndex());
    Loutput_hx.resize(nonZeroCountL);
    thrust::copy(Loutx.begin(), Loutx.end(), Loutput_hx.begin());

    thrust::device_vector<float> Louty(nonZeroCountL);
    thrust::copy_if(LPoint_y.begin(), LPoint_y.end(), Louty.begin(), isNonZeroIndex());
    Loutput_hy.resize(nonZeroCountL);
    thrust::copy(Louty.begin(), Louty.end(), Loutput_hy.begin());

    thrust::device_vector<float> Routx(nonZeroCountR);
    thrust::copy_if(RPoint_x.begin(), RPoint_x.end(), Routx.begin(), isNonZeroIndex());
    Routput_hx.resize(nonZeroCountR);
    thrust::copy(Routx.begin(), Routx.end(), Routput_hx.begin());

    thrust::device_vector<float> Routy(nonZeroCountR);
    thrust::copy_if(RPoint_y.begin(), RPoint_y.end(), Routy.begin(), isNonZeroIndex());
    Routput_hy.resize(nonZeroCountR);
    thrust::copy(Routy.begin(), Routy.end(), Routput_hy.begin());

    cudaFree(right_fit_last_d);
    cudaFree(left_fit_last_d);
}

extern bool do_calib;

void calibration_on();
void getIndicesOfNonZeroPixels(cuda::GpuMat &src, vector<float> &output_hx, vector<float> &output_hy);
void getIndicesOfNonZeroPixelsnext(cuda::GpuMat &src, vector<float> &Loutput_hx, vector<float> &Loutput_hy, vector<float> &Routput_hx, vector<float> &Routput_hy, struct last_fit *last_fit);
cv::Mat LANEDETECTION::get_frame()
{
    cv::Mat frame;
    this->cap >> frame;
    return frame;
}
void lane_detect(struct lane_struct *shared_request, bool *in_use, int stream_id, struct last_fit *last_fit, LANEDETECTION *lanedetection)
{
    cv::Mat frame, cudaout_frame, frame_out;
    cv::cuda::GpuMat process_frame, process_frameout, resize_frame, process_framein;
    pthread_mutex_lock(&shared_request->pthread_mutex);
    // lanedetection->cap >> frame;
    frame = lanedetection->get_frame();
    if (frame.empty())
        return;

    process_framein.upload(frame);
    lanedetection->processinga_frame(process_framein, resize_frame, process_frame);
    process_frame.download(cudaout_frame);
    lanedetection->processingb_frame(cudaout_frame, resize_frame, process_frameout, last_fit);
    process_frameout.download(frame_out);
    memcpy(shared_request->response.frame, frame_out.data, frame_out.total() * frame_out.elemSize());
    shared_request->response.rows = frame_out.rows;
    shared_request->response.rows = frame_out.cols;
    shared_request->response.type = frame_out.type();
    shared_request->ready = true;
    *in_use = false;
    pthread_cond_signal(&shared_request->pthread_cv);
    pthread_mutex_unlock(&shared_request->pthread_mutex);

    imshow("FrameL", frame_out);
    if (waitKey(10) >= 0)
        return;
}

bool do_calib;
LANEDETECTION::LANEDETECTION()
{
    return;
}
void LANEDETECTION::init_lane_detection(std::string calibration_file)
{
    // cv::VideoCapture cap("/home/aamf/Documents/ros2_gpu_server/src/gpu_responder/openv.avi"); // rename
    if (!this->cap.isOpened())
    {
        cout << "Error opening video stream or file" << endl;
        return;
    }
    cv::Mat intrinsicn = cv::Mat(3, 3, CV_32FC1);
    cv::Mat distCoeffsn = cv::Mat(3, 3, CV_32FC1);
    cv::FileStorage fs2(calibration_file, cv::FileStorage::READ); // for my webcam
    fs2["intrinsic"] >> intrinsicn;
    fs2["distCoeffs"] >> distCoeffsn;
    cv::Mat mapc, mapd;
    cv::cuda::GpuMat gpu_mapc, gpu_mapd;
    initUndistortRectifyMap(intrinsicn, distCoeffsn, cv::Mat::eye(3, 3, CV_32FC1), intrinsicn, cv::Size(640, 360), CV_32FC1, mapc, mapd);
    gpu_mapc.upload(mapc);
    gpu_mapd.upload(mapd);
    // initUndistortRectifyMap(intrinsicn, distCoeffsn, cv::Mat::eye(3, 3, CV_32FC1), intrinsicn, cv::Size(640, 360), CV_32FC1, this->mapa, this->mapb);
    // this->gpu_mapa.upload(this->mapa);
    // this->gpu_mapb.upload(this->mapb);
}
LANEDETECTION::LANEDETECTION(std::string device)
{
    this->mapa = cv::Mat();
    this->mapa = cv::Mat();
    this->gpu_mapa = cv::cuda::GpuMat();
    this->gpu_mapa = cv::cuda::GpuMat();
    this->device = device;
    if (device == "/dev/video0")
    {
        this->cap.open("v4l2src device=/dev/video0 ! videoconvert ! appsink", cv::CAP_GSTREAMER);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        this->init_lane_detection("/home/aamf/Documents/ros2_gpu_server/src/gpu_responder/calibration_camera.yml");
    }
    else if (device == "/home/aamf/Documents/ros2_gpu_server/src/gpu_responder/openv.avi")
    {
        this->cap.open("/home/aamf/Documents/ros2_gpu_server/src/gpu_responder/openv.avi");
        this->init_lane_detection("/home/aamf/Documents/ros2_gpu_server/src/gpu_responder/calibration.yml");
    }
    if (!this->cap.isOpened())
    {
        printf("Capture failed to open\n");
    }
}
LANEDETECTION::LANEDETECTION(cv::VideoCapture *cap)
{
    this->device = cap->getBackendName();
    this->cap = *cap;
}
LANEDETECTION::~LANEDETECTION()
{
    this->cap.release();
}
void LANEDETECTION::binary_frame(cuda::GpuMat &src, cuda::GpuMat &dst)
{
    int threshold_value = 110;
    int max_value = 255;
    cuda::threshold(src, dst, threshold_value, max_value, THRESH_BINARY);
}

void calibration_on()
{
    if (do_calib == true)
    {
        int numBoards = 17;
        int numCornersHor = 9;
        int numCornersVer = 6;
        int numSquares = numCornersHor * numCornersVer;
        Size board_sz = Size(numCornersHor, numCornersVer);
        VideoCapture capture;
        capture.open("D:/camera_cal/%02d.jpg");

        vector<vector<Point3f>> object_points;
        vector<vector<Point2f>> image_points;

        vector<Point2f> corners;
        int successes = 0;
        Mat image;
        Mat gray_image;

        capture >> image;

        vector<Point3f> obj;
        for (int j = 0; j < numSquares; j++)
            obj.push_back(Point3f(float(j / numCornersHor), float(j % numCornersHor), 0.0f));

        while (successes < numBoards)
        {
            cvtColor(image, gray_image, COLOR_BGR2GRAY);

            bool found = findChessboardCorners(image, board_sz, corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FILTER_QUADS);

            if (found)
            {
                cornerSubPix(gray_image, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.1));
                drawChessboardCorners(gray_image, board_sz, corners, found);
            }

            capture >> image;
            int key = waitKey(1);

            if (found != 0)
            {
                image_points.push_back(corners);
                object_points.push_back(obj);
                cout << "Snap stored!" << endl;
                successes++;
                if (successes >= numBoards)
                    break;
            }
        }

        VideoCapture capt;
        capt.open("D:/camera_cal/%02d.jpg");
        capt >> image;
        Mat intrinsic = Mat(3, 3, CV_32FC1);
        Mat distCoeffs;
        vector<Mat> rvecs;
        vector<Mat> tvecs;

        intrinsic.ptr<float>(0)[0] = 1;
        intrinsic.ptr<float>(1)[1] = 1;

        calibrateCamera(object_points, image_points, image.size(), intrinsic, distCoeffs, rvecs, tvecs);

        cv::FileStorage fs("calibration.yml", cv::FileStorage::WRITE);
        fs << "intrinsic" << intrinsic;
        fs << "distCoeffs" << distCoeffs;
        fs.release();
        do_calib = false;
    }
}

void LANEDETECTION::erode_dilate(cuda::GpuMat &src, cuda::GpuMat &dst)
{
    cuda::GpuMat erode_out, dilate_out;
    int noise = 3;
    int dilate_const = 1;
    Mat element_erosion = getStructuringElement(MORPH_RECT, Size(noise * 2 + 1, noise * 2 + 1));
    Ptr<cuda::Filter> erode = cuda::createMorphologyFilter(cv::MORPH_ERODE, src.type(), element_erosion);
    erode->apply(src, erode_out);
    Mat element_dilation = getStructuringElement(MORPH_RECT, Size(dilate_const * 2 + 1, dilate_const * 2 + 1));
    Ptr<cuda::Filter> dilateFilter = cuda::createMorphologyFilter(MORPH_DILATE, src.type(), element_dilation);
    dilateFilter->apply(erode_out, dst);
}
void LANEDETECTION::first_frame(cuda::GpuMat &src, vector<float> &polyright, vector<float> &polyleft)
{

    cuda::GpuMat hist;
    float margin = 60;
    float minpix = 50;
    float windows_no = 8;
    float src_rows = float(src.rows);
    float windows_height = src_rows / windows_no;
    vector<float> main_hx;
    vector<float> main_hy;
    getIndicesOfNonZeroPixels(src, main_hx, main_hy);
    vector<float> leftx;
    vector<float> lefty;
    vector<float> rightx;
    vector<float> righty;
    cuda::reduce(src(Rect(0, src.rows / 2, src.cols, src.rows / 2)), hist, 0, cv::ReduceTypes::REDUCE_SUM, CV_32S);
    int midpoint = (int(hist.cols / 2));
    Point max_locL, max_locR;
    cuda::minMaxLoc(hist(Rect(50, 0, midpoint, hist.rows)), NULL, NULL, NULL, &max_locL);
    cuda::minMaxLoc(hist(Rect(midpoint, 0, midpoint, hist.rows)), NULL, NULL, NULL, &max_locR);
    float leftxbase = float(int(max_locL.x + 50));
    float rightxbase = float(int(max_locR.x + midpoint));

    for (int window = 1; window <= windows_no; window++)
    {
        vector<float> leftx_t;
        vector<float> lefty_t;
        vector<float> rightx_t;
        vector<float> righty_t;
        float win_y_low = float(int(src_rows - (window + 1) * windows_height));
        float win_y_high = float(int(src_rows - window * windows_height));
        float winxleft_low = float(int(leftxbase - margin));
        float winxleft_high = float(int(leftxbase + margin));
        float winxright_low = float(int(rightxbase - margin));
        float winxright_high = float(int(rightxbase + margin));
        float mean_left = 0;
        float mean_right = 0;

        for (auto idx = 0; idx < main_hy.size(); idx++)
        {
            float good_left_inds = float((float(main_hy[idx]) >= win_y_low) & (float(main_hy[idx]) < win_y_high) & (float(main_hx[idx]) >= winxleft_low) & (float(main_hx[idx]) < winxleft_high));
            float good_right_inds = float((float(main_hy[idx]) >= win_y_low) & (float(main_hy[idx]) < win_y_high) & (float(main_hx[idx]) >= winxright_low) & (float(main_hx[idx]) < winxright_high));
            if (good_left_inds != 0.f)
            {
                leftx_t.push_back(float(main_hx[idx]));
                lefty_t.push_back(float(main_hy[idx]));
                mean_left = mean_left + float(main_hx[idx]);
            }
            if (good_right_inds != 0.f)
            {
                rightx_t.push_back(float(main_hx[idx]));
                righty_t.push_back(float(main_hy[idx]));
                mean_right = mean_right + float(main_hx[idx]);
            }
        }
        if (leftx_t.size() > minpix)
        {
            leftxbase = float(int(mean_left / leftx_t.size()));
        }
        if (rightx_t.size() > minpix)
        {
            rightxbase = float(int(mean_right / rightx_t.size()));
        }

        leftx.insert(leftx.end(), leftx_t.begin(), leftx_t.end());
        lefty.insert(lefty.end(), lefty_t.begin(), lefty_t.end());
        rightx.insert(rightx.end(), rightx_t.begin(), rightx_t.end());
        righty.insert(righty.end(), righty_t.begin(), righty_t.end());
    }

    polyright = LANEDETECTION::polyfiteigen(righty, rightx, 2);
    polyleft = LANEDETECTION::polyfiteigen(lefty, leftx, 2);
}
void LANEDETECTION::gray_frame(cuda::GpuMat &src, cuda::GpuMat &dst)
{
    cuda::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
}
void LANEDETECTION::hsv_frame(cuda::GpuMat &src, cuda::GpuMat &dst)

{
    cuda::GpuMat hsv_frame, temp;
    cuda::GpuMat channels_device[3];
    cuda::GpuMat channels_device_dest[3];
    cuda::cvtColor(src, hsv_frame, COLOR_BGR2HSV);
    cuda::split(hsv_frame, channels_device);
    cuda::threshold(channels_device[0], channels_device_dest[0], 0, 100, THRESH_BINARY);
    cuda::threshold(channels_device[2], channels_device_dest[1], 210, 255, THRESH_BINARY);
    cuda::threshold(channels_device[2], channels_device_dest[2], 200, 255, THRESH_BINARY);
    cuda::merge(channels_device_dest, 3, temp);
    cuda::cvtColor(temp, dst, COLOR_HSV2BGR);
}

void LANEDETECTION::wrap_frame(cuda::GpuMat &src, cuda::GpuMat &dst, Point2f *src_points, Point2f *dst_points)
{

    Mat trans_points = getPerspectiveTransform(src_points, dst_points);
    cuda::warpPerspective(src, dst, trans_points, src.size(), cv::INTER_LINEAR);
}

void LANEDETECTION::video_frame(cuda::GpuMat &src, vector<float> &polyleft_out, vector<float> &polyright_out, struct last_fit *last_fit)
{

    if ((last_fit->polyleft_last.size() == 0) && (0 == last_fit->polyright_last.size()))
    {
        LANEDETECTION::first_frame(src, polyright_out, polyleft_out);
        last_fit->polyright_last = polyright_out;
        last_fit->polyleft_last = polyleft_out;
    }
    else
    {
        LANEDETECTION::nxt_frame(src, polyright_out, polyleft_out, last_fit);
    }
}

vector<float> LANEDETECTION::polyfiteigen(const vector<float> &xv, const vector<float> &yv, int order)
{
    Eigen::initParallel();
    Eigen::MatrixXf A = Eigen::MatrixXf::Ones(xv.size(), order + 1);
    Eigen::VectorXf yv_mapped = Eigen::VectorXf::Map(&yv.front(), yv.size());
    Eigen::VectorXf xv_mapped = Eigen::VectorXf::Map(&xv.front(), xv.size());
    Eigen::VectorXf result;

    assert(xv.size() == yv.size());
    assert(xv.size() >= order + 1);

    for (int j = 1; j < order + 1; j++)
    {
        A.col(j) = A.col(j - 1).cwiseProduct(xv_mapped);
    }

    result = A.householderQr().solve(yv_mapped);
    vector<float> coeff;
    coeff.resize(order + 1);
    for (size_t i = 0; i < order + 1; i++)
        coeff[i] = result[i];

    return coeff;
}

vector<float> LANEDETECTION::polyvaleigen(const vector<float> &oCoeff,
                                          const vector<float> &oX)
{
    int nCount = int(oX.size());
    int nDegree = int(oCoeff.size());
    vector<float> oY(nCount);

    for (int i = 0; i < nCount; i++)
    {
        float nY = 0;
        float nXT = 1;
        float nX = oX[i];
        for (int j = 0; j < nDegree; j++)
        {
            nY += oCoeff[j] * nXT;
            nXT *= nX;
        }
        oY[i] = nY;
    }

    return oY;
}
void LANEDETECTION::nxt_frame(cuda::GpuMat &src, vector<float> &polyright_n, vector<float> &polyleft_n, struct last_fit *last_fit)
{

    vector<float> leftx;
    vector<float> lefty;
    vector<float> rightx;
    vector<float> righty;

    getIndicesOfNonZeroPixelsnext(src, leftx, lefty, rightx, righty, last_fit);

    LANEDETECTION Polyfit;

    polyright_n = Polyfit.polyfiteigen(righty, rightx, 2);
    polyleft_n = Polyfit.polyfiteigen(lefty, leftx, 2);
}

void LANEDETECTION::sobel_frame(cuda::GpuMat &src, cuda::GpuMat &dst)
{

    cuda::GpuMat sobelx, sobely, adwsobelx, adwsobely, gray_framea;
    LANEDETECTION::gray_frame(src, gray_framea);
    Ptr<cuda::Filter> filter = cv::cuda::createSobelFilter(gray_framea.type(), CV_16S, 1, 0, 3, 1, BORDER_DEFAULT);
    filter->apply(gray_framea, sobelx);
    cuda::abs(sobelx, sobelx);
    sobelx.convertTo(dst, CV_8UC1);
}

float center_point(float x1, float x2)
{
    return (x1 + x2) / 2.f;
}

float distance(float x1, float x2)
{
    return ((sqrt(pow(float(x1 - x2), 2.f))));
}

vector<float> LinearSpacedArray(float a, float b, size_t N)
{
    float h = (b - a) / static_cast<double>(N - 1);
    std::vector<float> xs(N);
    std::vector<float>::iterator x;
    float val;
    for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h)
    {
        *x = val;
    }
    return xs;
}

void LANEDETECTION::curvature_sanity_check(vector<float> &polyleft_in, vector<float> &polyright_in, vector<int> &Leftx, vector<int> &rightx, vector<int> &main_y, struct last_fit *last_fit)
{
    float xm_per_pix = 3.7f / 350.0f;
    float ym_per_pix = 30.0f / 360.0f;
    vector<float> Plot_ys(360);
    std::iota(begin(Plot_ys), end(Plot_ys), 0.f);

    vector<float> Leftx_out_x;
    vector<float> rightx_out_x;
    vector<float> Plot_y = LinearSpacedArray(0.f, 20.f, 10);
    Leftx_out_x = LANEDETECTION::polyvaleigen(polyleft_in, Plot_y);
    rightx_out_x = LANEDETECTION::polyvaleigen(polyright_in, Plot_y);
    float Lmean = float(accumulate(Leftx_out_x.begin(), Leftx_out_x.end(), 0.0) / Leftx_out_x.size());
    float Rmean = float(accumulate(rightx_out_x.begin(), rightx_out_x.end(), 0.0) / rightx_out_x.size());
    float delta_lines = (Rmean - Lmean);

    float L_0 = 2 * polyleft_in[2] * 180 + polyleft_in[1];
    float R_0 = 2 * polyright_in[2] * 180 + polyright_in[1];
    float delta_slope_mid = abs(R_0 - L_0);

    float L_1 = 2 * polyleft_in[2] * 360 + polyleft_in[1];
    float R_1 = 2 * polyright_in[2] * 360 + polyright_in[1];
    float delta_slope_bottom = abs(L_1 - R_1);

    float L_2 = 2 * polyleft_in[2] + polyleft_in[1];
    float R_2 = 2 * polyright_in[2] + polyright_in[1];
    float delta_slope_top = abs(L_2 - R_2);

    vector<float> Leftx_sanity;
    vector<float> rightx_sanity;

    if (((delta_slope_top <= 0.9) && (delta_slope_bottom <= 0.9) && (delta_slope_mid <= 0.9)) && ((delta_lines > 75)))
    {
        last_fit->polyleft_last = polyleft_in;
        last_fit->polyright_last = polyright_in;
        Leftx_sanity = polyleft_in;
        rightx_sanity = polyright_in;
    }
    else
    {
        Leftx_sanity = last_fit->polyleft_last;
        rightx_sanity = last_fit->polyright_last;
    }

    vector<float> Leftx_out;
    vector<float> rightx_out;

    LANEDETECTION polyfitpolyval;

    Leftx_out = polyfitpolyval.polyvaleigen(Leftx_sanity, Plot_ys);

    rightx_out = polyfitpolyval.polyvaleigen(rightx_sanity, Plot_ys);

    vector<float> Leftx_out_m = Leftx_out;
    vector<float> rightx_out_m = rightx_out;
    vector<float> Plot_ysm = Plot_ys;
    float first_element_L = Leftx_out[359];
    float first_element_R = rightx_out[359];

    float center_x = center_point(first_element_L, first_element_R);
    float center_ix = 320;
    LANEDETECTION::center_dist = (distance(center_x, center_ix) * xm_per_pix);
    // cout << center_dist <<endl;

    transform(Leftx_out.begin(), Leftx_out.end(), Leftx_out.begin(),
              bind1st(std::multiplies<float>(), xm_per_pix));

    transform(rightx_out.begin(), rightx_out.end(), rightx_out.begin(),
              bind1st(std::multiplies<float>(), xm_per_pix));

    transform(Plot_ys.begin(), Plot_ys.end(), Plot_ys.begin(),
              bind1st(std::multiplies<float>(), ym_per_pix));

    vector<float> left_fit_cr;
    vector<float> right_fit_cr;

    left_fit_cr = polyfitpolyval.polyfiteigen(Plot_ys, Leftx_out, 2);
    LANEDETECTION::left_curverad = float((1 + pow(pow((2 * left_fit_cr[2] * 359 * ym_per_pix + left_fit_cr[1]), 2), 1.5)) / abs(2 * left_fit_cr[2]));

    right_fit_cr = polyfitpolyval.polyfiteigen(Plot_ys, rightx_out, 2);
    LANEDETECTION::right_curverad = float((1 + pow(pow((2 * right_fit_cr[2] * 359 * ym_per_pix + right_fit_cr[1]), 2), 1.5)) / abs(2 * right_fit_cr[2]));

    rightx.insert(rightx.end(), rightx_out_m.begin(), rightx_out_m.end());
    Leftx.insert(Leftx.end(), Leftx_out_m.begin(), Leftx_out_m.end());
    main_y.insert(main_y.end(), Plot_ysm.begin(), Plot_ysm.end());
}
void LANEDETECTION::resize_frame(cuda::GpuMat &src, cuda::GpuMat &dst, int resize_height, int resize_width)
{
    cuda::resize(src, dst, Size(resize_width, resize_height));
}
void LANEDETECTION::processinga_frame(cuda::GpuMat &src, cuda::GpuMat &resize, cuda::GpuMat &dst)
{

    cv::Point2f src_points[4];
    cv::Point2f dst_points[4];
    int resize_height = 360;
    int resize_width = 640;

    src_points[0] = cv::Point2f(290, 230);
    src_points[1] = cv::Point2f(350, 230);
    src_points[2] = cv::Point2f(520, 340);
    src_points[3] = cv::Point2f(130, 340);

    dst_points[0] = cv::Point2f(130, 0);
    dst_points[1] = cv::Point2f(520, 0);
    dst_points[2] = cv::Point2f(520, 360);
    dst_points[3] = cv::Point2f(130, 360);
    Mat frame, cudaout_frame, MergeFrameOut, cudaout_framet;
    cuda::GpuMat resize_framea, gray_framea, binary_framea, birdview_framea, hsv_framea, threshold_frame, sobel_frameout, gpu_undisort;

    LANEDETECTION::resize_frame(src, resize_framea, resize_height, resize_width);
    cuda::remap(resize_framea, gpu_undisort, gpu_mapa, gpu_mapb, cv::INTER_LINEAR);
    resize = gpu_undisort;
    LANEDETECTION::wrap_frame(resize_framea, birdview_framea, src_points, dst_points);
    LANEDETECTION::sobel_frame(birdview_framea, sobel_frameout);
    LANEDETECTION::hsv_frame(birdview_framea, hsv_framea);
    LANEDETECTION::gray_frame(hsv_framea, gray_framea);
    LANEDETECTION::binary_frame(gray_framea, binary_framea);
    cuda::addWeighted(binary_framea, 0.9, sobel_frameout, 0.1, -1, dst);
}

void left_point(vector<int> &left_X, vector<int> &main_Y, vector<Point2i> &Pointleft)
{
    int m = int(main_Y.size());
    for (int r = 0; r < m; r++)
    {
        Pointleft.push_back(Point2i(left_X[r], main_Y[r]));
    }
}

void right_point(vector<int> &right_X, vector<int> &main_Y, vector<Point2i> &Pointright)
{

    int m = int(main_Y.size());
    for (int r = 0; r < m; r = r + 10)
    {

        int c = 359 - r;
        Pointright.push_back(Point2i(right_X[c], main_Y[c]));
    }
}

void LANEDETECTION::processingb_frame(Mat &frame, cuda::GpuMat &src, cuda::GpuMat &dst, struct last_fit *last_fit)
{
    cuda::GpuMat unwrap_framein, unwrap_frameout, cuda_frameout, dilate_out;
    cv::Point2f src_points[4];
    cv::Point2f dst_points[4];

    src_points[0] = cv::Point2f(290, 230);
    src_points[1] = cv::Point2f(350, 230);
    src_points[2] = cv::Point2f(520, 340);
    src_points[3] = cv::Point2f(130, 340);

    dst_points[0] = cv::Point2f(130, 0);
    dst_points[1] = cv::Point2f(520, 0);
    dst_points[2] = cv::Point2f(520, 360);
    dst_points[3] = cv::Point2f(130, 360);

    vector<Point2i> nonZeroCoordinates;
    vector<float> polyleft_in;
    vector<float> polyright_in;
    vector<int> Leftx;
    vector<int> rightx;
    vector<int> main_y;

    cuda_frameout.upload(frame);
    LANEDETECTION::erode_dilate(cuda_frameout, dilate_out);
    LANEDETECTION::video_frame(dilate_out, polyleft_in, polyright_in, last_fit);
    LANEDETECTION::curvature_sanity_check(polyleft_in, polyright_in, Leftx, rightx, main_y, last_fit);

    Mat maskImage = Mat(frame.size(), CV_8UC3, Scalar(0));
    vector<Point2i> PointLeft;
    vector<Point2i> PointRight;

    left_point(Leftx, main_y, PointLeft);

    right_point(rightx, main_y, PointRight);

    vector<Point2i> PointLeftRight;
    PointLeft.insert(PointLeft.end(), PointRight.begin(), PointRight.end());
    PointLeftRight = PointLeft;

    polylines(maskImage, PointLeft, false, Scalar(0, 0, 255), 20, 150, 0);
    polylines(maskImage, PointRight, false, Scalar(0, 0, 255), 20, 150, 0);

    const Point *pts = (const cv::Point *)Mat(PointLeftRight).data;
    int npts = Mat(PointLeftRight).rows;
    fillPoly(maskImage, &pts, &npts, 1, Scalar(0, 255, 0), 8);
    unwrap_framein.upload(maskImage);
    LANEDETECTION::wrap_frame(unwrap_framein, unwrap_frameout, dst_points, src_points);

    cuda::addWeighted(src, 1, unwrap_frameout, 0.5, -1, dst);
}

__global__ void VecAdd(int n, const float *A, const float *B, float *C)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n)
    {
        C[tid] = A[tid] + B[tid];
    }
}

void basicVecAdd(float *A, float *B, float *C, int n, int bucket)
{
    // Initialize thread block and kernel grid dimensions ---------------------
    unsigned int gridx_lim = 2147483647;
    unsigned int GRID_SIZE = max(1, min((n + BLOCK_SIZE) / BLOCK_SIZE, gridx_lim));
    /* Keep Grid size divisible by the warp size (32) */

    if ((GRID_SIZE % 32) != 0)
    {
        GRID_SIZE = GRID_SIZE + (32 - (GRID_SIZE % 32)) < gridx_lim ? GRID_SIZE + (32 - (GRID_SIZE % 32)) : GRID_SIZE - (GRID_SIZE % 32);
    }

    dim3 dim_grid(GRID_SIZE, 1, 1), dim_block(BLOCK_SIZE, 1, 1);

    VecAdd<<<dim_grid, dim_block, 0, streams[bucket]>>>(n, A, B, C);
}

void run_vec_add(struct vec_struct *shared_request, int bucket, bool *in_use)
{
    pthread_mutex_lock(&shared_request->pthread_mutex);
    float *A_d, *B_d, *C_d;
    size_t A_sz, B_sz, C_sz;
    unsigned VecSize = shared_request->request.VecSize;
    A_sz = VecSize;
    B_sz = VecSize;
    C_sz = VecSize;
    cudaMallocAsync((void **)&A_d, sizeof(float) * A_sz, streams[bucket]);
    cudaMallocAsync((void **)&B_d, sizeof(float) * B_sz, streams[bucket]);
    cudaMallocAsync((void **)&C_d, sizeof(float) * C_sz, streams[bucket]);

    cudaMemcpyAsync(A_d, shared_request->request.A_h, sizeof(float) * A_sz, cudaMemcpyHostToDevice, streams[bucket]);
    cudaMemcpyAsync(B_d, shared_request->request.B_h, sizeof(float) * B_sz, cudaMemcpyHostToDevice, streams[bucket]);
    // cudaMemcpyAsync(C_d, shared_request->request.C_h, sizeof(float) * C_sz, cudaMemcpyHostToDevice,streams[bucket]);
    basicVecAdd(A_d, B_d, C_d, VecSize, bucket);
    cudaMemcpyAsync(shared_request->response.C_h, C_d, sizeof(float) * C_sz, cudaMemcpyDeviceToHost, streams[bucket]);
    cudaStreamSynchronize(streams[bucket]);
    shared_request->ready = true;
    *in_use = false;
    pthread_cond_signal(&shared_request->pthread_cv);
    pthread_mutex_unlock(&shared_request->pthread_mutex);
    cudaFreeAsync(A_d, streams[bucket]);
    cudaFreeAsync(B_d, streams[bucket]);
    cudaFreeAsync(C_d, streams[bucket]);
}

__global__ void reduction(float *output, float *input, unsigned len)
{

    __shared__ float partialSum[sizeof(float) * BLOCK_SIZE];

    unsigned int tid = threadIdx.x + blockIdx.x * (blockDim.x * 2);
    partialSum[threadIdx.x] = input[tid];
    if (tid + blockDim.x < len)
    {
        partialSum[threadIdx.x] += input[tid + blockDim.x];
    }
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            partialSum[threadIdx.x] += partialSum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        output[blockIdx.x] = partialSum[0];
    }
    __syncthreads();
}

void reduction(float *in_d, float *out_d, unsigned int in_elements, unsigned int out_elements, int bucket)
{
    dim3 dim_grid, dim_block;
    dim_block.x = BLOCK_SIZE;
    dim_block.y = dim_block.z = 1;
    dim_grid.x = out_elements;
    dim_grid.y = dim_grid.z = 1;
    reduction<<<dim_grid, dim_block, 0, streams[bucket]>>>(out_d, in_d, in_elements);
}

void run_reduction(struct reduction_struct *shared_request, int bucket, bool *in_use)
{
    pthread_mutex_lock(&shared_request->pthread_mutex);
    float *in_h, *out_h;
    float *in_d, *out_d;
    unsigned in_elements, out_elements;
    int i;
    in_elements = shared_request->request.in_elements;
    out_elements = in_elements / (BLOCK_SIZE << 1);
    if (in_elements % (BLOCK_SIZE << 1))
        out_elements++;
    cudaMallocAsync((void **)&in_d, in_elements * sizeof(float), streams[bucket]);
    cudaMallocAsync((void **)&out_d, out_elements * sizeof(float), streams[bucket]);
    cudaMemcpyAsync(in_d, shared_request->request.in_h, in_elements * sizeof(float), cudaMemcpyHostToDevice, streams[bucket]);
    cudaMemsetAsync(out_d, 0, out_elements * sizeof(float), streams[bucket]);
    reduction(in_d, out_d, in_elements, out_elements, bucket);
    cudaMemcpyAsync(shared_request->response.out_h, out_d, out_elements * sizeof(float), cudaMemcpyDeviceToHost, streams[bucket]);
    cudaStreamSynchronize(streams[bucket]);
    shared_request->ready = true;
    *in_use = false;
    pthread_cond_signal(&shared_request->pthread_cv);
    pthread_mutex_unlock(&shared_request->pthread_mutex);
    cudaFreeAsync(in_d, streams[bucket]);
    cudaFreeAsync(out_d, streams[bucket]);
}
