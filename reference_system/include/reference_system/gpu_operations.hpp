#ifndef GPU_OPERATIONS_HPP
#define GPU_OPERATIONS_HPP

#include <functional>
#include <memory>
#include <thread>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h> //might need cmakelist update
#include <sys/time.h>
#include <unistd.h> //might need cmakelist update
#include <chrono>
#include <cinttypes>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <utility>
#include <boost/uuid/uuid.hpp>            // uuid class
#include <boost/uuid/uuid_generators.hpp> // generators
#include <boost/uuid/uuid_io.hpp>         // streaming operators etc.
#include <boost/lexical_cast.hpp>
#include "edgetpu.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"
// void run_sgemm(float *A_h, float *B_h, float *C_h, size_t A_sz, size_t B_sz, size_t C_sz, unsigned int matArow, unsigned int matAcol, unsigned int matBrow, unsigned int matBcol);
// void run_hist(unsigned int* in_h, unsigned int* bins_h, unsigned int num_elements, unsigned int num_bins);
class tpu_operator
{
    public:
    tpu_operator();
    ~tpu_operator();
    void init_tpu();
    void tpu_wrapper();
    std::string model_file = "/home/aamf/Research/AAMF-RTAS/src/aamf_server/test_data/inception_v2_224_quant_edgetpu.tflite";
    std::string label_file = "/home/aamf/Research/AAMF-RTAS/src/aamf_server/test_data/imagenet_labels.txt.1";
    std::string input_file = "/home/aamf/Research/AAMF-RTAS/src/aamf_server/test_data/resized_cat.bmp";
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    size_t file_size;
    std::shared_ptr<std::vector<uint8_t>> file_data;
    std::vector<std::string> labels;
    std::vector<uint8_t> image;
    const size_t kBmpFileHeaderSize = 14;
    const size_t kBmpInfoHeaderSize = 40;
    const size_t kBmpHeaderSize = kBmpFileHeaderSize + kBmpInfoHeaderSize;
    double threshold = 0.1;
    std::pair<int, float> result[3];
    int32_t ToInt32(const char p[4])
    {
        return (p[3] << 24) | (p[2] << 16) | (p[1] << 8) | p[0];
    }

    std::vector<uint8_t> ReadBmpImage(const char *filename,
                                      int *out_width = nullptr,
                                      int *out_height = nullptr,
                                      int *out_channels = nullptr)
    {
        assert(filename);

        std::ifstream file(filename, std::ios::binary);
        if (!file)
            return {}; // Open failed.

        char header[kBmpHeaderSize];
        if (!file.read(header, sizeof(header)))
            return {}; // Read failed.

        const char *file_header = header;
        const char *info_header = header + kBmpFileHeaderSize;

        if (file_header[0] != 'B' || file_header[1] != 'M')
            return {}; // Invalid file type.

        const int channels = info_header[14] / 8;
        if (channels != 1 && channels != 3)
            return {}; // Unsupported bits per pixel.

        if (ToInt32(&info_header[16]) != 0)
            return {}; // Unsupported compression.

        const uint32_t offset = ToInt32(&file_header[10]);
        if (offset > kBmpHeaderSize &&
            !file.seekg(offset - kBmpHeaderSize, std::ios::cur))
            return {}; // Seek failed.

        int width = ToInt32(&info_header[4]);
        if (width < 0)
            return {}; // Invalid width.

        int height = ToInt32(&info_header[8]);
        const bool top_down = height < 0;
        if (top_down)
            height = -height;

        const int line_bytes = width * channels;
        const int line_padding_bytes =
            4 * ((8 * channels * width + 31) / 32) - line_bytes;
        std::vector<uint8_t> image(line_bytes * height);
        for (int i = 0; i < height; ++i)
        {
            uint8_t *line = &image[(top_down ? i : (height - 1 - i)) * line_bytes];
            if (!file.read(reinterpret_cast<char *>(line), line_bytes))
                return {}; // Read failed.
            if (!file.seekg(line_padding_bytes, std::ios::cur))
                return {}; // Seek failed.
            if (channels == 3)
            {
                for (int j = 0; j < width; ++j)
                    std::swap(line[3 * j], line[3 * j + 2]);
            }
        }

        if (out_width)
            *out_width = width;
        if (out_height)
            *out_height = height;
        if (out_channels)
            *out_channels = channels;
        return image;
    }

    std::vector<std::string> ReadLabels(const std::string &filename)
    {
        std::ifstream file(filename);
        if (!file)
            return {}; // Open failed.

        std::vector<std::string> lines;
        for (std::string line; std::getline(file, line);)
            lines.emplace_back(line);
        return lines;
    }

    std::string GetLabel(const std::vector<std::string> &labels, int label)
    {
        if (label >= 0 && label < labels.size())
            return labels[label];
        return std::to_string(label);
    }

    std::vector<float> Dequantize(const TfLiteTensor &tensor)
    {
        const auto *data = reinterpret_cast<const uint8_t *>(tensor.data.data);
        std::vector<float> result(tensor.bytes);
        for (int i = 0; i < tensor.bytes; ++i)
            result[i] = tensor.params.scale * (data[i] - tensor.params.zero_point);
        return result;
    }

    std::vector<std::pair<int, float>> Sort(const std::vector<float> &scores,
                                            float threshold)
    {
        std::vector<const float *> ptrs(scores.size());
        std::iota(ptrs.begin(), ptrs.end(), scores.data());
        auto end = std::partition(ptrs.begin(), ptrs.end(),
                                  [=](const float *v)
                                  { return *v >= threshold; });
        std::sort(ptrs.begin(), end,
                  [](const float *a, const float *b)
                  { return *a > *b; });

        std::vector<std::pair<int, float>> result;
        result.reserve(end - ptrs.begin());
        for (auto it = ptrs.begin(); it != end; ++it)
            result.emplace_back(*it - scores.data(), **it);
        return result;
    }
};

tpu_operator::tpu_operator()
{
    init_tpu();
}
tpu_operator::~tpu_operator()
{
    interpreter->~Interpreter();
    edgetpu_context->~EdgeTpuContext();
}
void tpu_operator::init_tpu()
{
    this->model = tflite::FlatBufferModel::BuildFromFile(this->model_file.c_str());
    if (!this->model)
    {
        std::cerr << "Cannot read model from " << this->model_file << std::endl;
        return;
    }

    this->edgetpu_context = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();

    this->resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());

    if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk)
    {
        std::cerr << "Cannot create interpreter" << std::endl;
        return;
    }
    interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context.get());
    interpreter->SetNumThreads(1);
    // Allocate tensor buffers.
    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cerr << "Cannot allocate interpreter tensors" << std::endl;
        return;
    }
    // Set interpreter input.
    int image_width = 0;
    int image_height = 0;
    int image_bpp = 0;
    this->image = ReadBmpImage(this->input_file.c_str(), &image_width, &image_height, &image_bpp);
    std::printf("Image size: %d x %d x %d. Image vector size: %d\n", image_width, image_height, image_bpp, image.size());
    labels = ReadLabels(this->label_file);
    double threshold = 0.1;
    if (image.empty())
    {
        std::cerr << "Cannot read image from " << input_file << std::endl;
        return;
    }

    // Allocate tensor buffers.
    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cerr << "Cannot allocate interpreter tensors" << std::endl;
        return;
    }
    // Set interpreter input.
    const auto *input_tensor = interpreter->input_tensor(0);
    if (input_tensor->type != kTfLiteUInt8 ||          //
        input_tensor->dims->data[0] != 1 ||            //
        input_tensor->dims->data[1] != image_height || //
        input_tensor->dims->data[2] != image_width ||  //
        input_tensor->dims->data[3] != image_bpp)
    {
        std::cerr << "Input tensor shape does not match input image" << std::endl;
        return;
    }
    std::copy(image.begin(), image.end(),
              interpreter->typed_input_tensor<uint8_t>(0));
    // Run inference.
    if (interpreter->Invoke() != kTfLiteOk)
    {
        std::cerr << "Cannot invoke interpreter" << std::endl;
        return;
    }
    // Get interpreter output.
    auto results = Sort(Dequantize(*interpreter->output_tensor(0)), threshold);
    // std::printf("Size of Result Tensor: %d\n", results.size());
    for (auto &result : results)
        std::cout << std::setw(7) << std::fixed << std::setprecision(5)
                  << result.second << GetLabel(labels, result.first) << std::endl;
}

void tpu_operator::tpu_wrapper()
{

    timeval t1, t2;
    gettimeofday(&t1, NULL);
    //std::copy(image.begin(), image.end(), interpreter->typed_input_tensor<uint8_t>(0));
    // Run inference.
    std::memcpy(interpreter->typed_input_tensor<uint8_t>(0), &image[0], image.size());
    if (interpreter->Invoke() != kTfLiteOk)
    {
        std::cerr << "Cannot invoke interpreter" << std::endl;
        return;
    }
    // Get interpreter output.

    auto results = Sort(Dequantize(*interpreter->output_tensor(0)), threshold);
    // std::printf("Size of Result Tensor: %d\n", results.size());
    for ( int i = 0; i < 3; i++)
    {
        result[i] = std::make_pair(results[i].first, results[i].second);
    }
    gettimeofday(&t2, NULL);
    long elapsedTime = (t2.tv_usec - t1.tv_usec) ;
    std::cout << "Time taken by function: " << elapsedTime << " microseconds" << std::endl;
    /*for (auto &result : results)
        std::cout << std::setw(7) << std::fixed << std::setprecision(5)
                  << result.second << GetLabel(labels, result.first) << std::endl;*/
}

class gemm_operator
{
public:
    gemm_operator();
    ~gemm_operator();
    // void run_sgemm(float *A_h, float *B_h, float *C_h, size_t A_sz, size_t B_sz, size_t C_sz, unsigned int matArow, unsigned int matAcol, unsigned int matBrow, unsigned int matBcol);
    void init_sgemm();
    void gemm_wrapper();
    float *A_h, *B_h, *C_h;
    unsigned matArow, matAcol;
    unsigned matBrow, matBcol;
    size_t A_sz, B_sz, C_sz;
};

void gemm_operator::init_sgemm()
{
    matArow = 750;
    matAcol = matBrow = 750;
    matBcol = 750;

    A_sz = matArow * matAcol;
    B_sz = matBrow * matBcol;
    C_sz = matArow * matBcol;

    A_h = (float *)malloc(sizeof(float) * A_sz);
    for (unsigned int i = 0; i < A_sz; i++)
    {
        A_h[i] = (rand() % 100) / 100.00;
    }

    B_h = (float *)malloc(sizeof(float) * B_sz);
    for (unsigned int i = 0; i < B_sz; i++)
    {
        B_h[i] = (rand() % 100) / 100.00;
    }

    C_h = (float *)malloc(sizeof(float) * C_sz);
}
gemm_operator::gemm_operator()
{
    init_sgemm();
}
gemm_operator::~gemm_operator()
{
    free(A_h);
    free(B_h);
    free(C_h);
}
void di_gemm(gemm_operator *gemm_op);
void gemm_operator::gemm_wrapper(void)
{
    timeval t1, t2;
    gettimeofday(&t1, NULL);
    di_gemm(this);
    gettimeofday(&t2, NULL);
    long elapsedTime = (t2.tv_usec - t1.tv_usec) ;
    std::cout << "Time taken by GEMM: " << elapsedTime << " microseconds" << std::endl;
}
#endif // GPU_OPERATIONS_HPP