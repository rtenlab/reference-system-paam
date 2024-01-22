#ifndef __GPU_OPERATIONS_H__
#define __GPU_OPERATIONS_H__

#include "paam_server_interfaces/msg/gpu_request.hpp"
#include "paam_server_interfaces/msg/callback_row.hpp"
#include "paam_server_interfaces/msg/admissions.hpp"

#include "paam_structs.h"
#ifdef OVERHEAD_DEBUG
    std::ofstream ex_time;
    static struct timeval overhead_ctime;
    std::string filename = "/home/paam/Research/overhead/overhead_breakdown.csv";
    log_struct* logger;
    //std::vector <std::stringstream> buffer;
    std::stringstream buffer;
#endif
class LANEDETECTION
{
private:
    std::string device;
    cv::VideoCapture cap;
    cv::Mat mapa, mapb;
    cv::cuda::GpuMat gpu_mapa, gpu_mapb;

public:
    LANEDETECTION();
    LANEDETECTION(std::string device);
    LANEDETECTION(cv::VideoCapture *cap);
    ~LANEDETECTION();
    float center_dist;
    float left_curverad;
    float right_curverad;
    std::vector<float> polyfiteigen(const std::vector<float> &xv, const std::vector<float> &yv, int order);
    std::vector<float> polyvaleigen(const std::vector<float> &oCoeff, const std::vector<float> &oX);
    void gray_frame(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst);
    void binary_frame(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst);
    void hsv_frame(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst);
    void wrap_frame(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst, cv::Point2f *src_points, cv::Point2f *dst_points);
    void sobel_frame(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst);
    void resize_frame(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst, int resize_height, int resize_width);
    void erode_dilate(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst);
    void video_frame(cv::cuda::GpuMat &src, std::vector<float> &polyleft_out, std::vector<float> &polyright_out, struct last_fit *last_fit);
    void first_frame(cv::cuda::GpuMat &src, std::vector<float> &polyright_f, std::vector<float> &polyleft_f);
    void nxt_frame(cv::cuda::GpuMat &src, std::vector<float> &polyright_n, std::vector<float> &polyleft_n, struct last_fit *last_fit);
    void curvature_sanity_check(std::vector<float> &polyleft_in, std::vector<float> &polyright_in, std::vector<int> &Leftx, std::vector<int> &rightx, std::vector<int> &main_y, struct last_fit *last_fit);
    void processinga_frame(cv::cuda::GpuMat &src, cv::cuda::GpuMat &resize, cv::cuda::GpuMat &dst);
    void processingb_frame(cv::Mat &frame, cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst, struct last_fit *last_fit);
    void init_lane_detection(std::string calibration_file);
    cv::Mat get_frame();
};
void run_hist(struct hist_struct *shared_request, int bucket, bool *in_use);
void run_sgemm(struct gemm_struct *shared_request, int bucket, bool *in_use);
void run_vec_add(struct vec_struct *shared_request, int bucket, bool *in_use);
void run_reduction(struct reduction_struct *shared_request, int bucket, bool *in_use);
void yolo_detect(struct yolo_struct *shared_request, bool *in_use, cv::dnn::Net net, const std::vector<std::string> className);
void lane_detect(struct lane_struct *shared_request, bool *in_use, int stream_id, struct last_fit *last_fit, LANEDETECTION *lanedetection);
unsigned int* run_remote_hist(int in_h, int num_elements, int num_bins,  int bucket, bool *in_use);
int* run_remote_sgemm(int in_h, int num_elements, int num_bins,  int bucket, bool *in_use);
int* run_remote_vec_add(int in_h, int num_elements, int num_bins,  int bucket, bool *in_use);
int* run_remote_reduction(int in_h, int num_elements, int num_bins,  int bucket, bool *in_use);

std::vector<cv::VideoCapture> devices;
int init_cuda_streams();
void destroy_cuda_streams();
#endif
