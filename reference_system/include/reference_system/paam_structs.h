#include <iostream>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/time.h>
#include <stdio.h>
#include <chrono>
#include <functional>
#include <queue>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <fstream>
#include <sched.h>
#include <condition_variable>
#include <mutex>
#include <pthread.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <list>
#include <numeric>
#include <random>
#include <vector>

const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};
const int WIDTH = 1280;
const int HEIGHT = 720;
const int CHANNELS = 3;

struct gemm_request
{
  float A_h[100000000];
  float B_h[100000000];
  uint32_t priority;
  unsigned char uuid[16];
  size_t A_sz;
  size_t B_sz;
  size_t C_sz;
  uint32_t matArow;
  uint32_t matAcol;
  uint32_t matBrow;
  uint32_t matBcol;
  uint32_t matCrow;
  uint32_t matCcol;
};

struct gemm_response
{
  float C_h[100000000];
};

struct gemm_struct
{
  volatile bool ready;
  pthread_mutex_t pthread_mutex;
  pthread_cond_t pthread_cv;
  uint64_t size;
  struct gemm_request request;
  struct gemm_response response;
};

struct hist_request
{
  unsigned int in_h[1000000];
  uint32_t priority;
  unsigned char uuid[16];
  int nbins;
};

struct hist_response
{
  int bins_h[1000000];
};
struct hist_struct
{
  volatile bool ready;
  pthread_mutex_t pthread_mutex;
  pthread_cond_t pthread_cv;
  uint64_t size;
  struct hist_request request;
  struct hist_response response;
};

struct vec_request
{
  size_t A_sz;
  size_t B_sz;
  size_t C_sz;
  float A_h[1000000];
  float B_h[1000000];
  unsigned int VecSize = 1000000;
};
struct vec_response
{
  float C_h[1000000];
};

struct vec_struct
{
  volatile bool ready;
  pthread_mutex_t pthread_mutex;
  pthread_cond_t pthread_cv;
  uint64_t size;
  struct vec_request request;
  struct vec_response response;
};

struct reduction_request
{
  unsigned int in_elements = 1000000;
  unsigned int out_elements = 128;
  float in_h[1000000];
};
struct reduction_response
{
  float out_h[1024];
};

struct reduction_struct
{
  volatile bool ready;
  pthread_mutex_t pthread_mutex;
  pthread_cond_t pthread_cv;
  uint64_t size;
  struct reduction_request request;
  struct reduction_response response;
};

struct Detection
{
  int class_id;
  float confidence;
  cv::Rect box;
};

struct yolo_request
{
  int rows;
  int cols;
  int type;
  int device_id;
  unsigned char frame[WIDTH * HEIGHT * CHANNELS];
  struct timeval last_timestamp;
  struct timeval current_timestamp;
};

struct yolo_response
{
  int size;
  unsigned char frame[WIDTH * HEIGHT * CHANNELS];
  Detection detections[25];
};

struct yolo_struct
{
  volatile bool ready;
  pthread_mutex_t pthread_mutex;
  pthread_cond_t pthread_cv;
  struct yolo_request request;
  struct yolo_response response;
};
struct lane_request
{
  int device_id;
  struct timeval last_timestamp;
  struct timeval current_timestamp;
};

struct lane_response
{
  int rows;
  int cols;
  int type;
  int size_L;
  int size_R;
  float polyfit_left[128];
  float polyfit_right[128];
  unsigned char frame[WIDTH * HEIGHT * CHANNELS];
};
struct lane_struct
{
  volatile bool ready;
  pthread_mutex_t pthread_mutex;
  pthread_cond_t pthread_cv;
  struct lane_request request;
  struct lane_response response;
  struct timeval last_timestamp;
  struct timeval current_timestamp;
};
struct last_fit
{
  std::vector<float> polyright_last;
  std::vector<float> polyleft_last;
};

struct tpu_request
{
  unsigned char image[224 * 224 * 3];
  int image_width = 0;
  int image_height = 0;
  int image_bpp = 0;
  double threshold = 0.1;
};
struct tpu_response
{
  std::pair<int, float> result[3];
};
struct tpu_struct
{
  volatile bool ready;
  pthread_mutex_t pthread_mutex;
  pthread_cond_t pthread_cv;
  struct tpu_request request;
  struct tpu_response response;
};
class log_struct
{
private:
  char filename[52];
  struct timeval start_time;
  struct timeval end_time;

public:
  // log_struct();
  void set_filename();
  void set_start();
  void set_end();
  void log_latency(std::string type);
};

void log_struct::set_start()
{
  gettimeofday(&start_time, NULL);
}
void log_struct::set_end()
{
  gettimeofday(&end_time, NULL);
}
void log_struct::log_latency(std::string type)
{
  std::ofstream fileout;
  fileout.open("/home/paam/Research/overhead/overhead_breakdown.csv", std::ios_base::app);
  fileout << type << ": " << end_time.tv_sec - start_time.tv_sec << "," << end_time.tv_usec - start_time.tv_usec << std::endl;
  fileout.close();
}
void log_struct::set_filename()
{
  char temp[52] = "/home/paam/Research/overhead/overhead_breakdown.csv";
  strcpy(this->filename, temp);
}
