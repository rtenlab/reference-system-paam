#include <iostream>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdio.h>
#include <chrono>
#include <functional>
#include <queue>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <fstream>
#include <sched.h>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <pthread.h>
#include <unordered_map>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/multi_array_layout.hpp>
#include <std_msgs/msg/multi_array_dimension.hpp>
#include <boost/uuid/uuid.hpp>            // uuid class
#include <boost/uuid/uuid_generators.hpp> // generators
#include <boost/uuid/uuid_io.hpp>         // streaming operators etc.
#include <boost/lexical_cast.hpp>
#include <unique_identifier_msgs/msg/uuid.hpp>
#include "aamf_server_interfaces/msg/gpu_request.hpp"
#include "aamf_server_interfaces/msg/gpu_register.hpp"
#include "aamf_server_interfaces/msg/admissions_response.hpp"
#include "gpu_operations.h"
#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include "admissions.h"
#include "edgetpu.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "model_utils.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"
#include <algorithm>
#include <cassert>
#include <iomanip>
#include <numeric>
#include <utility>
#include <vector>
#include "logging.h"
using namespace std::chrono_literals;
using namespace cv::dnn;
using namespace cv;
using namespace std;
//#define OVERHEAD_DEBUG

#ifdef OVERHEAD_DEBUG
// std::ofstream ex_time;
// static struct timeval overhead_ctime;
static struct timeval start;
static struct timeval end;
static inline void log_time(std::string type)
{
  gettimeofday(&overhead_ctime, NULL);
  ex_time.open(filename, std::ios::app);
  ex_time << type << "," << overhead_ctime.tv_sec << "," << overhead_ctime.tv_usec << "\n";
  ex_time.close();
}

#endif

class AamfServer : public rclcpp::Node
{

public:
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<aamf_server_interfaces::msg::GPURegister>::SharedPtr publisher_;
  rclcpp::Publisher<aamf_server_interfaces::msg::GPURequest>::SharedPtr remote_publisher_;
  rclcpp::Publisher<aamf_server_interfaces::msg::AdmissionsResponse>::SharedPtr admissions_publisher_;
  rclcpp::Subscription<aamf_server_interfaces::msg::GPURequest>::SharedPtr subscription_;
  rclcpp::Subscription<aamf_server_interfaces::msg::GPURegister>::SharedPtr register_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<aamf_server_interfaces::msg::Admissions>::SharedPtr admissions_sub_;
  AamfServer() : Node("aamf_server"), count_(0)
  {
    // RCLCPP_INFO(this->get_logger(), "Making Publishers");
    publisher_ = this->create_publisher<aamf_server_interfaces::msg::GPURegister>("handshake_topic", 1000);
    remote_publisher_ = this->create_publisher<aamf_server_interfaces::msg::GPURequest>("remote_response_topic", 1000);
    subscription_ = this->create_subscription<aamf_server_interfaces::msg::GPURequest>(
        "request_topic", 1024, std::bind(&AamfServer::topic_callback, this, std::placeholders::_1));
    register_sub_ = this->create_subscription<aamf_server_interfaces::msg::GPURegister>(
        "registration_topic", 10, std::bind(&AamfServer::register_client_callback, this, std::placeholders::_1));
    admissions_sub_ = this->create_subscription<aamf_server_interfaces::msg::Admissions>("admissions_topic", 10, std::bind(&AamfServer::admissions_callback, this, std::placeholders::_1));
    admissions_publisher_ = this->create_publisher<aamf_server_interfaces::msg::AdmissionsResponse>("admissions_response_topic", 1000);

    // RCLCPP_INFO(this->get_logger(), "Initializing Cuda Streams");
    this->numCudaStreams = init_cuda_streams();
    this->init_worker_locks();
    // RCLCPP_INFO(this->get_logger(), "Initializing Worker Threads");
    this->init_worker_threads();
    // std::thread{ std::bind(&AamfServer::runtime_monitor, this) }.detach();
    std::thread{&AamfServer::runtime_monitor, this}.detach();
    this->init_tpu();
    this->use_tpu();
    this->attach_logging_shm();
    // std::thread rm_t(AamfServer::runtime_monitor).detach();
  }
  void attach_logging_shm(void)
  {
    int shmid = 65535;
    int logging_id = shmget(shmid, sizeof(log_struct), 0666 | IPC_CREAT);
    logger = (log_struct *)shmat(logging_id, (void *)0, 0);
    logger->set_filename();
    // pthread_mutexattr_t psharedm;
    // (void)pthread_mutexattr_init(&psharedm);
    // (void)pthread_mutexattr_setpshared(&psharedm, PTHREAD_PROCESS_SHARED);
    // (void)pthread_mutex_init(&logger->start_mutex, &psharedm);
    // logger = new(p) log_struct();
  }
  // aamf_logger* logger;
  // struct log_struct *logger;
  ~AamfServer()
  {
    int cv_worker = 6;
    int tpu_worker = 7;
    for (int bucket = 0; bucket < this->numCudaStreams; bucket++)
    {
      // std::lock_guard<std::mutex> lk(this->worker_mutex[bucket]);
      // this->cv[bucket].notify_one();
      pthread_mutex_lock(&worker_mutex[bucket]);
      pthread_cond_signal(&cv[bucket]);
      pthread_mutex_unlock(&worker_mutex[bucket]);
    }
    // std::lock_guard<std::mutex> cv_lk(this->worker_mutex[cv_worker]);
    // this->cv[cv_worker].notify_one();
    pthread_mutex_lock(&worker_mutex[cv_worker]);
    pthread_cond_signal(&cv[cv_worker]);
    pthread_mutex_unlock(&worker_mutex[cv_worker]);
    // std::lock_guard<std::mutex> tpu_lk(this->worker_mutex[tpu_worker]);
    // this->cv[tpu_worker].notify_one();
    pthread_mutex_lock(&worker_mutex[tpu_worker]);
    pthread_cond_signal(&cv[tpu_worker]);
    pthread_mutex_unlock(&worker_mutex[tpu_worker]);
    // this->clean_shared_memory();
    // RCLCPP_INFO(this->get_logger(), "Destroying Cuda Streams");
    destroy_cuda_streams();

    // ex_time.close();
  }

private:
  struct ThreadArg
  {
    AamfServer *server;
    int id;
  };
  std::vector<std::shared_ptr<chainset>> admitted_chains;
  std::vector<callback_row> admitted_callback_list;
  void init_accelerators(int num_gpus, int num_tpus)
  {
    for (int i = 0; i < num_gpus; i++)
    {
      this->accelerators.push_back(accelerator(i, 0.0, "gpu"));
    }
    for (int i = 0; i < num_tpus; i++)
    {
      this->accelerators.push_back(accelerator(i, 0.0, "tpu"));
    }
  }
  class compare_priority
  {
  public:
    bool operator()(const aamf_server_interfaces::msg::GPURequest::SharedPtr Request1,
                    const aamf_server_interfaces::msg::GPURequest::SharedPtr Request2)
    {
      if (Request1 == nullptr || Request2 == nullptr)
      {
        return false;
      }
      // Quickly Check priority, and return if priority is greater
      if (Request1->chain_priority >= Request2->chain_priority)
      {
        return false;
      }
      else if (Request1->callback_priority >= Request2->callback_priority)
      {
        return false;
      }
      return true;
    }
  };

  int numCudaStreams = 0;
  pthread_mutex_t worker_mutex[8];
  pthread_cond_t cv[8];
  void init_worker_locks(void)
  {
    for (int i = 0; i < 8; i++)
    {
      worker_mutex[i] = PTHREAD_MUTEX_INITIALIZER;
      cv[i] = PTHREAD_COND_INITIALIZER;
      // pthread_mutex_init(&worker_mutex[i], NULL);
      // pthread_cond_init(&cv[i], NULL);
    }
  }
  // std::condition_variable cv[8];
  // std::mutex worker_mutex[8];
  std::mutex stream_queue_mutex[8];
  std::priority_queue<std::shared_ptr<aamf_server_interfaces::msg::GPURequest>,
                      std::vector<std::shared_ptr<aamf_server_interfaces::msg::GPURequest>>, compare_priority>
      stream_queues[8];

  std::vector<int> in_use_key_list;
  // std::mutex //callback_key_map_mutex;
  struct callback_key_struct
  {
    std::vector<std_msgs::msg::String> kernels;
    std::vector<int> keys;
    std::vector<int> ids;
    int bucket; // this is the cuda stream or opencv worker that we will execute on. 0 to 7.
    struct hist_struct *hist_request = nullptr;
    struct gemm_struct *gemm_request = nullptr;
    struct yolo_struct *yolo_request = nullptr;
    struct reduction_struct *reduction_request = nullptr;
    struct vec_struct *vec_request = nullptr;
    struct lane_struct *lane_request = nullptr;
    struct last_fit last_fit;
    struct tpu_struct *tpu_request = nullptr;
    char chain_priority;
    bool data_in_use = false;
    int pid;
    bool remote = false;
    std::vector<std::string> devices;
  };
  void clean_working_queues(std::string callback_name, callback_key_struct *delete_struct)
  {
    int stream_id = delete_struct->bucket;                                            // access the appropriate queue
    //stream_queue_mutex[stream_id].lock();                                             // lock the stream queue
    std::vector<std::shared_ptr<aamf_server_interfaces::msg::GPURequest>> queue_list; // temporary list
    int i = 0;
    while (!stream_queues[stream_id].empty()) // while we access alll requests in the worker's queue
    {
      auto active_request = stream_queues[stream_id].top(); // get the top request
      // std::string callback_uuid(std::begin(active_request->uuid), std::end(active_request->uuid)); // get the callback UUID from the request
      std::string callback_uuid = this->toHexString(active_request->uuid);

      stream_queues[stream_id].pop();           // pop the item
      if (callback_uuid.compare(callback_name)) // If the uuid and the selected uuid are not equal, push the item to the temporary queue
      {
        queue_list.push_back(active_request);
      }
    }
    for (auto it = queue_list.begin(); it != queue_list.end(); it++) // place everything back into the appropriate queue
    {
      stream_queues[stream_id].push(*it);
    }
    //
    stream_queue_mutex[stream_id].unlock(); // lock the stream queue
  }
  void print_pq(int stream_id)
  {
    std::vector<std::shared_ptr<aamf_server_interfaces::msg::GPURequest>> queue_list;
    // stream_queue_mutex[stream_id].lock();
    int i = 0;
    while (!stream_queues[stream_id].empty())
    {
      auto active_request = stream_queues[stream_id].top();
      std::string callback_uuid = this->toHexString(active_request->uuid);

      // std::string callback_uuid(std::begin(active_request->uuid), std::end(active_request->uuid));

      stream_queues[stream_id].pop();
      queue_list.push_back(active_request);
      RCLCPP_INFO(this->get_logger(), "Priority Queue %i: Element %i, belonging to callback %s has priority: %i",
                  stream_id, ++i, callback_uuid.c_str(), active_request->chain_priority);
    }
    for (auto it = queue_list.begin(); it != queue_list.end(); it++)
    {
      stream_queues[stream_id].push(*it);
    }
    // stream_queue_mutex[stream_id].unlock();
  }
  // std::unordered_map<std::array<char, 16>, struct callback_key_struct> callback_key_map;
  std::unordered_map<std::string, struct callback_key_struct> callback_key_map;
  std::unordered_map<std::string, LANEDETECTION> device_detect_map;

  inline void clean_shared_memory(void)
  {
    // callback_key_map_mutex.lock();
    for (const auto &[callback_name, key_list] : callback_key_map)
    {
      detach_mem_from_callback_name(callback_name);
    }
    this->callback_key_map.clear();
    // callback_key_map_mutex.unlock();
  }

  std::vector<std::string> load_class_list()
  {
    std::vector<std::string> class_list;
    std::ifstream ifs("/home/aamf/Research/AAMF-RTAS/src/aamf_server/net/classes.txt");
    std::string line;
    while (getline(ifs, line))
    {
      class_list.push_back(line);
    }
    return class_list;
  }

  void load_net(cv::dnn::Net &net)
  {
    auto result = cv::dnn::readNet("/home/aamf/Research/AAMF-RTAS/src/aamf_server/net/yolov5s.onnx");
    result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    net = result;
  }
  void opencv_worker(void)
  {
    int cv_worker = 6;
    cv::dnn::Net net;
    load_net(net);
    std::vector<std::string> class_list = load_class_list();

    while (rclcpp::ok())
    {
      stream_queue_mutex[cv_worker].lock();

      if (stream_queues[cv_worker].empty())
      {
        stream_queue_mutex[cv_worker].unlock();
        pthread_mutex_lock(&worker_mutex[cv_worker]);
        pthread_cond_wait(&cv[cv_worker], &worker_mutex[cv_worker]);
        pthread_mutex_unlock(&worker_mutex[cv_worker]);
        // std::unique_lock<std::mutex> lk(worker_mutex[cv_worker]);
        // std::lock_guard<std::mutex> lk(worker_mutex[cv_worker]);
        // cv[cv_worker].wait(lk);
        // lk.unlock();
        continue;
      }

      RCLCPP_INFO(this->get_logger(), "Popping Request From Queue");
      auto active_request = stream_queues[cv_worker].top();
      stream_queues[cv_worker].pop();
      stream_queue_mutex[cv_worker].unlock();
      // callback_key_map_mutex.lock();
      std::string callback_uuid = this->toHexString(active_request->uuid);

      // std::string callback_uuid(std::begin(active_request->uuid), std::end(active_request->uuid));

      // if (this->callback_key_map.find(active_request->callback_name.data) == callback_key_map.end())
      if (this->callback_key_map.find(callback_uuid) == callback_key_map.end())
      {
        // RCLCPP_INFO(this->get_logger(), "Request invalid");
        continue;
      }
      auto key = &(*this->callback_key_map.find(callback_uuid)).second;
      // callback_key_map_mutex.unlock();
      switch (active_request->kernel_id)
      {
      case 2: // This is the histogram request
        // RCLCPP_INFO(this->get_logger(), "Running yolo object detector");
        key->data_in_use = true;
        yolo_detect(key->yolo_request, &key->data_in_use, net,
                    class_list); // if prioritized cuda stream, run independently
        // RCLCPP_INFO(this->get_logger(), "Request from %s Succeeded", callback_uuid.c_str());
        break;
      case 3: // Lane Detection
        // RCLCPP_INFO(this->get_logger(), "Running lane detection");
        key->data_in_use = true;

        // get_device_name_for_request
        // lane_detect(key->lane_request, &key->data_in_use, cv_worker, "/dev/video0", &key->last_fit); // if prioritized cuda stream, run independently
        // lane_detect(key->lane_request, &key->data_in_use, cv_worker, &key->last_fit, &(device_detect_map.find(active_request->dev_name.data)->second));
        // RCLCPP_INFO(this->get_logger(), "Request from %s Succeeded", callback_uuid.c_str());
        break;
      default:

        RCLCPP_INFO(this->get_logger(), "OPENCV Shmid invalid. Killing Server. callback_name: %s",
                    callback_uuid.c_str());
      }
    }
    // RCLCPP_INFO(this->get_logger(), "Destroying OpenCV Worker Thread");
  }

  void tpu_worker(void)
  {
    int tpu_worker = 7;
    while (rclcpp::ok())
    {
      stream_queue_mutex[tpu_worker].lock();

      if (stream_queues[tpu_worker].empty())
      {
        stream_queue_mutex[tpu_worker].unlock();
        pthread_mutex_lock(&worker_mutex[tpu_worker]);
        pthread_cond_wait(&cv[tpu_worker], &worker_mutex[tpu_worker]);
        pthread_mutex_unlock(&worker_mutex[tpu_worker]);
        // std::unique_lock<std::mutex> lk(worker_mutex[tpu_worker]);
        // cv[tpu_worker].wait(lk);
        // lk.unlock();
        continue;
      }

      // RCLCPP_INFO(this->get_logger(), "Popping Request From Queue");
      auto active_request = stream_queues[tpu_worker].top();
      stream_queues[tpu_worker].pop();
      stream_queue_mutex[tpu_worker].unlock();
      // callback_key_map_mutex.lock();
      std::string callback_uuid = this->toHexString(active_request->uuid);
      // std::string callback_uuid(std::begin(active_request->uuid), std::end(active_request->uuid));

      if (this->callback_key_map.find(callback_uuid) == callback_key_map.end())
      {
        // RCLCPP_INFO(this->get_logger(), "Request invalid");
        continue;
      }
      auto key = &(*this->callback_key_map.find(callback_uuid)).second;
      // callback_key_map_mutex.unlock();
      switch (active_request->kernel_id)
      {
      case 6: // This is the tpu request
        // RCLCPP_INFO(this->get_logger(), "Running Inference on Edge TPU");
        key->data_in_use = true;
        this->invoke_tpu(key->tpu_request, 7, &key->data_in_use);
        // yolo_detect(key->yolo_request, &key->data_in_use, net, class_list);
        // RCLCPP_INFO(this->get_logger(), "Request from %s Succeeded", callback_uuid.c_str());
        break;
      default:
        RCLCPP_INFO(this->get_logger(), "TPU Kern ID invalid. callback_name: %s",
                    callback_uuid.c_str());
      }
    }
    // RCLCPP_INFO(this->get_logger(), "Destroying TPU Worker Thread");
  }

  void gpu_worker_thread(int stream_id)
  // void gpu_worker_thread(void* args)
  {
    // int custom_stream_id = *(int *)arg;
    // int stream_id = custom_stream_id;
    //  RCLCPP_INFO(this->get_logger(), "Starting Worker for Stream %i", stream_id);

    int custom_stream_id = stream_id;
    while (rclcpp::ok())
    {
      stream_queue_mutex[stream_id].lock();

      if (stream_queues[stream_id].empty())
      {
        stream_queue_mutex[stream_id].unlock();
        pthread_mutex_lock(&worker_mutex[stream_id]);
        pthread_cond_wait(&cv[stream_id], &worker_mutex[stream_id]);
        pthread_mutex_unlock(&worker_mutex[stream_id]);
          //logger->set_end();
      //logger->log_latency("Worker Awakening");
        // std::unique_lock<std::mutex> lk(worker_mutex[stream_id]);
        // cv[stream_id].wait(lk);
        // lk.unlock();
        continue;
      }
#ifdef OVERHEAD_DEBUG
      logger->set_end();
      logger->log_latency("Worker Awakening");
      logger->set_start();
#endif
      // stream_queue_mutex[stream_id].lock();

      RCLCPP_INFO(this->get_logger(), "Popping Request From Queue");
      auto active_request = stream_queues[stream_id].top();
      if (active_request == nullptr)
      {
        continue;
      }
      stream_queues[stream_id].pop();
      stream_queue_mutex[stream_id].unlock();
#ifdef OVERHEAD_DEBUG
      logger->set_end();
      logger->log_latency("Request Popped");
      // log_end();
      // log_latency("Request Popped");
      //  log_time("Request Popped");
#endif
      // callback_key_map_mutex.lock();
      // std::string callback_uuid(std::begin(active_request->uuid), std::end(active_request->uuid));
      std::string callback_uuid = this->toHexString(active_request->uuid);
      if (this->callback_key_map.find(callback_uuid) == callback_key_map.end())
      {
        RCLCPP_INFO(this->get_logger(), "Request invalid");
        continue;
      }
      auto key = &(*this->callback_key_map.find(callback_uuid)).second;
      // callback_key_map_mutex.unlock();
      //  stream_id = 5;
      //  int custom_stream_id = 5;
      //  custom_stream_id = stream_id;
      switch (active_request->kernel_id)
      {
      case 0: // This is the histogram request
        // RCLCPP_INFO(this->get_logger(), "Running Hist Kernel");
        //  if(active_request->remote){
        //    key->data_in_use = true;
        //    auto response = run_remote_hist(active_request->in_h, active_request->num_elements, active_request->num_bins);
        //    remote_publisher_->publish(response);
        //  }
        if (custom_stream_id == 0) // if minimum cuda stream run concurrently
        {
          key->data_in_use = true;
          std::thread{&run_hist, key->hist_request, custom_stream_id, &key->data_in_use}.detach();
        }
        else
        {
          key->data_in_use = true;
          run_hist(key->hist_request, custom_stream_id, &key->data_in_use); // if prioritized cuda stream, run independently
        }
        RCLCPP_INFO(this->get_logger(), "Request from %s Succeeded", callback_uuid.c_str());

        break;

      case 1: // this is the GEMM kernel
        RCLCPP_INFO(this->get_logger(), "Running GEMM Kernel");

        if (custom_stream_id == 0)
        {
          key->data_in_use = true;
          std::thread{&run_sgemm, key->gemm_request, custom_stream_id, &key->data_in_use}.detach();
        }
        else
        {
          key->data_in_use = true;
          run_sgemm(key->gemm_request, custom_stream_id, &key->data_in_use);
        }
        RCLCPP_INFO(this->get_logger(), "Request from %s Succeeded", callback_uuid.c_str());
        break;
      case 4: // this is the VEC kernel
        // RCLCPP_INFO(this->get_logger(), "Running VEC Kernel");

        if (custom_stream_id == 0)
        {
          key->data_in_use = true;
          std::thread{&run_vec_add, key->vec_request, custom_stream_id, &key->data_in_use}.detach();
        }
        else
        {
          key->data_in_use = true;
          run_vec_add(key->vec_request, custom_stream_id, &key->data_in_use);
        }
        // RCLCPP_INFO(this->get_logger(), "Request from %s Succeeded", callback_uuid.c_str());
        break;
      case 5: // this is the RED kernel
        // RCLCPP_INFO(this->get_logger(), "Running Reduction Kernel");

        if (custom_stream_id == 0)
        {
          key->data_in_use = true;
          std::thread{&run_reduction, key->reduction_request, custom_stream_id, &key->data_in_use}.detach();
        }
        else
        {
          key->data_in_use = true;
          run_reduction(key->reduction_request, custom_stream_id, &key->data_in_use);
        }
        // RCLCPP_INFO(this->get_logger(), "Request from %s Succeeded", callback_uuid.c_str());
        break;
      default:

        RCLCPP_INFO(this->get_logger(), "Shmid invalid. Killing Server. callback_name: %s",
                    callback_uuid.c_str());
      }
    }
    if (stream_id == 0)
    {
      // RCLCPP_INFO(this->get_logger(), "Cleaning Up Shared Memory");
      //  this->clean_shared_memory();
    }
    // RCLCPP_INFO(this->get_logger(), "Destroying Worker Thread: %i", stream_id);
  }
  pthread_t worker_threads[8];
  // std::thread worker_threads[6];
  void init_worker_threads(void)
  {
    while (!this->numCudaStreams)
      ;

    for (int i = 0; i < this->numCudaStreams; i++)
    {
      // worker_threads[i] = std::thread{&AamfServer::gpu_worker_thread, this, i}; //.detach();
      // auto handle = worker_threads[i].native_handle();
      ThreadArg *arg = new ThreadArg{this, i};
      pthread_create(&worker_threads[i], nullptr, gpu_worker_thread_wrapper, arg);
      // worker_threads[i] = pthread_create(&worker_threads[i], NULL, &this->gpu_worker_thread, (void *)i);
      sched_param sch;
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(0, &cpuset);
      sch.sched_priority = 91 + i;
      if (pthread_setschedparam(worker_threads[i], SCHED_FIFO, &sch))
      {
        std::cout << "Failed to setschedparam: " << std::strerror(errno) << '\n';
      }
      if (pthread_setaffinity_np(worker_threads[i], sizeof(cpuset), &cpuset))
      {
        std::cout << "Failed to setcpuaffinity " << std::strerror(errno) << '\n';
      }
      pthread_detach(worker_threads[i]);
    }
    pthread_create(&worker_threads[6], nullptr, opencv_worker_thread_wrapper, new ThreadArg{this, 6});
    sched_param sch;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    sch.sched_priority = 90;
    if(pthread_setschedparam(worker_threads[6], SCHED_FIFO, &sch)){
      std::cout << "Failed to setschedparam: " << std::strerror(errno) << '\n';
    }
    if(pthread_setaffinity_np(worker_threads[6], sizeof(cpuset), &cpuset)){
      std::cout << "Failed to setcpuaffinity " << std::strerror(errno) << '\n';
    }
    pthread_detach(worker_threads[6]);

    pthread_create(&worker_threads[7], nullptr, tpu_worker_thread_wrapper, new ThreadArg{this, 7});
    CPU_ZERO(&cpuset);
    CPU_SET(1, &cpuset);
    sch.sched_priority = 99;
    if(pthread_setschedparam(worker_threads[7], SCHED_FIFO, &sch)){
      std::cout << "Failed to setschedparam: " << std::strerror(errno) << '\n';
    }
    if(pthread_setaffinity_np(worker_threads[7], sizeof(cpuset), &cpuset)){
      std::cout << "Failed to setcpuaffinity " << std::strerror(errno) << '\n';
    }
    pthread_detach(worker_threads[7]);
    //std::thread{&AamfServer::opencv_worker, this}.detach();
    //std::thread{&AamfServer::tpu_worker, this}.detach();
  }
  static void *opencv_worker_thread_wrapper(void *arg)
  {
    ThreadArg *args = (ThreadArg *)arg;
    args->server->opencv_worker();
    delete args;
    return nullptr;
  }
 static void *tpu_worker_thread_wrapper(void *arg)
  {
    ThreadArg *args = (ThreadArg *)arg;
    args->server->tpu_worker();
    delete args;
    return nullptr;
  }
  
  void topic_callback(aamf_server_interfaces::msg::GPURequest::SharedPtr request)
  {
#ifdef OVERHEAD_DEBUG
    // log_time("Message Received");
    logger->set_end();
    logger->log_latency("Message Transport");
    // log_start();
    logger->set_start();
#endif
    // std::thread{ std::bind(&AamfServer::enqueue, this, request) }.detach();
    // auto msg = std::make_shared<aamf_server_interfaces::msg::GPURequest>();
    //*msg = *request;
    // this->enqueue(msg);
    this->enqueue(request); // Throw pointer to request into queue
  }
  // bool chainset_addition_schedulable(std::vector<callback_row> incoming)
  // {
  //   std::vector<callback_row> temp = admitted_callback_list;
  //   for (auto &row : incoming)
  //   {
  //     temp.push_back(row);
  //   }
  //   chainset temp_chainset(temp);
  //   bool admitted = temp_chainset.schedulable();
  //   if (admitted)
  //   {
  //     admitted_callback_list = temp;
  //     return true;
  //   }
  //   return false;
  // }
  // callback_row make_callback_rows(aamf_server_interfaces::msg::CallbackRow row)
  // {
  //   return callback_row(row.period, row.cpu_time, row.gpu_time, row.deadline, row.chain_id, row.order, row.priority, row.cpu_id, row.executor_id, row.bucket);
  // }
  std::vector<accelerator> accelerators;
  int accelerator_types = 2;
  void admissions_callback(aamf_server_interfaces::msg::Admissions::SharedPtr request)
  {
    // std::vector<callback_row> data; // incoming callback chain
    // for (auto &callback : request->data)
    // {
    //   data.push_back(make_callback_rows(callback));
    // }
    // chainset incoming(data);

    // std::vector<std::vector<callback_row>> chains = incoming.to_callback_row_vector();

    // std::vector<chainset> chainsets;
    // for (auto &chain : chains)
    // {
    //   chainsets.push_back(chainset(chain));
    // }
    // // GPU WFD Assignment
    // for (auto &cs : chainsets)
    // {
    //   accelerator *assigned_accelerators[accelerator_types];
    //   for (int i = 0; i < accelerator_types; i++)
    //   {
    //     assigned_accelerators[i] = nullptr;
    //   }
    //   // ACCELERATOR INIT ASSIGNMENT
    //   for (auto &a : this->accelerators)
    //   {
    //     if (a.get_type() == "GPU" && assigned_accelerators[0]->get_type() == "none")
    //     {
    //       assigned_accelerators[0] = &a;
    //     }
    //     if (a.get_type() == "TPU" && assigned_accelerators[1]->get_type() == "none")
    //     {
    //       assigned_accelerators[1] = &a;
    //     }
    //   }
    //   // Worst fit decreasing assignment
    //   for (auto &a : this->accelerators)
    //   {
    //     if (a.get_type() == "GPU" && request->gpu)
    //     {
    //       if (a.get_util() < assigned_accelerators[0]->get_util())
    //       {
    //         assigned_accelerators[0] = &a;
    //       }
    //     }
    //     if (a.get_type() == "TPU" && request->tpu)
    //     {
    //       if (a.get_util() < assigned_accelerators[1]->get_util())
    //       {
    //         assigned_accelerators[1] = &a;
    //       }
    //     }
    //   }
    //   // Assign Accelerators
    //   for (auto &a : assigned_accelerators)
    //   {
    //     if (a != nullptr)
    //     {
    //       std::vector<chainset> temp_chainsets = a->get_chainsets();
    //       std::vector<callback_row> temp_callback_list;
    //       for (auto &c : temp_chainsets)
    //       {
    //         for (auto &row : c.to_callback_row())
    //         {
    //           temp_callback_list.push_back(row);
    //         }
    //       }
    //       //temp_callback_list.push_back(*data);
    //       chainset temp_chainset(temp_callback_list);
    //       if (temp_chainset.schedulable())
    //       {
    //         a->add_chainset(temp_chainset);
    //         break;
    //       }
    //     }
    //   }
    //}

    // aamf_server_interfaces::msg::AdmissionsResponse response;
    // response.id = request->id;
    // bool chainset_schedulable = incoming.schedulable();
    // if (chainset_schedulable)
    // {
    //   response.admitted = chainset_addition_schedulable(data);
    //   if (response.admitted)
    //   {
    //     admitted_chains.push_back(std::make_shared<chainset>(incoming));
    //   }
    // }
    // else
    // {
    //   response.admitted = false;
    // }
    // admissions_publisher_->publish(response);
  }
  inline int assign_bucket(char chain_priority)
  {
    int num_buckets = this->numCudaStreams;
    int min_prio = 0, max_prio = 70;
    if (chain_priority > max_prio)
    {
      return num_buckets -1;
    }
    else if (chain_priority < min_prio)
    {
      return 0;
    }
    int num_priorities = max_prio - min_prio + 1;
    int bucket_width = num_priorities / num_buckets;
    while (bucket_width * num_buckets < max_prio)
    {
      bucket_width++;
    }
    // bucket_width = 17;
    // //RCLCPP_INFO(this->get_logger(), "Bucket width: %i", bucket_width);
    for (auto i = 0; i < num_buckets; i++)
    {
      if (min_prio + bucket_width * i <= chain_priority && min_prio + bucket_width * (i + 1) >= chain_priority)
      {
        return i;
      }
    }
    // RCLCPP_INFO(this->get_logger(), "Invalid chain priority, bucket not assigned.");
    //  return 5;
    return -1;
  }

  void remote_register_logic(aamf_server_interfaces::msg::GPURegister::SharedPtr request)
  {
    struct callback_key_struct remote_client;
    remote_client.kernels = request->kernels;
    remote_client.pid = request->pid;
    remote_client.remote = true;
    remote_client.bucket = assign_bucket(request->chain_priority);
    // callback_key_map_mutex.lock();
    std::string callback_uuid = this->toHexString(request->uuid);
    // std::string callback_uuid(std::begin(request->uuid), std::end(request->uuid));

    this->callback_key_map.insert(std::make_pair(callback_uuid, remote_client));
    // callback_key_map_mutex.unlock();
    aamf_server_interfaces::msg::GPURegister response = *request; // makes a response and copies data
    response.should_register = false;                             // sets this so that clients know this is not a reg call
    publisher_->publish(response);
  }

  const size_t kBmpFileHeaderSize = 14;
  const size_t kBmpInfoHeaderSize = 40;
  const size_t kBmpHeaderSize = kBmpFileHeaderSize + kBmpInfoHeaderSize;

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
  std::string toHexString(const std::array<uint8_t, 16> &uuid)
  {
    std::string callback_uuid;
    callback_uuid.reserve(32); // 16 bytes * 2 characters per byte

    const char hexChars[] = "0123456789abcdef";
    for (const char &byteChar : uuid)
    {
      unsigned char byte = static_cast<unsigned char>(byteChar);
      callback_uuid.push_back(hexChars[byte >> 4]);
      callback_uuid.push_back(hexChars[byte & 0x0F]);
    }

    return callback_uuid;
  }
  void register_logic(aamf_server_interfaces::msg::GPURegister::SharedPtr request)
  {
    // std::string callback_uuid(std::begin(request->uuid), std::end(request->uuid));
    // std::array<char, 16> uuid_arr;
    // std::copy(request->uuid.begin(), request->uuid.end(), uuid_arr.begin());
    // std::string callback_uuid(std::begin(uuid_arr), std::end(uuid_arr));
    // std::stringstream ss;
    // ss << std::hex << std::setfill('0');
    // for (int i = 0; i < 16; ++i)
    // {
    //   ss << std::setw(2) << static_cast<unsigned>(request->uuid[i]);
    // }
    std::string callback_uuid = this->toHexString(request->uuid);
    RCLCPP_INFO(this->get_logger(), "Registration Request Received for Callback:  %s",
    callback_uuid.c_str());
    if (request->should_register) // If this is a creation request
    {
      pthread_mutexattr_t psharedm;
      pthread_condattr_t psharedc;
      (void)pthread_mutexattr_init(&psharedm);
      (void)pthread_mutexattr_setpshared(&psharedm, PTHREAD_PROCESS_SHARED);
      (void)pthread_condattr_init(&psharedc);
      (void)pthread_condattr_setpshared(&psharedc, PTHREAD_PROCESS_SHARED);

      struct callback_key_struct uuid_keys;
      uuid_keys.kernels = request->kernels;
      uuid_keys.pid = request->pid;
      aamf_server_interfaces::msg::GPURegister response = *request; // makes a response and copies data
      response.should_register = false;                             // sets this so that clients know this is not a reg call
      response.keys.clear();                                        // clear the client-defined keys
      uuid_keys.bucket = assign_bucket(request->chain_priority);
      for (auto kernel : request->kernels) // For each kernel
      {
        int key = rand() % 65535; // lets randomly generate a key within the range

        for (int i = 0; i < (int)this->in_use_key_list.size(); i++)
        {
          if (this->in_use_key_list.at(i) == key) // If the key is already in use
          {
            key = rand() & 65535; //  regenerate the key and try again
            i = 0;
          }
        }
        uuid_keys.keys.push_back(key); // Store the key for that kernel
        //  register shared memory regions and associate them with the approriate datatype and key
        if (kernel.data == "GEMM")
        {
          int gemm_id = shmget(key, sizeof(struct gemm_struct), 0666 | IPC_CREAT);
          if (gemm_id == -1)
          {
            RCLCPP_INFO(this->get_logger(), "Shmget failed, errno: %i, key: %i, gemm_id: %i", errno, key, gemm_id);
          }
          uuid_keys.ids.push_back(gemm_id);
          uuid_keys.gemm_request = (struct gemm_struct *)shmat(gemm_id, (void *)0, 0);
          if (uuid_keys.gemm_request == (void *)-1 || uuid_keys.gemm_request == nullptr)
          {
            RCLCPP_INFO(this->get_logger(), "Shmat failed, errno: %i", errno);
          }
          (void)pthread_mutex_init(&(uuid_keys.gemm_request->pthread_mutex), &psharedm);
          (void)pthread_cond_init(&(uuid_keys.gemm_request->pthread_cv), &psharedc);
        }
        else if (kernel.data == "HIST")
        {
          int hist_id = shmget(key, sizeof(struct hist_struct), 0666 | IPC_CREAT);
          if (hist_id == -1)
          {
            // RCLCPP_INFO(this->get_logger(), "Shmget failed, errno: %i, key: %i, hist_id: %i", errno, key, hist_id);
          }
          uuid_keys.ids.push_back(hist_id);
          uuid_keys.hist_request = (struct hist_struct *)shmat(hist_id, (void *)0, 0);
          if (uuid_keys.hist_request == (void *)-1 || uuid_keys.hist_request == nullptr)
          {
            // RCLCPP_INFO(this->get_logger(), "Shmat failed, errno: %i", errno);
          }
          (void)pthread_mutex_init(&(uuid_keys.hist_request->pthread_mutex), &psharedm);
          (void)pthread_cond_init(&(uuid_keys.hist_request->pthread_cv), &psharedc);
        }
        else if (kernel.data == "YOLO")
        {
          int yolo_id = shmget(key, sizeof(struct yolo_struct), 0666 | IPC_CREAT);
          if (yolo_id == -1)
          {
            // RCLCPP_INFO(this->get_logger(), "Shmget failed, errno: %i, key: %i, hist_id: %i", errno, key, yolo_id);
          }
          uuid_keys.ids.push_back(yolo_id);
          uuid_keys.yolo_request = (struct yolo_struct *)shmat(yolo_id, (void *)0, 0);
          if (uuid_keys.yolo_request == (void *)-1 || uuid_keys.yolo_request == nullptr)
          {
            // RCLCPP_INFO(this->get_logger(), "Shmat failed, errno: %i", errno);
          }
          (void)pthread_mutex_init(&(uuid_keys.yolo_request->pthread_mutex), &psharedm);
          (void)pthread_cond_init(&(uuid_keys.yolo_request->pthread_cv), &psharedc);
        }
        else if (kernel.data == "VEC")
        {
          int vec_id = shmget(key, sizeof(struct vec_struct), 0666 | IPC_CREAT);
          if (vec_id == -1)
          {
            // RCLCPP_INFO(this->get_logger(), "Shmget failed, errno: %i, key: %i, hist_id: %i", errno, key, vec_id);
          }
          uuid_keys.ids.push_back(vec_id);
          uuid_keys.vec_request = (struct vec_struct *)shmat(vec_id, (void *)0, 0);
          if (uuid_keys.vec_request == (void *)-1 || uuid_keys.vec_request == nullptr)
          {
            // RCLCPP_INFO(this->get_logger(), "Shmat failed, errno: %i", errno);
          }
          (void)pthread_mutex_init(&(uuid_keys.vec_request->pthread_mutex), &psharedm);
          (void)pthread_cond_init(&(uuid_keys.vec_request->pthread_cv), &psharedc);
        }
        else if (kernel.data == "RED")
        {
          // uuid_keys.bucket = 6;
          int reduction_id = shmget(key, sizeof(struct reduction_struct), 0666 | IPC_CREAT);
          if (reduction_id == -1)
          {
            // RCLCPP_INFO(this->get_logger(), "Shmget failed, errno: %i, key: %i, hist_id: %i", errno, key, reduction_id);
          }
          uuid_keys.ids.push_back(reduction_id);
          uuid_keys.reduction_request = (struct reduction_struct *)shmat(reduction_id, (void *)0, 0);
          if (uuid_keys.reduction_request == (void *)-1 || uuid_keys.reduction_request == nullptr)
          {
            // RCLCPP_INFO(this->get_logger(), "Shmat failed, errno: %i", errno);
          }
          (void)pthread_mutex_init(&(uuid_keys.reduction_request->pthread_mutex), &psharedm);
          (void)pthread_cond_init(&(uuid_keys.reduction_request->pthread_cv), &psharedc);
        }
        else if (kernel.data == "TPU")
        {
          int tpu_id = shmget(key, sizeof(struct tpu_struct), 0666 | IPC_CREAT);
          if (tpu_id == -1)
          {
            RCLCPP_INFO(this->get_logger(), "Shmget failed, errno: %i, key: %i, hist_id: %i", errno, key, tpu_id);
          }
          uuid_keys.ids.push_back(tpu_id);
          uuid_keys.tpu_request = (struct tpu_struct *)shmat(tpu_id, (void *)0, 0);
          if (uuid_keys.tpu_request == (void *)-1 || uuid_keys.tpu_request == nullptr)
          {
            RCLCPP_INFO(this->get_logger(), "Shmat failed, errno: %i", errno);
          }
          (void)pthread_mutex_init(&(uuid_keys.tpu_request->pthread_mutex), &psharedm);
          (void)pthread_cond_init(&(uuid_keys.tpu_request->pthread_cv), &psharedc);
        }
        else if (kernel.data == "LANE")
        {
          // uuid_keys.bucket = 6;
          int lane_id = shmget(key, sizeof(struct lane_struct), 0666 | IPC_CREAT);
          if (lane_id == -1)
          {
            // RCLCPP_INFO(this->get_logger(), "Shmget failed, errno: %i, key: %i, hist_id: %i", errno, key, lane_id);
          }
          uuid_keys.ids.push_back(lane_id);
          uuid_keys.lane_request = (struct lane_struct *)shmat(lane_id, (void *)0, 0);
          if (uuid_keys.lane_request == (void *)-1 || uuid_keys.lane_request == nullptr)
          {
            // RCLCPP_INFO(this->get_logger(), "Shmat failed, errno: %i", errno);
          }
          (void)pthread_mutex_init(&(uuid_keys.lane_request->pthread_mutex), &psharedm);
          (void)pthread_cond_init(&(uuid_keys.lane_request->pthread_cv), &psharedc);
          // Init lane detection modules
          for (auto &request : request->devices)
          {
            uuid_keys.devices.push_back(request.data);
            if (device_detect_map.find(request.data) == device_detect_map.end()) // if not found, make new data
            {
              LANEDETECTION lane_detection(request.data);
              // LANEDETECTION lane_detection((std::string *)&request.data);
              device_detect_map.insert(std::make_pair(request.data, lane_detection));
            }
          }
        }
        // show that the key is used
        in_use_key_list.push_back(key);
        // send response so the client node can attach to the appropriate memory regions
        response.keys.push_back(key);
      }

      // uuid_keys.devices = request->devices;
      // callback_key_map_mutex.lock();
      // this->callback_key_map.insert(std::make_pair(request->callback_name.data, uuid_keys));

      // std::string str(std::begin(arr), std::end(arr));
      this->callback_key_map.insert(std::make_pair(callback_uuid, uuid_keys));
      // callback_key_map_mutex.unlock();
      publisher_->publish(response);
    }
    else // DELETE MEM REGION AND ERASE KEY -- Call this deregistration
    {
      RCLCPP_INFO(this->get_logger(), "De-Registration Request Received for Callback:  %s",
                  callback_uuid.c_str());
      // callback_key_map_mutex.lock();
      this->detach_mem_from_callback_name(callback_uuid);
      // callback_key_map_mutex.unlock();
    }
  }

  void register_client_callback(aamf_server_interfaces::msg::GPURegister::SharedPtr request)
  {
    // std::thread{std::bind(&AamfServer::register_logic, this, request)}.detach();
    if (request->remote)
    {
      this->remote_register_logic(request);
    }
    else
    {
      this->register_logic(request);
    }
  }

  inline int find_bucket_for_callback_name(std::string callback_name)
  {
    // callback_key_map_mutex.lock();
    auto it = callback_key_map.find(callback_name);
    if (it == callback_key_map.end())
    {
      RCLCPP_INFO(this->get_logger(), "Callback Name %s not found in key_map. Cannot find appropriate bucket",
                  callback_name.c_str());
      // callback_key_map_mutex.unlock();
      return 0;
    }
    else
    {
      // callback_key_map_mutex.unlock();
      return (*it).second.bucket;
    }
  }

  inline void enqueue(aamf_server_interfaces::msg::GPURequest::SharedPtr shared_request)
  {
    RCLCPP_INFO(this->get_logger(), "Enqueueing GPU Access Request");
    //  Determine Bucket
    bool notify = false;
    // int bucket = this->find_bucket_for_callback_name(shared_request->callback_name.data);
    // std::string callback_uuid(std::begin(shared_request->uuid), std::end(shared_request->uuid));
    std::string callback_uuid = this->toHexString(shared_request->uuid);

    int bucket = this->find_bucket_for_callback_name(callback_uuid);

    if (shared_request->open_cv == true)
    {
      bucket = 6;
    }
    if (shared_request->tpu == true)
    {
      bucket = 7;
    }
    // bucket = this->numCudaStreams - 1;
    RCLCPP_INFO(this->get_logger(), "Enqueueing Request from Callback %s with executor PID %i to Bucket %i",
    callback_uuid.c_str(), shared_request->pid, bucket);

    stream_queue_mutex[bucket].lock(); // queue mutex
    // while (stream_queue_mutex[bucket].try_lock() != 0)
    ///{
    //};

    // if (stream_queues[bucket].empty())
    //{
    //   notify = true;
    // }
    stream_queues[bucket].push(shared_request); // Push Goal Handle to Queue
#ifdef OVERHEAD_DEBUG
    logger->set_end();
    logger->log_latency("Request Queueing");
    // log_end();
    // log_latency("Request Queued");
    //  log_time("Request Queued");
#endif
    //    //RCLCPP_INFO(this->get_logger(), "Top item in queue %i is a request from callback %s with priority %i", bucket,
    //    stream_queues[bucket].top()->callback_name.data.c_str(), stream_queues[bucket].top()->chain_priority);
    // print_pq(bucket);
    stream_queue_mutex[bucket].unlock(); // queue mutex

    // if (notify)
    //{
    //  std::lock_guard<std::mutex> lk(worker_mutex[bucket]);
    //  cv[bucket].notify_one();
    // pthread_mutex_lock(&worker_mutex[bucket]);
    pthread_cond_signal(&cv[bucket]);
    logger->set_start();

    // pthread_mutex_unlock(&worker_mutex[bucket]);

    //}
    // else
    //{
    // logger->set_start();
    //}
  }
  // Given the PID destroy and detach from all memory regions as well as delete the entry from the key map
  // inline void detach_mem_from_callback_name(pid_t pid)
  inline int detach_mem_from_callback_name(std::string callback_name)
  {
    // RCLCPP_INFO(this->get_logger(), "Deleting Memory For %s", callback_name.c_str());
    //  go thru key map and mark all regions for destruction and detach from memory
    if (callback_key_map.find(callback_name) == callback_key_map.end())
    {
      // RCLCPP_INFO(this->get_logger(), "Callback Name %s not found in key_map", callback_name.c_str());
      return -1;
    }
    auto delete_struct = this->callback_key_map.find(callback_name);
    if (delete_struct->second.data_in_use)
    {
      // RCLCPP_INFO(this->get_logger(), "Data In Use For %s", callback_name.c_str());
      return -1;
    }
    clean_working_queues(callback_name, &delete_struct->second);
    if (delete_struct->second.gemm_request != nullptr)
    {
      // pthread_mutex_lock(&delete_struct->second.gemm_request->pthread_mutex);
      delete_struct->second.gemm_request->ready = true;
      pthread_cond_signal(&delete_struct->second.gemm_request->pthread_cv);
      pthread_mutex_unlock(&delete_struct->second.gemm_request->pthread_mutex);
      pthread_mutex_destroy(&delete_struct->second.gemm_request->pthread_mutex);
      pthread_cond_destroy(&delete_struct->second.gemm_request->pthread_cv);
    }
    if (delete_struct->second.hist_request != nullptr)
    {
      // pthread_mutex_lock(&delete_struct->second.hist_request->pthread_mutex);
      delete_struct->second.hist_request->ready = true;
      pthread_cond_signal(&delete_struct->second.hist_request->pthread_cv);
      pthread_mutex_unlock(&delete_struct->second.hist_request->pthread_mutex);
      pthread_mutex_destroy(&delete_struct->second.hist_request->pthread_mutex);
      pthread_cond_destroy(&delete_struct->second.hist_request->pthread_cv);
    }
    if (delete_struct->second.vec_request != nullptr)
    {
      delete_struct->second.vec_request->ready = true;
      pthread_cond_signal(&delete_struct->second.vec_request->pthread_cv);
      pthread_mutex_unlock(&delete_struct->second.vec_request->pthread_mutex);
      pthread_mutex_destroy(&delete_struct->second.vec_request->pthread_mutex);
      pthread_cond_destroy(&delete_struct->second.vec_request->pthread_cv);
    }
    if (delete_struct->second.reduction_request != nullptr)
    {
      delete_struct->second.reduction_request->ready = true;
      pthread_cond_signal(&delete_struct->second.reduction_request->pthread_cv);
      pthread_mutex_unlock(&delete_struct->second.reduction_request->pthread_mutex);
      pthread_mutex_destroy(&delete_struct->second.reduction_request->pthread_mutex);
      pthread_cond_destroy(&delete_struct->second.reduction_request->pthread_cv);
    }
    if (delete_struct->second.tpu_request != nullptr)
    {
      delete_struct->second.tpu_request->ready = true;
      pthread_cond_signal(&delete_struct->second.tpu_request->pthread_cv);
      pthread_mutex_unlock(&delete_struct->second.tpu_request->pthread_mutex);
      pthread_mutex_destroy(&delete_struct->second.tpu_request->pthread_mutex);
      pthread_cond_destroy(&delete_struct->second.tpu_request->pthread_cv);
    }
    for (auto shmid : delete_struct->second.ids)
    {
      if (shmctl(shmid, IPC_RMID, NULL) == -1)
      {
        // RCLCPP_INFO(this->get_logger(), "Shctl failed, errno: %i", errno);
      }
    }
    for (auto key : delete_struct->second.keys)
    {
      for (std::vector<int>::iterator it = in_use_key_list.begin(); it != in_use_key_list.end(); it++)
      {
        if (*it == key)
        {
          in_use_key_list.erase(it); // remove key from use
          break;
        }
      }
    }
    auto struct_key = &delete_struct->second;
    if (struct_key->hist_request != nullptr)
    {
      if (shmdt(struct_key->hist_request) == -1)
      {
        // RCLCPP_INFO(this->get_logger(), "Shmdt failed, errno: %i", errno);
      }
    }
    if (struct_key->gemm_request != nullptr)
    {
      if (shmdt(struct_key->gemm_request) == -1)
      {
        // RCLCPP_INFO(this->get_logger(), "Shmdt failed, errno: %i", errno);
      }
    }
    if (struct_key->vec_request != nullptr)
    {
      if (shmdt(struct_key->vec_request) == -1)
      {
        // RCLCPP_INFO(this->get_logger(), "Shmdt failed, errno: %i", errno);
      }
    }
    if (struct_key->reduction_request != nullptr)
    {
      if (shmdt(struct_key->reduction_request) == -1)
      {
        // RCLCPP_INFO(this->get_logger(), "Shmdt failed, errno: %i", errno);
      }
    }
    if (struct_key->yolo_request != nullptr)
    {
      if (shmdt(struct_key->yolo_request) == -1)
      {
        // RCLCPP_INFO(this->get_logger(), "Shmdt failed, errno: %i", errno);
      }
    }
    if (struct_key->lane_request != nullptr)
    {
      if (shmdt(struct_key->lane_request) == -1)
      {
        // RCLCPP_INFO(this->get_logger(), "Shmdt failed, errno: %i", errno);
      }
    }
    if (struct_key->tpu_request != nullptr)
    {
      if (shmdt(struct_key->tpu_request) == -1)
      {
        // RCLCPP_INFO(this->get_logger(), "Shmdt failed, errno: %i", errno);
      }
    }
    return 0;
  }

  void runtime_monitor(void)
  {
    // RCLCPP_INFO(this->get_logger(), "Starting Runtime Monitor");
    while (rclcpp::ok())
    {
      std::this_thread::sleep_for(1000ms);
      // //RCLCPP_INFO(this->get_logger(), "Running Runtime Monitor");
      if (!rclcpp::ok())
      {
        // RCLCPP_INFO(this->get_logger(), "RCLCPP NOT OK");
        break;
      }
      struct stat sts;
      // callback_key_map_mutex.lock();
      if (this->callback_key_map.empty())
      {

        // callback_key_map_mutex.unlock();
        // RCLCPP_INFO(this->get_logger(), "Callback_key_map empty");
        continue;
      }

      for (auto key = this->callback_key_map.begin(); key != this->callback_key_map.end();)
      {
        std::string callback_name = key->first;
        auto pid = key->second.pid;
        std::string pid_key = "/proc/" + std::to_string(pid);
        if (stat(pid_key.c_str(), &sts) == -1 && errno == ENOENT)
        {
          RCLCPP_INFO(this->get_logger(),
                      "Runtime_Monitor: Client PID %s associated with callback %s does not exist anymore, detaching "
                      "from shared memory regions",
                      pid_key.c_str(), callback_name.c_str());
          if (!detach_mem_from_callback_name(callback_name))
          {
            key = this->callback_key_map.erase(key);
          }
          else
          {
            key++;
          }
        }
        else
        {
          key++;
        }
      }
    }
    // callback_key_map_mutex.unlock();
    this->clean_shared_memory();
    // RCLCPP_INFO(this->get_logger(), "Runtime Monitor Thread is Dead");
  }

  void kill_timer(void)
  {
    this->timer_->cancel();
  }
  std::string model_file = "/home/aamf/Research/AAMF-RTAS/src/aamf_server/test_data/inception_v2_224_quant_edgetpu.tflite";
  std::string label_file = "/home/aamf/Research/AAMF-RTAS/src/aamf_server/test_data/imagenet_labels.txt.1";
  std::string input_file = "/home/aamf/Research/AAMF-RTAS/src/aamf_server/test_data/resized_cat.bmp";
  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
  std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context;
  tflite::ops::builtin::BuiltinOpResolver resolver;

  void init_tpu(void)
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
    const auto *input_tensor = interpreter->input_tensor(0);
  }
  void invoke_tpu(struct tpu_struct *shared_request, int bucket, bool *in_use)
  {
    std::memcpy(interpreter->typed_input_tensor<uint8_t>(0), &shared_request->request.image[0], 224 * 224 * 3);
    // std::copy(&shared_request->request.image[0], &shared_request->request.image[224*224*3 -1], interpreter->typed_input_tensor<uint8_t>(0));
    //  Run inference.
    if (interpreter->Invoke() != kTfLiteOk)
    {
      std::cerr << "Cannot invoke interpreter" << std::endl;
      return;
    }
    // Get interpreter output.
    auto results = Sort(Dequantize(*interpreter->output_tensor(0)), shared_request->request.threshold);
    // std::copy(results[0], results[3], &shared_request->response.result[0]);
    // std::memcpy(&shared_request->response.result[0], &results[0], 3* sizeof(std::pair<float, int>));
    for (int i = 0; i < 3; i++)
    {
      shared_request->response.result[i].first = results[i].first;
      shared_request->response.result[i].second = results[i].second;
    }
    shared_request->ready = true;
    pthread_cond_signal(&shared_request->pthread_cv);
    pthread_mutex_unlock(&shared_request->pthread_mutex);
    *in_use = false;
  }
  void use_tpu(void)
  {
    // Read image from file.
    int image_width = 0;
    int image_height = 0;
    int image_bpp = 0;
    auto image = ReadBmpImage(this->input_file.c_str(), &image_width, &image_height, &image_bpp);
    std::printf("Image size: %d x %d x %d. Image vector size: %d\n", image_width, image_height, image_bpp, image.size());
    auto labels = ReadLabels(this->label_file);
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
  static void *
  gpu_worker_thread_wrapper(void *arg)
  {
    struct ThreadArg *threadArg = static_cast<struct ThreadArg *>(arg);
    threadArg->server->gpu_worker_thread(threadArg->id);
    delete threadArg;
    return nullptr;
  }
  size_t count_;
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  // std::vector<callback_row> data;
  //   callback_row r1(220, 1, 2, 220, 0, 1, 98, 0, 0, 0);
  //   callback_row r2(0, 1, 2, 0, 0, 2, 99, 0, 0, 0);
  //   callback_row r3(220, 1, 2, 220, 1, 1, 96, 1, 1, 1);
  //   callback_row r4(0, 1, 2, 0, 1, 2, 97, 1, 1, 1);
  //   data.push_back(r1);
  //   data.push_back(r2);
  //   data.push_back(r3);
  //   data.push_back(r4);
  //   chainset test(data);
  //   if (test.schedulable())
  //   {
  //       printf("Chainset is Schedulable\n");
  //   }
  //   else
  //   {
  //       printf("Chainset is not Schedulable\n");
  //   }
  auto gpu_server = std::make_shared<AamfServer>();
  rclcpp::executors::SingleThreadedExecutor exec1;
  exec1.enable_callback_priority();
  exec1.set_executor_priority_cpu(99, 0);
  exec1.add_node(gpu_server);
  exec1.set_callback_priority(gpu_server->subscription_, 95);
  exec1.set_callback_priority(gpu_server->register_sub_, 98);
  exec1.set_callback_priority(gpu_server->admissions_sub_, 99);
  // RCLCPP_INFO(gpu_server->get_logger(), "Server Instantiated");
  std::thread spinThread1(&rclcpp::executors::SingleThreadedExecutor::spin_rt, &exec1);
  spinThread1.join();
  exec1.remove_node(gpu_server);
  // RCLCPP_INFO(gpu_server->get_logger(), "Server Killed");
  rclcpp::shutdown();
  return 0;
}
