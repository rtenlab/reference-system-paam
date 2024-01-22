#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <sys/time.h>
#include "case_study.h"
// For ROS2RTF
#include <unistd.h>
#include <sys/types.h>
#include <errno.h>
#include <sys/syscall.h>
#include <mutex>
#include <boost/uuid/uuid.hpp>            // uuid class
#include <boost/uuid/uuid_generators.hpp> // generators
#include <boost/uuid/uuid_io.hpp>         // streaming operators etc.
#include <boost/lexical_cast.hpp>
#include "trace_picas/trace.hpp"

#include "rclcpp/rclcpp.hpp"
// #include "std_msgs/msg/string.hpp"
#include "test_msgs/msg/detail/test_string__struct.hpp"
// #include "test_msgs/msg/TestString.msg"

using std::placeholders::_1;
// std::mutex mtx;

#define gettid() syscall(__NR_gettid)
#define USE_INTRA_PROCESS_COMMS false
// #define USE_INTRA_PROCESS_COMMS true

#ifdef OVERHEAD_DEBUG
std::ofstream ex_time;
log_struct *logger;
static struct timeval overhead_ctime;
std::string filename = "/home/aamf/Research/overhead/overhead_breakdown.csv";

static inline void log_time(std::string type)
{
  gettimeofday(&overhead_ctime, NULL);
  ex_time.open(filename, std::ios::app);
  ex_time << type << "," << overhead_ctime.tv_sec << "," << overhead_ctime.tv_usec << "\n";
  ex_time.close();
}

void attach_log_shm(void)
{
  int shmid = 65535;
  int logging_id = shmget(shmid, sizeof(log_struct), 0666 | IPC_CREAT);
  logger = (log_struct *)shmat(logging_id, (void *)0, 0);
}
#endif

#define DUMMY_LOAD_ITER 10000
int dummy_load_calib = 1;

void dummy_load(int load_ms)
{
  int i, j;
  for (j = 0; j < dummy_load_calib * load_ms; j++)
    for (i = 0; i < DUMMY_LOAD_ITER; i++)
      __asm__ volatile("nop");
}

using namespace std::chrono_literals;
using namespace std::placeholders;

class StartNode : public rclcpp::Node
{
public:
  StartNode(const std::string node_name, const std::string pub_topic, std::shared_ptr<trace::Trace> trace_ptr,
            int exe_time, int period, bool end_flag, int callback_priority, int kernel_id, int gemm_size)
      : Node(node_name, rclcpp::NodeOptions().use_intra_process_comms(USE_INTRA_PROCESS_COMMS)), count_(0), trace_callbacks_(trace_ptr), exe_time_(exe_time), period_(period), end_flag_(end_flag), callback_priority_(callback_priority), kernel_id_(kernel_id), gemm_size(gemm_size)
  {
    this->uuid = boost::uuids::random_generator()();
    std::vector<uint8_t> v(this->uuid.size());
    std::copy(this->uuid.begin(), this->uuid.end(), v.begin());
    std::copy_n(v.begin(), 16, uuid_array.begin());
    #ifdef OVERHEAD_DEBUG
    attach_log_shm();
    #endif
    unsigned int i = 0;

    for (i = 0; i < 16; i++)
    {
      std::printf("UUID_CHAR: %02x\n", uuid_array[i]);
    }

    // std::printf("UUID_CHAR: %04x\n", this->uuid_char);
    this->callback_priority = callback_priority_;
    this->request_publisher_ = this->create_publisher<aamf_server_interfaces::msg::GPURequest>("request_topic", 10);
    this->reg_publisher_ = this->create_publisher<aamf_server_interfaces::msg::GPURegister>("registration_topic", 10);
    std::this_thread::sleep_for(7000ms);
    this->register_sub_ = this->create_subscription<aamf_server_interfaces::msg::GPURegister>(
        "handshake_topic", 1000, std::bind(&StartNode::handshake_callback, this, std::placeholders::_1));
    this->pid = getpid();
    // std::string filename = "/home/aamf/Research/overhead/overhead_breakdown.csv";
    // ex_time.open(filename, std::ios::out);
    //  load_class_list();
    publisher_ = this->create_publisher<test_msgs::msg::TestString>(pub_topic, 1);
    this->send_handshake(this->get_name(), callback_priority_);
  }
  boost::uuids::uuid uuid;
  uint8_t *uuid_char;
  std::array<uint8_t, 16> uuid_array;
  // uint8_t uuid_char[16];
  //  std::vector<uint8_t> uuid_vector(16, 0);
  ~StartNode()
  {

    if (this->gemm_shm != nullptr)
      detach_gemm_shm(this->gemm_shm);

    if (this->hist_shm != nullptr)
      detach_hist_shm(this->hist_shm);

    if (this->yolo_shm != nullptr)
    {
      detach_yolo_shm(this->yolo_shm);
    }
    if (this->lane_shm != nullptr)
    {
      detach_lane_shm(this->lane_shm);
    }
    if (this->red_shm != nullptr)
    {
      detach_red_shm(this->red_shm);
    }
    if (this->tpu_shm != nullptr)
    {
      detach_tpu_shm(this->tpu_shm);
    }
  }
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<test_msgs::msg::TestString>::SharedPtr publisher_;
  void aamf_hist_wrapper(int size, int chain_priority, bool sleep)
  {
    if (!this->handshake_complete)
    {
      RCLCPP_INFO(this->get_logger(), "Handshake Not Complete");
      return;
    }
    this->send_hist_request(chain_priority, this->get_name());
    if (sleep)
    {
      this->sleep_on_hist_ready();
    }
    else
    {
      this->wait_on_hist_ready();
    }
  }
  void aamf_tpu_wrapper(int size, int chain_priority, bool sleep)
  {
    if (!this->handshake_complete)
    {
      RCLCPP_INFO(this->get_logger(), "Handshake Not Complete");
      return;
    }
    this->send_tpu_request(chain_priority, this->get_name());
    if (sleep)
    {
      this->sleep_on_tpu_ready();
    }
    else
    {
      this->wait_on_tpu_ready();
    }
  }
  void aamf_gemm_wrapper(int size, int chain_priority, bool sleep)
  {
    if (!this->handshake_complete)
    {
      RCLCPP_INFO(this->get_logger(), "Handshake Not Complete");
      return;
    }
#ifdef OVERHEAD_DEBUG
    // log_time("Message Building");
    logger->set_start();
#endif
    this->send_gemm_request(chain_priority, this->get_name());
#ifdef OVERHEAD_DEBUG
    // logger->set_end();
    // logger->log_latency("Message Sent");
    //  log_time("Message Sent and Process Sleep");
#endif
    if (sleep)
    {
      this->sleep_on_gemm_ready();
#ifdef OVERHEAD_DEBUG
      logger->set_end();
      logger->log_latency("Process Awakening");
      // log_time("Process Awakened and flag set");
#endif
    }
    else
    {
      this->wait_on_gemm_ready();
    }
  }
  void aamf_red_wrapper(int size, int chain_priority, bool sleep)
  {
    if (!this->handshake_complete)
    {
      RCLCPP_INFO(this->get_logger(), "Handshake Not Complete");
      return;
    }
    this->send_red_request(chain_priority, this->get_name());
    if (sleep)
    {
      this->sleep_on_red_ready();
    }
    else
    {
      this->wait_on_red_ready();
    }
  }
  void aamf_vec_wrapper(int size, int chain_priority, bool sleep)
  {
    if (!this->handshake_complete)
    {
      RCLCPP_INFO(this->get_logger(), "Handshake Not Complete");
      return;
    }
    this->send_vec_request(chain_priority, this->get_name());
    if (sleep)
    {
      this->sleep_on_vec_ready();
    }
    else
    {
      this->wait_on_vec_ready();
    }
  }
  void aamf_yolo_wrapper(int size, int chain_priority, bool sleep)
  {
    if (!this->handshake_complete)
    {
      RCLCPP_INFO(this->get_logger(), "Handshake Not Complete");
      return;
    }
    this->send_yolo_request(chain_priority, this->get_name());
    if (sleep)
    {
      this->sleep_on_yolo_ready();
    }
    else
    {
      this->wait_on_yolo_ready();
    }
  }

  void aamf_lane_wrapper(int size, int chain_priority, bool sleep)
  {
    if (!this->handshake_complete)
    {
      RCLCPP_INFO(this->get_logger(), "Handshake Not Complete");
      return;
    }
    this->send_lane_request(chain_priority, this->get_name());
    if (sleep)
    {
      this->sleep_on_lane_ready();
    }
    else
    {
      this->wait_on_lane_ready();
    }
  }
  void send_handshake(std::string callback_name, int chain_priority)
  {
    RCLCPP_INFO(this->get_logger(), "Sending Handshake");
    aamf_server_interfaces::msg::GPURegister message;
    message.should_register = true;
    message.pid = getpid();
    std_msgs::msg::String data;
    data.data = "HIST";
    message.kernels.push_back(data);
    data.data = "GEMM";
    message.kernels.push_back(data);
    // data.data = "YOLO";
    // message.kernels.push_back(data);
    // data.data = "LANE";
    // message.kernels.push_back(data);
    /*
    data.data = "RED";
    message.kernels.push_back(data);
    data.data = "VEC";
    message.kernels.push_back(data);
    */
    data.data = "TPU";
    message.kernels.push_back(data);
    message.priority = 0;                    // 1-99
    message.chain_priority = chain_priority; // 1-99
    // std::memcpy(&message.uuid[0], &uuid, sizeof(char));
    // message.uuid = *this->uuid_char;
    message.uuid = this->uuid_array;
    reg_publisher_->publish(message);
    RCLCPP_INFO(this->get_logger(), "Handshake Sent");
  }
  rclcpp::Subscription<aamf_server_interfaces::msg::GPURegister>::SharedPtr register_sub_;

private:
  size_t count_;
  int exe_time_;
  int period_;
  timeval ctime, ftime, create_timer, latency_time;
  bool end_flag_;
  std::shared_ptr<trace::Trace> trace_callbacks_;
  int callback_priority_;
  std::unordered_map<std::string, int> key_map;
  int pid;
  bool handshake_complete = false;
  int callback_priority;
  rclcpp::Publisher<aamf_server_interfaces::msg::GPURequest>::SharedPtr request_publisher_;
  rclcpp::Publisher<aamf_server_interfaces::msg::GPURegister>::SharedPtr reg_publisher_;
  struct hist_struct *hist_shm = nullptr;
  struct reduction_struct *red_shm = nullptr;
  struct vec_struct *vec_shm = nullptr;
  struct gemm_struct *gemm_shm = nullptr;
  struct yolo_struct *yolo_shm = nullptr;
  struct lane_struct *lane_shm = nullptr;
  struct tpu_struct *tpu_shm = nullptr;
  std::vector<std::string> class_list;

  int kernel_id_;
  cv::VideoCapture capture; //('/home/aamf/Documents/sample.mp4');
  void load_class_list()
  {
    std::ifstream ifs("/home/aamf/Documents/ros2_gpu_server/src/gpu_responder/net/classes.txt");
    std::string line;
    while (getline(ifs, line))
    {
      this->class_list.push_back(line);
    }
  }
  boost::uuids::uuid toBoostUUID(const std::array<uint8_t, 16> &arr)
  {
    boost::uuids::uuid uuid;
    std::copy(arr.begin(), arr.end(), uuid.begin());
    return uuid;
  }
  void handshake_callback(aamf_server_interfaces::msg::GPURegister::SharedPtr request)
  {
    // if (strcmp(request->callback_name.data.c_str(), this->get_name()))
    boost::uuids::uuid incoming_uuid = this->toBoostUUID(request->uuid);

    // memcpy(&incoming_uuid, &request->uuid, 16);
    if (incoming_uuid != uuid)
    {
      RCLCPP_INFO(this->get_logger(), "Handshake Not For Me");
      return;
    }

    for (unsigned long i = 0; i < request->keys.size(); i++)
    {
      key_map.insert(std::make_pair(request->kernels.at(i).data, request->keys.at(i)));
    }

    this->attach_to_shm();
    this->write_to_shm();

    this->handshake_complete = true;
    if (period_ == 16)
    {
      timer_ = this->create_wall_timer(16ms, std::bind(&StartNode::timer_callback, this));
    }
    else if (period_ == 70)
      timer_ = this->create_wall_timer(70ms, std::bind(&StartNode::timer_callback, this));
    else if (period_ == 80)
      timer_ = this->create_wall_timer(80ms, std::bind(&StartNode::timer_callback, this));
    else if (period_ == 100)
      timer_ = this->create_wall_timer(100ms, std::bind(&StartNode::timer_callback, this));
    else if (period_ == 120)
      timer_ = this->create_wall_timer(120ms, std::bind(&StartNode::timer_callback, this));
    else if (period_ == 160)
      timer_ = this->create_wall_timer(160ms, std::bind(&StartNode::timer_callback, this));
    else if (period_ == 200)
      timer_ = this->create_wall_timer(200ms, std::bind(&StartNode::timer_callback, this));
    else if (period_ == 1000)
      timer_ = this->create_wall_timer(1000ms, std::bind(&StartNode::timer_callback, this));
    else
      timer_ = this->create_wall_timer(10000ms, std::bind(&StartNode::timer_callback, this));

    gettimeofday(&create_timer, NULL);
    RCLCPP_INFO(this->get_logger(), "Create wall timer at %ld",
                create_timer.tv_sec * 1000 + create_timer.tv_usec / 1000);
    RCLCPP_INFO(this->get_logger(), "Server Confirmed Handshake for callback %s", request->callback_name.data.c_str());
  }
  void make_requests()
  {
    while (rclcpp::ok())
    {
      std::string name = this->get_name();
      RCLCPP_INFO(this->get_logger(), ("callback: " + name).c_str());
      int num = kernel_id_;
      switch (num)
      {
      case 0:
        this->aamf_hist_wrapper(1000000, callback_priority_, true);

        break;
      case 1:
        this->aamf_gemm_wrapper(1000000, callback_priority_, true);

        break;
      case 2:
        // this->make_yolo_goal(this->yolo_shm);
        // this->aamf_yolo_wrapper(1000000, callback_priority_, true);
        // this->yolo_verify(this->yolo_shm);
        break;
      case 3:
        this->aamf_red_wrapper(1000000, callback_priority_, true);
        break;
      case 4:
        this->aamf_vec_wrapper(1000000, callback_priority_, true);
        break;
      case 5:
        this->aamf_tpu_wrapper(1000000, callback_priority_, true);
        break;
      default:
        break;
      }
    }
  }
  int gemm_size;
  void write_to_shm()
  {
    this->make_gemm_goal(this->gemm_shm, this->gemm_size);
    //this->make_hist_goal(this->hist_shm);
    // this->capture.open("/home/aamf/Research/sample.mp4");
    // this->make_yolo_goal(this->yolo_shm);
    // this->make_lane_goal(this->lane_shm);
    //this->make_red_goal(this->red_shm);
    //this->make_vec_goal(this->vec_shm);
    this->make_tpu_goal(this->tpu_shm);
  }
  void attach_to_shm(void)
  {
    for (const auto &[kernel, key] : key_map)
    {
      if (kernel == "GEMM")
      {
        int gemm_shmid = shmget(key, sizeof(struct gemm_struct), 0666 | IPC_CREAT); // Get the shmid
        if (gemm_shmid == -1)
        {
          RCLCPP_INFO(this->get_logger(), "GEMM Shmget failed, errno: %i", errno);
        }
        this->gemm_shm = (struct gemm_struct *)shmat(gemm_shmid, (void *)0, 0);
        if (gemm_shm == (void *)-1)
        {
          RCLCPP_INFO(this->get_logger(), "GEMM Shmat failed, errno: %i", errno);
        }
      }
      else if (kernel == "HIST")
      {
        int hist_shmid = shmget(key, sizeof(struct hist_struct), 0666 | IPC_CREAT);
        if (hist_shmid == -1)
        {
          RCLCPP_INFO(this->get_logger(), "HIST Shmget failed, errno: %i", errno);
        }
        this->hist_shm = (struct hist_struct *)shmat(hist_shmid, (void *)0, 0);
        if (hist_shm == (void *)-1)
        {
          RCLCPP_INFO(this->get_logger(), "HIST Shmat failed, errno: %i", errno);
        }
      }
      else if (kernel == "RED")
      {
        int red_shmid = shmget(key, sizeof(struct reduction_struct), 0666 | IPC_CREAT);
        if (red_shmid == -1)
        {
          RCLCPP_INFO(this->get_logger(), "HIST Shmget failed, errno: %i", errno);
        }
        this->red_shm = (struct reduction_struct *)shmat(red_shmid, (void *)0, 0);
        if (red_shm == (void *)-1)
        {
          RCLCPP_INFO(this->get_logger(), "HIST Shmat failed, errno: %i", errno);
        }
      }
      else if (kernel == "VEC")
      {
        int vec_shmid = shmget(key, sizeof(struct vec_struct), 0666 | IPC_CREAT);
        if (vec_shmid == -1)
        {
          RCLCPP_INFO(this->get_logger(), "HIST Shmget failed, errno: %i", errno);
        }
        this->vec_shm = (struct vec_struct *)shmat(vec_shmid, (void *)0, 0);
        if (vec_shm == (void *)-1)
        {
          RCLCPP_INFO(this->get_logger(), "HIST Shmat failed, errno: %i", errno);
        }
      }
      else if (kernel == "YOLO")
      {
        int yolo_shmid = shmget(key, sizeof(struct yolo_struct), 0666 | IPC_CREAT); // Get the shmid
        if (yolo_shmid == -1)
        {
          RCLCPP_INFO(this->get_logger(), "YOLO Shmget failed, errno: %i", errno);
        }
        this->yolo_shm = (struct yolo_struct *)shmat(yolo_shmid, (void *)0, 0);
        if (yolo_shm == (void *)-1)
        {
          RCLCPP_INFO(this->get_logger(), "YOLO Shmat failed, errno: %i", errno);
        }
      }
      else if (kernel == "LANE")
      {
        int lane_shmid = shmget(key, sizeof(struct lane_struct), 0666 | IPC_CREAT); // Get the shmid
        if (lane_shmid == -1)
        {
          RCLCPP_INFO(this->get_logger(), "LANE Shmget failed, errno: %i", errno);
        }
        this->lane_shm = (struct lane_struct *)shmat(lane_shmid, (void *)0, 0);
        if (lane_shm == (void *)-1)
        {
          RCLCPP_INFO(this->get_logger(), "LANE Shmat failed, errno: %i", errno);
        }
      }
      else if (kernel == "TPU")
      {
        int tpu_shmid = shmget(key, sizeof(struct tpu_struct), 0666 | IPC_CREAT); // Get the shmid
        if (tpu_shmid == -1)
        {
          RCLCPP_INFO(this->get_logger(), "TPU Shmget failed, errno: %i", errno);
        }
        this->tpu_shm = (struct tpu_struct *)shmat(tpu_shmid, (void *)0, 0);
        if (tpu_shm == (void *)-1)
        {
          RCLCPP_INFO(this->get_logger(), "TPU Shmat failed, errno: %i", errno);
        }
      }
      else
      {
        RCLCPP_INFO(this->get_logger(), "Kernel Type Unsupported: %s", kernel.c_str());
      }
    }
  }
  void populateLoanedGEMMMessage(rclcpp::LoanedMessage<aamf_server_interfaces::msg::GPURequest> &loanedMsg, int chain_priority)
  {
    auto &message = loanedMsg.get();
    message.ready = false;
    message.open_cv = false;
    message.size = sizeof(*gemm_shm);
    message.priority = 1;
    message.kernel_id = 1;
    message.pid = this->pid;
    message.chain_priority = chain_priority;
    message.uuid = this->uuid_array;
    //printf("Message size: %d \n", sizeof(message));
  }
  void send_gemm_request(int chain_priority, std::string callback_name)
  {
    auto gemm_message = request_publisher_->borrow_loaned_message();
    this->populateLoanedGEMMMessage(gemm_message, chain_priority);
#ifdef OVERHEAD_DEBUG
    logger->set_end();
    logger->log_latency("Message Building");
    logger->set_start();
#endif
    request_publisher_->publish(std::move(gemm_message));

    // request_publisher_->publish(message);
  }
  void send_hist_request(int chain_priority, std::string callback_name)
  {
    auto hist_message = aamf_server_interfaces::msg::GPURequest();
    hist_message.ready = false;
    hist_message.open_cv = false;
    hist_message.size = sizeof(*hist_shm);
    hist_message.priority = 1;
    hist_message.kernel_id = 0;
    hist_message.pid = this->pid;
    hist_message.chain_priority = chain_priority;
    // std::memcpy(&uuid, hist_message.uuid, 16 * sizeof(char));
    // hist_message.uuid = this->uuid_char;
    hist_message.uuid = this->uuid_array;
    // hist_message.callback_name.data = callback_name;
    request_publisher_->publish(hist_message);
  }
  void send_vec_request(int chain_priority, std::string callback_name)
  {
    auto vec_message = aamf_server_interfaces::msg::GPURequest();
    vec_message.ready = false;
    vec_message.open_cv = false;
    vec_message.size = sizeof(*vec_shm);
    vec_message.priority = 1;
    vec_message.kernel_id = 4;
    vec_message.pid = this->pid;
    vec_message.chain_priority = chain_priority;
    vec_message.uuid = this->uuid_array;
    // vec_message.uuid = this->uuid_char;
    // std::memcpy(&uuid, vec_message.uuid, 16 * sizeof(char));
    //  vec_message.callback_name.data = callback_name;
    request_publisher_->publish(vec_message);
  }
  void send_red_request(int chain_priority, std::string callback_name)
  {
    auto red_message = aamf_server_interfaces::msg::GPURequest();
    red_message.ready = false;
    red_message.open_cv = false;
    red_message.size = sizeof(*red_shm);
    red_message.priority = 1;
    red_message.kernel_id = 5;
    red_message.pid = this->pid;
    red_message.chain_priority = chain_priority;
    // red_message.uuid = this->uuid_char;
    red_message.uuid = this->uuid_array;
    // std::memcpy(&uuid, red_message.uuid, 16 * sizeof(char));

    // red_message.callback_name.data = callback_name;
    request_publisher_->publish(red_message);
  }
  void send_yolo_request(int chain_priority, std::string callback_name)
  {
    auto yolo_message = aamf_server_interfaces::msg::GPURequest();
    yolo_message.open_cv = true;
    yolo_message.ready = false;
    yolo_message.size = sizeof(*yolo_shm);
    yolo_message.priority = 99;
    yolo_message.kernel_id = 2;
    yolo_message.pid = this->pid;
    yolo_message.chain_priority = chain_priority;
    // yolo_message.uuid = this->uuid_char;
    // std::memcpy(&uuid, yolo_message.uuid, 16 * sizeof(char));
    yolo_message.uuid = this->uuid_array;
    // yolo_message.callback_name.data = callback_name;
    request_publisher_->publish(yolo_message);
  }
  void populateLoanedTPUMessage(rclcpp::LoanedMessage<aamf_server_interfaces::msg::GPURequest> &loanedMsg, int chain_priority)
  {
    auto &tpu_message = loanedMsg.get();
    tpu_message.open_cv = false;
    tpu_message.tpu = true;
    tpu_message.ready = false;
    tpu_message.size = sizeof(*tpu_shm);
    tpu_message.priority = 99;
    tpu_message.kernel_id = 6;
    tpu_message.pid = this->pid;
    tpu_message.chain_priority = chain_priority;
    tpu_message.uuid = this->uuid_array;
  }
  void send_tpu_request(int chain_priority, std::string callback_name)
  {
    auto tpu_message = request_publisher_->borrow_loaned_message();
    this->populateLoanedTPUMessage(tpu_message, chain_priority);
    request_publisher_->publish(std::move(tpu_message));
  }

  void send_lane_request(int chain_priority, std::string callback_name)
  {
    auto lane_message = aamf_server_interfaces::msg::GPURequest();
    lane_message.open_cv = true;
    lane_message.ready = false;
    lane_message.size = sizeof(*lane_shm);
    lane_message.priority = 99;
    lane_message.kernel_id = 3;
    lane_message.pid = this->pid;
    lane_message.chain_priority = chain_priority;
    lane_message.uuid = this->uuid_array;
    // lane_message.uuid = this->uuid_char;
    //  lane_message.callback_name.data = callback_name;
    //  lane_message.dev_name.data = "/dev/video0";
    request_publisher_->publish(lane_message);
  }
  void wait_on_gemm_ready()
  {
    while (!this->gemm_shm->ready && rclcpp::ok())
      ;
    this->gemm_shm->ready = false;
  }
  void wait_on_vec_ready()
  {
    while (!this->vec_shm->ready && rclcpp::ok())
      ;
    this->vec_shm->ready = false;
  }
  void wait_on_red_ready()
  {
    while (!this->red_shm->ready && rclcpp::ok())
      ;
    this->red_shm->ready = false;
  }
  void wait_on_yolo_ready()
  {
    while (!this->yolo_shm->ready && rclcpp::ok())
      ;
    this->yolo_shm->ready = false;
  }
  void wait_on_lane_ready()
  {
    while (!this->lane_shm->ready && rclcpp::ok())
      ;
    this->lane_shm->ready = false;
  }
  void wait_on_hist_ready()
  {
    while (!this->hist_shm->ready && rclcpp::ok())
      ;
    this->hist_shm->ready = false;
  }

  void wait_on_tpu_ready()
  {
    while (!this->tpu_shm->ready && rclcpp::ok())
      ;
    this->tpu_shm->ready = false;
  }

  void sleep_on_gemm_ready()
  {
    pthread_mutex_lock(&this->gemm_shm->pthread_mutex);
    do
    {
      pthread_cond_wait(&this->gemm_shm->pthread_cv, &this->gemm_shm->pthread_mutex);
    } while (this->gemm_shm->ready == false);
    pthread_mutex_unlock(&this->gemm_shm->pthread_mutex);
    this->gemm_shm->ready = false;
  }
  void sleep_on_vec_ready()
  {
    pthread_mutex_lock(&this->vec_shm->pthread_mutex);
    pthread_cond_wait(&this->vec_shm->pthread_cv, &this->vec_shm->pthread_mutex);
    pthread_mutex_unlock(&this->vec_shm->pthread_mutex);
    this->vec_shm->ready = false;
  }
  void sleep_on_red_ready()
  {
    pthread_mutex_lock(&this->red_shm->pthread_mutex);
    pthread_cond_wait(&this->red_shm->pthread_cv, &this->red_shm->pthread_mutex);
    pthread_mutex_unlock(&this->red_shm->pthread_mutex);
    this->red_shm->ready = false;
  }
  void sleep_on_hist_ready()
  {
    pthread_mutex_lock(&this->hist_shm->pthread_mutex);
    pthread_cond_wait(&this->hist_shm->pthread_cv, &this->hist_shm->pthread_mutex);
    pthread_mutex_unlock(&this->hist_shm->pthread_mutex);
    this->hist_shm->ready = false;
  }
  void sleep_on_yolo_ready()
  {
    pthread_mutex_lock(&this->yolo_shm->pthread_mutex);
    pthread_cond_wait(&this->yolo_shm->pthread_cv, &this->yolo_shm->pthread_mutex);
    pthread_mutex_unlock(&this->yolo_shm->pthread_mutex);
    this->yolo_shm->ready = false;
  }
  void sleep_on_lane_ready()
  {
    pthread_mutex_lock(&this->lane_shm->pthread_mutex);
    pthread_cond_wait(&this->lane_shm->pthread_cv, &this->lane_shm->pthread_mutex);
    pthread_mutex_unlock(&this->lane_shm->pthread_mutex);
    this->lane_shm->ready = false;
  }
  void sleep_on_tpu_ready()
  {
    pthread_mutex_lock(&this->tpu_shm->pthread_mutex);
    pthread_cond_wait(&this->tpu_shm->pthread_cv, &this->tpu_shm->pthread_mutex);
    pthread_mutex_unlock(&this->tpu_shm->pthread_mutex);
    this->tpu_shm->ready = false;
  }
  void detach_gemm_shm(struct gemm_struct *gemm_shm)
  {
    int success = shmdt(gemm_shm);
    if (success == -1)
    {
      RCLCPP_INFO(this->get_logger(), "Shmdt failed, errno: %i", errno);
    }
  }
  void detach_hist_shm(struct hist_struct *hist_shm)
  {
    int success = shmdt(hist_shm);
    if (success == -1)
    {
      RCLCPP_INFO(this->get_logger(), "Shmdt failed, errno: %i", errno);
    }
  }
  void detach_red_shm(struct reduction_struct *red_shm)
  {
    int success = shmdt(red_shm);
    if (success == -1)
    {
      RCLCPP_INFO(this->get_logger(), "Shmdt failed, errno: %i", errno);
    }
  }
  void detach_vec_shm(struct vec_struct *vec_shm)
  {
    int success = shmdt(vec_shm);
    if (success == -1)
    {
      RCLCPP_INFO(this->get_logger(), "Shmdt failed, errno: %i", errno);
    }
  }
  void detach_yolo_shm(struct yolo_struct *yolo_shm)
  {
    int success = shmdt(yolo_shm);
    if (success == -1)
    {
      RCLCPP_INFO(this->get_logger(), "Shmdt failed, errno: %i", errno);
    }
  }
  void detach_lane_shm(struct lane_struct *lane_shm)
  {
    int success = shmdt(lane_shm);
    if (success == -1)
    {
      RCLCPP_INFO(this->get_logger(), "Shmdt failed, errno: %i", errno);
    }
  }
  void detach_tpu_shm(struct tpu_struct *tpu_shm)
  {
    int success = shmdt(tpu_shm);
    if (success == -1)
    {
      RCLCPP_INFO(this->get_logger(), "Shmdt failed, errno: %i", errno);
    }
  }

  void send_goal()
  {
    if (rclcpp::ok())
    {
      this->gemm_verify((float *)&gemm_shm->request.A_h, (float *)&gemm_shm->request.B_h, (float *)&gemm_shm->response.C_h,
                        gemm_shm->request.matArow, gemm_shm->request.matAcol, gemm_shm->request.matBcol);
    }

    if (rclcpp::ok())
    {
      this->hist_verify((unsigned int *)&hist_shm->request.in_h, (unsigned int *)&hist_shm->response.bins_h, 1000000,
                        4096);
    }
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
  std::string label_file = "/home/aamf/Research/AAMF-RTAS/src/aamf_server/test_data/imagenet_labels.1.txt";
  std::string input_file = "/home/aamf/Research/AAMF-RTAS/src/aamf_server/test_data/resized_cat.bmp";
  void make_tpu_goal(struct tpu_struct *goal_struct)
  {
    auto image = ReadBmpImage(this->input_file.c_str(), &goal_struct->request.image_width,
                              &goal_struct->request.image_height, &goal_struct->request.image_bpp);
    goal_struct->request.threshold = 0.1;
    std::memcpy(goal_struct->request.image, image.data(), image.size());
  }
  void make_gemm_goal(struct gemm_struct *goal_struct, int size)
  {
    unsigned matArow = size, matAcol = size;
    unsigned matBrow = size, matBcol = size;
    // unsigned matArow = 1000, matAcol = 1000;
    // unsigned matBrow = 1000, matBcol = 1000;
    goal_struct->request.A_sz = matArow * matAcol;
    goal_struct->request.B_sz = matBrow * matBcol;
    for (unsigned int i = 0; i < goal_struct->request.A_sz; i++)
    {
      goal_struct->request.A_h[i] = (float)(rand() % 100) / 100.00;
    }
    for (unsigned int i = 0; i < goal_struct->request.B_sz; i++)
    {
      goal_struct->request.B_h[i] = (float)(rand() % 100) / 100.00;
    }
    goal_struct->request.matArow = matArow;
    goal_struct->request.matAcol = matArow;
    goal_struct->request.matBcol = matBcol;
    goal_struct->request.matBrow = matBrow;
  }
  cv::Mat image;

  void make_yolo_goal(struct yolo_struct *yolo_struct)
  {
    if (!capture.isOpened())
    {
      std::cerr << "Error opening video file\n";
      return;
    }
    capture.read(this->image);
    yolo_struct->request.rows = image.rows;
    yolo_struct->request.cols = image.cols;
    yolo_struct->request.type = image.type();
    memcpy(yolo_struct->request.frame, image.data, image.total() * image.elemSize());
  }

  void make_lane_goal(struct lane_struct *lane_struct)
  {
    struct timeval ctime;
    gettimeofday(&ctime, NULL);
    lane_struct->request.current_timestamp = ctime;
    lane_struct->request.device_id = 0;
  }

  void make_hist_goal(struct hist_struct *goal_struct)
  {
    goal_struct->ready = false;
    goal_struct->request.nbins = 4096;
    for (unsigned int i = 0; i < 1000000; i++)
    {
      goal_struct->request.in_h[i] = (unsigned int)(rand() % 4096);
    }
  }
  void make_vec_goal(struct vec_struct *goal_struct)
  {
    goal_struct->ready = false;
    goal_struct->request.VecSize = 1000000;
    goal_struct->request.A_sz = 1000000;
    goal_struct->request.B_sz = 1000000;
    goal_struct->request.C_sz = 1000000;

    for (unsigned int i = 0; i < 1000000; i++)
    {
      float reg = (float)rand();
      goal_struct->request.A_h[i] = reg;
      goal_struct->request.B_h[i] = reg;
    }
  }

  void make_red_goal(struct reduction_struct *goal_struct)
  {
    goal_struct->ready = false;
    goal_struct->request.out_elements = 128;
    goal_struct->request.in_elements = 1000000;
    for (unsigned int i = 0; i < 1000000; i++)
    {
      goal_struct->request.in_h[i] = (float)rand();
    }
  }

  void gemm_verify(float *A, float *B, float *C, unsigned int m, unsigned int k, unsigned int n)
  {
    const float relativeTolerance = 1e-6;
    unsigned int count = 0;

    for (unsigned int row = 0; row < m; ++row)
    {
      for (unsigned int col = 0; col < n; ++col)
      {
        float sum = 0;
        for (unsigned int i = 0; i < k; ++i)
        {
          sum += A[row * k + i] * B[i * n + col];
        }
        count++;
        float relativeError = (sum - C[row * n + col]) / sum;
        // printf("%f/%f ", sum, C[row*n + col]);
        if (relativeError > relativeTolerance || relativeError < -relativeTolerance)
        {
          printf("\nTEST FAILED %u\n\n", count);
        }
      }
    }
    printf("TEST PASSED %u\n\n", count);
  }

  void hist_verify(unsigned int *input, unsigned int *bins, unsigned int num_elements, unsigned int num_bins)
  {
    // Initialize reference
    unsigned int *bins_ref = (unsigned int *)malloc(num_bins * sizeof(unsigned int));
    for (unsigned int binIdx = 0; binIdx < num_bins; ++binIdx)
    {
      bins_ref[binIdx] = 0;
    }

    // Compute reference bins
    for (unsigned int i = 0; i < num_elements; ++i)
    {
      unsigned int binIdx = input[i];
      ++bins_ref[binIdx];
    }

    // Compare to reference bins
    for (unsigned int binIdx = 0; binIdx < num_bins; ++binIdx)
    {
      // printf("%u: %u/%u, ", binIdx, bins_ref[binIdx], bins[binIdx]);
      if (bins[binIdx] != bins_ref[binIdx])
      {
        printf("TEST FAILED at bin %u, cpu = %u, gpu = %u\n\n", binIdx, bins_ref[binIdx], bins[binIdx]);
      }
    }
    printf("\nTEST PASSED\n");

    free(bins_ref);
  }
  int frame_count = 0;
  float fps = -1;
  int total_frames = 0;

  void yolo_verify(struct yolo_struct *yolo_struct)
  {
    static auto start = std::chrono::high_resolution_clock::now();
    int detections = yolo_struct->response.size;
    // cv::Mat image(yolo_struct->request.rows, yolo_struct->request.cols, yolo_struct->request.type, yolo_struct->request.frame);
    frame_count++;
    total_frames++;
    for (int i = 0; i < detections; ++i)
    {
      auto detection = yolo_struct->response.detections[i];
      auto box = detection.box;
      auto classId = detection.class_id;
      const auto color = colors[classId % colors.size()];
      cv::rectangle(this->image, box, color, 3);
      cv::rectangle(this->image, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
      cv::putText(this->image, class_list[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    if (frame_count >= 30)
    {

      auto end = std::chrono::high_resolution_clock::now();
      fps = frame_count * 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

      frame_count = 0;
      start = std::chrono::high_resolution_clock::now();
    }

    if (fps > 0)
    {

      std::ostringstream fps_label;
      fps_label << std::fixed << std::setprecision(2);
      fps_label << "FPS: " << fps;
      std::string fps_label_str = fps_label.str();

      cv::putText(image, fps_label_str.c_str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
    }
    cv::imshow("output", image);

    if (cv::waitKey(1) != -1)
    {
      // this->capture.release();
      std::cout << "finished by user\n";
    }
  }

  void
  dummy_task(int load)
  {
    int i;
    for (i = 0; i < load; i++)
      __asm__ volatile("nop");
  }

  void timer_callback()
  {
    std::string name = this->get_name();
    RCLCPP_INFO(this->get_logger(), ("callback: " + name).c_str());
    gettimeofday(&ctime, NULL);
    // trace_callbacks_->trace_write(name + "_in", std::to_string(ctime.tv_sec * 1000 + ctime.tv_usec / 1000));
    //  trace_callbacks_->trace_write_count(name+"_in",std::to_string(ctime.tv_sec*1000+ctime.tv_usec/1000),std::to_string(count_));

    // dummy_load(exe_time_);
    if (callback_priority == 0)
    {
      this->timer_->cancel();
      // this->make_requests();
      return;
    }
    int num = kernel_id_;
    switch (num)
    {
    case 0:
      this->aamf_hist_wrapper(1000000, callback_priority_, true);

      break;
    case 1:
      this->aamf_gemm_wrapper(1000000, callback_priority_, true);

      break;
    case 2:
      // this->make_yolo_goal(this->yolo_shm);
      // this->aamf_yolo_wrapper(1000000, callback_priority_, true);
      // this->yolo_verify(this->yolo_shm);
      break;
    case 3:
      this->aamf_red_wrapper(1000000, callback_priority_, true);
      break;
    case 4:
      this->aamf_vec_wrapper(1000000, callback_priority_, true);
      break;
    case 5:
      this->aamf_tpu_wrapper(1000000, callback_priority_, true);
      break;
    default:
      break;
    }
    auto message = test_msgs::msg::TestString();
    message.data = std::to_string(count_++);
    message.stamp.sec = ctime.tv_sec;
    message.stamp.usec = ctime.tv_usec;

    gettimeofday(&ftime, NULL);
    trace_callbacks_->trace_write(name + "_out", std::to_string((ftime.tv_sec * 1000 - ctime.tv_sec * 1000) +
                                                                (ftime.tv_usec / 1000 - ctime.tv_usec / 1000)));
    if (end_flag_)
    {
      gettimeofday(&ftime, NULL);
      latency_time.tv_sec = (ftime.tv_sec - message.stamp.sec);
      latency_time.tv_usec = (ftime.tv_usec - message.stamp.usec);
      trace_callbacks_->trace_write_count(
          name + "_latency", std::to_string(latency_time.tv_sec * 1000000 + latency_time.tv_usec), message.data);
    }
    publisher_->publish(message);
  }
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "PID: %ld run in ROS2.", gettid());
  int kernel_id = 0;
  std::string kernel = "";
  if (argc > 1)
  {
    kernel_id = atoi(argv[1]);
  }
  switch (kernel_id)
  {
  case 0:
    kernel = "HIST";
    break;
  case 1:
    kernel = "GEMM";
    break;
  case 3:
    kernel = "RED";
    break;
  case 4:
    kernel = "VEC";
    break;
  case 5:
    kernel = "TPU";
    break;
  }
  // Naive way to calibrate dummy workload for current system
  while (1)
  {
    timeval ctime, ftime;
    int duration_us;
    gettimeofday(&ctime, NULL);
    dummy_load(100); // 100ms
    gettimeofday(&ftime, NULL);
    duration_us = (ftime.tv_sec - ctime.tv_sec) * 1000000 + (ftime.tv_usec - ctime.tv_usec);
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "dummy_load_calib: %d (duration_us: %d ns)", dummy_load_calib,
                duration_us);
    if (abs(duration_us - 100 * 1000) < 500)
    { // error margin: 500us
      break;
    }
    dummy_load_calib = 100 * 1000 * dummy_load_calib / duration_us;
    if (dummy_load_calib <= 0)
      dummy_load_calib = 1;
  }

  timeval ctime;
  gettimeofday(&ctime, NULL);
  std::shared_ptr<trace::Trace> trace_callbacks = std::make_shared<trace::Trace>("/home/aamf/Research/data/wcet_t1_" + kernel + "_aamf.txt");
  trace_callbacks->trace_write("init", std::to_string(ctime.tv_sec * 1000 + ctime.tv_usec / 1000));
  auto task1 = std::make_shared<StartNode>("C1T_12_26", "task1", trace_callbacks, 2, 100, false, 99, kernel_id, 1000);
  rclcpp::executors::SingleThreadedExecutor exec1;
  exec1.enable_callback_priority();
  exec1.set_executor_priority_cpu(90, 2);
  exec1.add_node(task1);
  exec1.set_callback_priority(task1->register_sub_, 99);
  exec1.set_callback_priority(task1->timer_, 98);
  std::shared_ptr<trace::Trace> trace_callbacks2 = std::make_shared<trace::Trace>("/home/aamf/Research/data/wcet_t2_" + kernel + "_aamf.txt");
  trace_callbacks2->trace_write("init", std::to_string(ctime.tv_sec * 1000 + ctime.tv_usec / 1000));
   auto task2 = std::make_shared<StartNode>("C2T_0_0", "task2", trace_callbacks2, 2, 100, false, 25, kernel_id, 5000);
   rclcpp::executors::SingleThreadedExecutor exec2;
   exec2.enable_callback_priority();
   exec2.set_executor_priority_cpu(90, 3);
   exec2.add_node(task2);
   exec2.set_callback_priority(task2->register_sub_, 25);
   exec2.set_callback_priority(task2->timer_, 25);
  std::thread spinThread1(&rclcpp::executors::SingleThreadedExecutor::spin_rt, &exec1);
   std::thread spinThread2(&rclcpp::executors::SingleThreadedExecutor::spin_rt, &exec2);
  spinThread1.join();
  spinThread2.join();
  exec1.remove_node(task1);
   exec2.remove_node(task2);

  rclcpp::shutdown();
  return 0;
}
