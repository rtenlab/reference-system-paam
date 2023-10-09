
#ifndef REFERENCE_SYSTEM__AAMF_WRAPPERS_HPP_
#define REFERENCE_SYSTEM__AAMF_WRAPPERS_HPP_

#include <chrono>
#include <cmath>
#include <vector>
#include <functional>
#include <memory>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include <sys/types.h>
#include <errno.h>
#include <sys/syscall.h>
#include <mutex>
#include <boost/uuid/uuid.hpp>            // uuid class
#include <boost/uuid/uuid_generators.hpp> // generators
#include <boost/uuid/uuid_io.hpp>         // streaming operators etc.
#include "aamf_server_interfaces/msg/gpu_request.hpp"
#include "aamf_server_interfaces/msg/gpu_register.hpp"
#include "aamf_structs.h"
#include <boost/lexical_cast.hpp>
#include "rclcpp/rclcpp.hpp"

class aamf_client_wrapper
{
public:
    aamf_client_wrapper(int callback_priority_, int chain_priority, rclcpp::Publisher<aamf_server_interfaces::msg::GPURequest>::SharedPtr request_publisher, rclcpp::Publisher<aamf_server_interfaces::msg::GPURegister>::SharedPtr reg_publisher)
        : callback_priority(callback_priority_), chain_priority(chain_priority), request_publisher_(request_publisher), reg_publisher_(reg_publisher)
    {
        this->pid = getpid();
        this->uuid = boost::uuids::random_generator()();
        std::vector<uint8_t> v(this->uuid.size());
        std::copy(this->uuid.begin(), this->uuid.end(), v.begin());
        std::copy_n(v.begin(), 16, uuid_array.begin());
    }

    ~aamf_client_wrapper()
    {
        this->detach_gemm_shm(this->gemm_shm);
        this->detach_tpu_shm(this->tpu_shm);
    }
    void aamf_tpu_wrapper(bool sleep)
    {
        if (!this->handshake_complete)
        {
            return;
        }
        this->send_tpu_request();
        if (sleep)
        {
            this->sleep_on_tpu_ready();
        }
        else
        {
            this->wait_on_tpu_ready();
        }
    }
    void aamf_gemm_wrapper(bool sleep)
    {
        if (!this->handshake_complete)
        {
            return;
        }
        this->send_gemm_request();
        if (sleep)
        {
            this->sleep_on_gemm_ready();
        }
        else
        {
            this->wait_on_gemm_ready();
        }
    }
    void send_handshake(void)
    {
        aamf_server_interfaces::msg::GPURegister message;
        message.should_register = true;
        message.pid = getpid();
        std_msgs::msg::String data;
        data.data = "GEMM";
        message.kernels.push_back(data);
        data.data = "TPU";
        message.kernels.push_back(data);
        message.priority = 0;                    // 1-99
        message.chain_priority = chain_priority; // 1-99
        message.callback_priority = callback_priority;
        message.uuid = uuid_array;
        reg_publisher_->publish(message);
    }
    void register_subscriber(rclcpp::Subscription<aamf_server_interfaces::msg::GPURegister>::SharedPtr register_sub)
    {
        this->register_sub_ = register_sub;
    }
    void handshake_callback(aamf_server_interfaces::msg::GPURegister::SharedPtr request)
    {
        boost::uuids::uuid incoming_uuid = this->toBoostUUID(request->uuid);
        //while(register_sub_ == nullptr);;

        if (incoming_uuid != uuid)
        {
            //std::printf("Handshake not for me\n");
            return;
        }

        for (unsigned long i = 0; i < request->keys.size(); i++)
        {
            key_map.insert(std::make_pair(request->kernels.at(i).data, request->keys.at(i)));
        }

        this->attach_to_shm();
        this->write_to_shm();
        std::printf("Handshake Complete\n");

        this->handshake_complete = true;
    }
    aamf_client_wrapper &operator=(const aamf_client_wrapper &other)
    {
        if (this != &other)
        {
            pid = other.pid;
            tpu_shm = other.tpu_shm;
            gemm_shm = other.gemm_shm;
            chain_priority = other.chain_priority;
            handshake_complete = other.handshake_complete;
            callback_priority = other.callback_priority;
            uuid = other.uuid;
            uuid_char = other.uuid_char;
            uuid_array = other.uuid_array;
            key_map = other.key_map;
            input_file = other.input_file;
            request_publisher_ = other.request_publisher_;
            reg_publisher_ = other.reg_publisher_;
            register_sub_ = other.register_sub_;

        }
        return *this;
    }
    aamf_client_wrapper(const aamf_client_wrapper &other)
    {
        pid = other.pid;
        tpu_shm = other.tpu_shm;
        gemm_shm = other.gemm_shm;
        chain_priority = other.chain_priority;
        handshake_complete = other.handshake_complete;
        callback_priority = other.callback_priority;
        uuid = other.uuid;
        uuid_char = other.uuid_char;
        uuid_array = other.uuid_array;
        key_map = other.key_map;
        input_file = other.input_file;
        request_publisher_ = other.request_publisher_;
        reg_publisher_ = other.reg_publisher_;
        register_sub_ = other.register_sub_;
    }
    rclcpp::Subscription<aamf_server_interfaces::msg::GPURegister>::SharedPtr register_sub_ = nullptr;

private:
    int pid;
    struct tpu_struct *tpu_shm = nullptr;
    struct gemm_struct *gemm_shm = nullptr;
    int chain_priority;
    bool handshake_complete = false;
    int callback_priority;
    boost::uuids::uuid uuid;
    uint8_t *uuid_char;
    std::array<uint8_t, 16> uuid_array;
    std::unordered_map<std::string, int> key_map;
    std::string input_file = "/home/aamf/Research/AAMF-RTAS/src/aamf_server/test_data/resized_cat.bmp";
    rclcpp::Publisher<aamf_server_interfaces::msg::GPURequest>::SharedPtr request_publisher_;
    rclcpp::Publisher<aamf_server_interfaces::msg::GPURegister>::SharedPtr reg_publisher_;

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
        tpu_message.chain_priority = this->chain_priority;
        tpu_message.uuid = this->uuid_array;
        tpu_message.callback_priority = this->callback_priority;

    }
    void send_tpu_request()
    {
        auto tpu_message = request_publisher_->borrow_loaned_message();
        this->populateLoanedTPUMessage(tpu_message, chain_priority);
        request_publisher_->publish(std::move(tpu_message));
    }
    size_t kBmpFileHeaderSize = 14;
    size_t kBmpInfoHeaderSize = 40;
    size_t kBmpHeaderSize = kBmpFileHeaderSize + kBmpInfoHeaderSize;

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
    void wait_on_gemm_ready()
    {
        while (!this->gemm_shm->ready)
            ;
        this->gemm_shm->ready = false;
    }

    void make_tpu_goal(struct tpu_struct *goal_struct)
    {
        auto image = ReadBmpImage(this->input_file.c_str(), &goal_struct->request.image_width,
                                  &goal_struct->request.image_height, &goal_struct->request.image_bpp);
        goal_struct->request.threshold = 0.1;
        std::memcpy(goal_struct->request.image, image.data(), image.size());
    }
    void wait_on_tpu_ready()
    {
        while (!this->tpu_shm->ready)
            ;
        this->tpu_shm->ready = false;
    }

    void make_gemm_goal(struct gemm_struct *goal_struct)
    {
        unsigned matArow = 500, matAcol = 500;
        unsigned matBrow = 500, matBcol = 500;
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
    void sleep_on_gemm_ready()
    {
        pthread_mutex_lock(&this->gemm_shm->pthread_mutex);
        pthread_cond_wait(&this->gemm_shm->pthread_cv, &this->gemm_shm->pthread_mutex);
        pthread_mutex_unlock(&this->gemm_shm->pthread_mutex);
        this->gemm_shm->ready = false;
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
            std::printf("Failed to detach from shared memory\n");
        }
    }
    void detach_tpu_shm(struct tpu_struct *tpu_shm)
    {
        int success = shmdt(tpu_shm);
        if (success == -1)
        {
            std::printf("Failed to detach from shared memory\n");
        }
    }

    boost::uuids::uuid toBoostUUID(const std::array<uint8_t, 16> &arr)
    {
        boost::uuids::uuid uuid;
        std::copy(arr.begin(), arr.end(), uuid.begin());
        return uuid;
    }

    void write_to_shm()
    {
        this->make_gemm_goal(this->gemm_shm);
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
                    std::printf("Failed to get shared memory\n");
                }
                this->gemm_shm = (struct gemm_struct *)shmat(gemm_shmid, (void *)0, 0);
                if (gemm_shm == (void *)-1)
                {
                    std::printf("Failed to attach to shared memory\n");
                }
            }
            else if (kernel == "TPU")
            {
                int tpu_shmid = shmget(key, sizeof(struct tpu_struct), 0666 | IPC_CREAT); // Get the shmid
                if (tpu_shmid == -1)
                {
                    std::printf("Failed to get shared memory\n");
                }
                this->tpu_shm = (struct tpu_struct *)shmat(tpu_shmid, (void *)0, 0);
                if (tpu_shm == (void *)-1)
                {
                    std::printf("Failed to attach to shared memory\n");
                }
            }
            else
            {
                std::printf("Unknown kernel name\n");
            }
        }
    }

    void populateLoanedGEMMMessage(rclcpp::LoanedMessage<aamf_server_interfaces::msg::GPURequest> &loanedMsg)
    {
        auto &message = loanedMsg.get();
        message.ready = false;
        message.open_cv = false;
        message.size = sizeof(*this->gemm_shm);
        message.priority = 1;
        message.kernel_id = 1;
        message.pid = this->pid;
        message.chain_priority = this->chain_priority;
        message.uuid = this->uuid_array;
        message.callback_priority = this->callback_priority;
    }

    void send_gemm_request(void)
    {
        auto gemm_message = request_publisher_->borrow_loaned_message();
        this->populateLoanedGEMMMessage(gemm_message);
        request_publisher_->publish(std::move(gemm_message));
    }
};
#endif // REFERENCE_SYSTEM__AAMF_WRAPPERS_HPP_