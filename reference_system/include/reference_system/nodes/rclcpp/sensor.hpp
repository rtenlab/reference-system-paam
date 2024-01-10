// Copyright 2021 Apex.AI, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef REFERENCE_SYSTEM__NODES__RCLCPP__SENSOR_HPP_
#define REFERENCE_SYSTEM__NODES__RCLCPP__SENSOR_HPP_
#include <chrono>
#include <string>
#include <utility>

#include "rclcpp/rclcpp.hpp"
#include "reference_system/nodes/settings.hpp"
#include "reference_system/sample_management.hpp"
#include "reference_system/msg_types.hpp"

#ifdef AAMF
#include "reference_system/aamf_wrappers.hpp"
#endif
#ifdef DIRECT_INVOCATION
#include "reference_system/gpu_operations.hpp"
#endif
namespace nodes
{
  namespace rclcpp_system
  {

    class Sensor : public rclcpp::Node
    {
    public:
      explicit Sensor(const SensorSettings &settings)
          : Node(settings.node_name)
      {

#ifdef AAMF
        this->request_publisher_ = this->create_publisher<aamf_server_interfaces::msg::GPURequest>("request_topic", 1024);
        this->reg_publisher_ = this->create_publisher<aamf_server_interfaces::msg::GPURegister>("registration_topic", 1024);
        //if(settings.node_name.compare("FrontLidarDriver") == 0){
       // std::this_thread::sleep_for(std::chrono::milliseconds(7000));
        //}
        this->aamf_client_ = std::make_shared<aamf_client_wrapper>(settings.callback_priority, settings.callback_priority, request_publisher_, reg_publisher_);
        this->register_sub_ = this->create_subscription<aamf_server_interfaces::msg::GPURegister>("handshake_topic", 1024, std::bind(&Sensor::handshake_callback, this, std::placeholders::_1));
        /*this->register_sub_ = this->create_subscription<aamf_server_interfaces::msg::GPURegister>("handshake_topic", 100, [this](const aamf_server_interfaces::msg::GPURegister::SharedPtr msg)
                                                                                                          { this->aamf_client_->handshake_callback(msg); });
        */
        // register_sub_->callback_priority = 99;

        aamf_client_->register_subscriber(register_sub_);
        aamf_client_->register_sub_->callback_priority = 99;
        aamf_client_->send_handshake();
#endif
#ifdef DIRECT_INVOCATION
        di_gemm = std::make_shared<gemm_operator>();
#endif
        publisher_ = this->create_publisher<message_t>(settings.topic_name, 1);
        timer_ = this->create_wall_timer(
            settings.cycle_time,
            [this]
            { timer_callback(); });
#ifdef PICAS
        timer_->callback_priority = settings.callback_priority;
#endif
      }

    private:
#ifdef AAMF
      void handshake_callback(const aamf_server_interfaces::msg::GPURegister::SharedPtr msg)
      {
        aamf_client_->handshake_callback(msg);
      }
#endif
      void timer_callback()
      {
        uint64_t timestamp = now_as_int();
        auto message = publisher_->borrow_loaned_message();
        message.get().size = 0;
#ifdef AAMF
        aamf_client_->aamf_gemm_wrapper(true);
#endif
#ifdef DIRECT_INVOCATION
        di_gemm->gemm_wrapper();
#endif
        set_sample(this->get_name(), sequence_number_++, 0, timestamp, message.get());

        publisher_->publish(std::move(message));
      }

    private:
      rclcpp::Publisher<message_t>::SharedPtr publisher_;
      rclcpp::TimerBase::SharedPtr timer_;
      uint32_t sequence_number_ = 0;
#ifdef AAMF
      std::shared_ptr<aamf_client_wrapper> aamf_client_;
      rclcpp::Publisher<aamf_server_interfaces::msg::GPURequest>::SharedPtr request_publisher_;
      rclcpp::Publisher<aamf_server_interfaces::msg::GPURegister>::SharedPtr reg_publisher_;
      // rclcpp::Subscription<aamf_server_interfaces::msg::GPURegister>::SharedPtr register_sub_;
      rclcpp::Subscription<aamf_server_interfaces::msg::GPURegister>::SharedPtr register_sub_;
#endif
#ifdef DIRECT_INVOCATION
      std::shared_ptr<gemm_operator> di_gemm;
#endif
    };

  } // namespace rclcpp_system
} // namespace nodes
#endif // REFERENCE_SYSTEM__NODES__RCLCPP__SENSOR_HPP_
