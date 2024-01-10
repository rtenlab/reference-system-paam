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
#ifndef REFERENCE_SYSTEM__NODES__RCLCPP__COMMAND_HPP_
#define REFERENCE_SYSTEM__NODES__RCLCPP__COMMAND_HPP_

#include <chrono>
#include <string>

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

    class Command : public rclcpp::Node
    {
    public:
      explicit Command(const CommandSettings &settings)
          : Node(settings.node_name)
      {

#ifdef AAMF
        this->request_publisher_ = this->create_publisher<aamf_server_interfaces::msg::GPURequest>("request_topic", 1024);
        this->reg_publisher_ = this->create_publisher<aamf_server_interfaces::msg::GPURegister>("registration_topic", 1024);
        aamf_client_.push_back(std::make_shared<aamf_client_wrapper>(settings.callback_priority, settings.callback_priority, request_publisher_, reg_publisher_));
        this->register_sub_.push_back(this->create_subscription<aamf_server_interfaces::msg::GPURegister>("handshake_topic", 1024, std::bind(&Command::handshake_callback, this, std::placeholders::_1)));

        // this->register_sub_.push_back(this->create_subscription<aamf_server_interfaces::msg::GPURegister>("handshake_topic", 100,
        //[this](const aamf_server_interfaces::msg::GPURegister::SharedPtr msg)
        //{ this->handshake_callback(msg); }));
        register_sub_[0]->callback_priority = 99;
        aamf_client_[0]->register_subscriber(register_sub_[0]);
        aamf_client_[0]->send_handshake();

#endif
#ifdef DIRECT_INVOCATION
        di_gemm = std::make_shared<gemm_operator>();
#endif
        subscription_ = this->create_subscription<message_t>(
            settings.input_topic, 10,
            [this](const message_t::SharedPtr msg)
            { input_callback(msg); });
#ifdef PICAS
        subscription_->callback_priority = settings.callback_priority;
#endif
      }

    private:
#ifdef AAMF
      void handshake_callback(const aamf_server_interfaces::msg::GPURegister::SharedPtr msg)
      {
        aamf_client_[0]->handshake_callback(msg);
      }
#endif

      void input_callback(const message_t::SharedPtr input_message)
      {
        
#ifdef AAMF
        aamf_client_[0]->aamf_gemm_wrapper(true);
#endif

#ifdef DIRECT_INVOCATION
        di_gemm->gemm_wrapper();
#endif
        uint32_t missed_samples = get_missed_samples_and_update_seq_nr(input_message, sequence_number_);
        print_sample_path(this->get_name(), missed_samples, input_message);
      }

    private:
      rclcpp::Subscription<message_t>::SharedPtr subscription_;
      uint32_t sequence_number_ = 0;
#ifdef AAMF
      std::vector<std::shared_ptr<aamf_client_wrapper>> aamf_client_;
      rclcpp::Publisher<aamf_server_interfaces::msg::GPURequest>::SharedPtr request_publisher_;
      rclcpp::Publisher<aamf_server_interfaces::msg::GPURegister>::SharedPtr reg_publisher_;
      // rclcpp::Subscription<aamf_server_interfaces::msg::GPURegister>::SharedPtr register_sub_;
      std::vector<rclcpp::Subscription<aamf_server_interfaces::msg::GPURegister>::SharedPtr> register_sub_;
#endif
#ifdef DIRECT_INVOCATION
      std::shared_ptr<gemm_operator> di_gemm;
#endif
    };
  } // namespace rclcpp_system
} // namespace nodes
#endif // REFERENCE_SYSTEM__NODES__RCLCPP__COMMAND_HPP_
