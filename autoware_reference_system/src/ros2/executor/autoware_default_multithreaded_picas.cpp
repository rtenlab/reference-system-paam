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

#include "rclcpp/rclcpp.hpp"

#include "reference_system/system/systems.hpp"

#include "autoware_reference_system/autoware_system_builder.hpp"
#include "autoware_reference_system/system/timing/benchmark.hpp"
#include "autoware_reference_system/system/timing/default.hpp"

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);

  using TimeConfig = nodes::timing::Default;
  // uncomment for benchmarking
  // using TimeConfig = nodes::timing::BenchmarkCPUUsage;
  // set_benchmark_mode(true);

  auto nodes = create_autoware_nodes<RclcppSystem, TimeConfig>();

// rclcpp::executors::MultiThreadedExecutor executor;
#ifdef PICAS
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.enable_callback_priority();
  executor.cpus ={2, 3, 4, 5, 6, 7};
  executor.rt_attr.sched_policy = SCHED_FIFO;
  executor.rt_attr.sched_priority = 90;
  RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "PiCAS executor 1's rt-priority %d and CPU %d", executor.executor_priority, executor.executor_cpu);
  RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Executor: %ld", executor.get_number_of_threads());
#else

  rclcpp::executors::MultiThreadedExecutor executor;

#endif
  for (auto &node : nodes)
  {
    executor.add_node(node);
  }
#ifdef PICAS
  std::thread spinThread(&rclcpp::executors::MultiThreadedExecutor::spin, &executor);
  spinThread.join();
  for (auto &node : nodes)
  {
    executor.remove_node(node);
  }
#else
  executor.spin();
#endif
  nodes.clear();
  rclcpp::shutdown();

  return 0;
}
