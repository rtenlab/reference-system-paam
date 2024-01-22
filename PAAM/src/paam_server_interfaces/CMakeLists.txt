cmake_minimum_required(VERSION 3.5)
project(paam_server_interfaces)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra )
endif()

set(ROS_VERSION $ENV{ROS_VERSION})
include_directories(include)

find_package(ros_environment REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(action_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/GPUrequest.action"
  "msg/GPURequest.msg"
  "msg/GPURegister.msg"
  "msg/Admissions.msg"
  "msg/CallbackRow.msg"
  "msg/AdmissionsResponse.msg"
  DEPENDENCIES std_msgs action_msgs
)

ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
