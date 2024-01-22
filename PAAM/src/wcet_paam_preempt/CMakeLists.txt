cmake_minimum_required(VERSION 3.5)
project(wcet_aamf_preempt)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wno-reorder)
endif()

if(${PICAS})
  add_compile_options(-DPICAS)
endif()
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)
include_directories(../../include)
include_directories(../AAMF/include)
include_directories(../)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(test_msgs REQUIRED)
find_package(trace_picas REQUIRED)
find_package(aamf_server_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)

IF (Boost_FOUND)
    INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
    ADD_DEFINITIONS( "-DHAS_BOOST" )
ENDIF()

add_executable(case_study src/case_study.cpp)
ament_target_dependencies(case_study 
"rclcpp" 
"test_msgs" 
"trace_picas"
"aamf_server_interfaces"
"Boost"
"OpenCV"
"std_msgs")
target_link_libraries(case_study ${OpenCV_LIBS})

install(TARGETS
  case_study
  DESTINATION lib/${PROJECT_NAME})    

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

