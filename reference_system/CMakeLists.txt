cmake_minimum_required(VERSION 3.5)
project(reference_system
  VERSION 0.0.1
)
find_package(CUDA REQUIRED)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../AAMF/include)

# Add header-only library
add_library(${PROJECT_NAME} INTERFACE)
cuda_add_library(cuda_gemm SHARED include/reference_system/gpu_operations.cu)
target_link_libraries(${PROJECT_NAME} INTERFACE cuda_gemm)

target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

if(${BUILD_TESTING})
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# Install
install(TARGETS ${PROJECT_NAME} cuda_gemm
        EXPORT "export_${PROJECT_NAME}"
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION include
)

ament_export_targets("export_${PROJECT_NAME}")
ament_auto_package()

