cmake_minimum_required(VERSION 3.10)
project(cuda_mppi_controller CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall")

enable_language(CUDA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -O3 -Xcompiler -fPIC -arch=sm_86  ")

# Find Eigen3
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  tf
  geometry_msgs
  visualization_msgs
  ackermann_msgs
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS
)

include_directories(
  include ${catkin_INCLUDE_DIRS}
)

# Include directories
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${catkin_INCLUDE_DIRS})


set(CPP_SOURCES
    src/mppi_controller.cpp
    src/mppi_functions.cpp
)

set(CUDA_SOURCES
    src/mppi_controller.cu
    src/mppi_functions.cu
)

add_library(c_mppi_controller_lib STATIC ${CPP_SOURCES})
add_library(cuda_mppi_controller_lib STATIC ${CUDA_SOURCES})
target_compile_definitions(cuda_mppi_controller_lib PRIVATE USE_CUDA)
set_property(TARGET cuda_mppi_controller_lib PROPERTY CUDA_STANDARD 14)
set_property(TARGET cuda_mppi_controller_lib PROPERTY CUDA_STANDARD_REQUIRED ON)
target_link_libraries(cuda_mppi_controller_lib curand)

# Create executable
add_executable(c_mppi_controller src/node.cpp)
add_executable(cuda_mppi_controller src/node.cpp)

# Create benchmark executables
add_executable(c_mppi_benchmark src/benchmark_cpu.cpp)
add_executable(cuda_mppi_benchmark src/benchmark_cuda.cu)

# Link libraries
target_link_libraries(c_mppi_controller ${catkin_LIBRARIES} c_mppi_controller_lib Eigen3::Eigen TBB::tbb)
target_link_libraries(cuda_mppi_controller ${catkin_LIBRARIES} cuda_mppi_controller_lib Eigen3::Eigen TBB::tbb)
target_link_libraries(cuda_mppi_controller curand)

# Link benchmark executables
target_link_libraries(c_mppi_benchmark c_mppi_controller_lib Eigen3::Eigen TBB::tbb)
target_link_libraries(cuda_mppi_benchmark cuda_mppi_controller_lib Eigen3::Eigen TBB::tbb)
target_link_libraries(cuda_mppi_benchmark curand)
target_compile_definitions(cuda_mppi_benchmark PRIVATE USE_CUDA)

# Print build information
message(STATUS "CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "Eigen3 include dir: ${EIGEN3_INCLUDE_DIR}")
