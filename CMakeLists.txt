cmake_minimum_required(VERSION 3.10)
project(cuda_mppi_controller CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")
enable_language(CUDA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -O3 -Xcompiler -fPIC -arch=sm_86  -use_fast_math")
set(ROS_DISTRO ${ROS_DISTRO})

add_subdirectory(core)
if(ROS_DISTRO STREQUAL "noetic")
  add_subdirectory(ros/ros1)
endif()
if(ROS_DISTRO STREQUAL "humble")
  add_subdirectory(ros/ros2)
endif()
