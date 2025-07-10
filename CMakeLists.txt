cmake_minimum_required(VERSION 3.10)
project(cuda_mppi_controller CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

# Find Eigen3
find_package(Eigen3 REQUIRED)
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

# Create executable
add_library(c_mppi_controller_lib STATIC src/mppi_controller.cpp src/mppi_functions.cpp)

add_executable(c_mppi_controller src/node.cpp)

# Link libraries
target_link_libraries(c_mppi_controller ${catkin_LIBRARIES} c_mppi_controller_lib Eigen3::Eigen)

# Print build information
message(STATUS "CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "Eigen3 include dir: ${EIGEN3_INCLUDE_DIR}")
