# Find Eigen3 and TBB
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)


# Include directories for Eigen3 and TBB
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(include)

# Print build information
message(STATUS "CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "Eigen3 include dir: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "ROS_DISTRO: ${ROS_DISTRO}")

set(CPP_SOURCES
    src/mppi_controller.cpp
    src/mppi_functions.cpp
)

set(CUDA_SOURCES
    src/mppi_controller.cu
    src/mppi_functions.cu
)

# build libraries
add_library(c_mppi_controller_lib STATIC ${CPP_SOURCES})
add_library(cuda_mppi_controller_lib STATIC ${CUDA_SOURCES})
# target_compile_definitions(cuda_mppi_controller_lib PRIVATE USE_CUDA)
set_property(TARGET cuda_mppi_controller_lib PROPERTY CUDA_STANDARD 14)
set_property(TARGET cuda_mppi_controller_lib PROPERTY CUDA_STANDARD_REQUIRED ON)
target_link_libraries(cuda_mppi_controller_lib curand)


# Create benchmark executables
add_executable(c_mppi_benchmark src/benchmark_cpu.cpp)
add_executable(cuda_mppi_benchmark src/benchmark_cuda.cu)

# Link benchmark executables
target_link_libraries(c_mppi_benchmark c_mppi_controller_lib Eigen3::Eigen TBB::tbb)
target_link_libraries(cuda_mppi_benchmark cuda_mppi_controller_lib Eigen3::Eigen TBB::tbb)
target_link_libraries(cuda_mppi_benchmark curand)
# target_compile_definitions(cuda_mppi_benchmark PRIVATE USE_CUDA)
