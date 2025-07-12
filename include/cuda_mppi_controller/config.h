#ifndef CONFIG_H
#define CONFIG_H

// #define CUDAMODE to use CUDA implementation
// #define CUDAMODE

#include <Eigen/Dense>
#include <algorithm>
#include <vector>


// Basic types similar to the original implementation
using State = Eigen::Vector3f;    // [x, y, theta]
using Control = Eigen::Vector2f;  // [velocity, steering_angle]
using Trajectory = std::vector<State>;


// Controller parameters
#define HORIZON 20
#define NUM_SAMPLES 8192
#define DT 0.05f
#define LAMBDA 10.0f

// Cost matrices
#define Q_X 1.0f
#define Q_Y 1.0f
#define Q_THETA 0.1f

#define R_VEL 0.001f
#define R_STEER 0.0001f

// Vehicle parameters
#define WHEELBASE 0.3f
#define MAX_VELOCITY 3.5f
#define MAX_STEERING 0.52f

#define KERNEL_SIZE 512

// Helper function for clamping values (C++17 std::clamp alternative)
template <typename T>
T clamp(const T& value, const T& min_val, const T& max_val) {
  return std::max(min_val, std::min(value, max_val));
}

#ifdef USE_CUDA
#include <cuda_runtime.h>
// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>
// #include <thrust/generate.h>
// #include <thrust/copy.h>
// #include <thrust/random.h>


struct CudaState {
    float x;
    float y;
    float theta;
  };
  
struct CudaControl {
float velocity;
float steering_angle;
};

struct CudaTrajectory {
  CudaState states[HORIZON];
};

#endif
#endif
