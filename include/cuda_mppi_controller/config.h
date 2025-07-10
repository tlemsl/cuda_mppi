#ifndef CONFIG_H
#define CONFIG_H

// #define CUDAMODE to use CUDA implementation
// #define CUDAMODE

#include <Eigen/Dense>
#include <algorithm>
#include <vector>

// Basic types similar to the original implementation
using State = Eigen::Vector3d;    // [x, y, theta]
using Control = Eigen::Vector2d;  // [velocity, steering_angle]
using Trajectory = std::vector<State>;

// Controller parameters
#define HORIZON 30
#define NUM_SAMPLES 1000
#define DT 0.05
#define LAMBDA 10.0

// Cost matrices
#define Q_X 1.0
#define Q_Y 1.0
#define Q_THETA 0.1

#define R_VEL 0.001
#define R_STEER 0.0001

// Vehicle parameters
#define WHEELBASE 0.3
#define MAX_VELOCITY 3.5
#define MAX_STEERING 0.52

// Helper function for clamping values (C++17 std::clamp alternative)
template <typename T>
T clamp(const T& value, const T& min_val, const T& max_val) {
  return std::max(min_val, std::min(value, max_val));
}
#endif
