#ifndef CONFIG_H
#define CONFIG_H

// Define robot models (Ackermann, Bicycle, Quadrotor

// #define ACKERMANN_MODEL
#define QUADROTOR_MODEL

#include <Eigen/Dense>
#include <algorithm>
#include <vector>

// Basic types similar to the original implementation
#ifdef ACKERMANN_MODEL
using State = Eigen::Vector3f;    // [x, y, theta]
using Control = Eigen::Vector2f;  // [velocity, steering_angle]
using Trajectory = std::vector<State>;

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

#endif

#ifdef QUADROTOR_MODEL
// position, orientation, velocity on Inertial frame
// Angular velocity is on Body frame
using State = Eigen::Matrix<float, 13, 1>;  // [x, y, z, qw, qx, qy, qz, vx, vy,
                                            // vz, wx, wy, wz]
using Control = Eigen::Vector4f;            // [u1, u2, u3, u4] 0<= u_n <= 1
// thrust 

using Trajectory = std::vector<State>;

// Cost matrices
#define Q_X 1.0f
#define Q_Y 1.0f
#define Q_Z 1.0f

// Hamiltonian Notation of Quaternion
#define Q_QW 1.0f
#define Q_QX 1.0f
#define Q_QY 1.0f
#define Q_QZ 1.0f

#define Q_VX 1.0f
#define Q_VY 1.0f
#define Q_VZ 1.0f

#define Q_WX 1.0f
#define Q_WY 1.0f
#define Q_WZ 1.0f

#define R_UN 0.001f

// #define R_FZ 0.001f
// #define R_TAU_X 0.0001f
// #define R_TAU_Y 0.0001f
// #define R_TAU_Z 0.0001f

// Vehicle parameters
#define ARM_LENGTH 0.3f
#define MASS 1.0f
#define INERTIA_XX 1.0f
#define INERTIA_YY 1.0f
#define INERTIA_ZZ 1.0f

// Control limits
#define MAX_THRUST 10.0f

#define G 9.81f // gravity

// Inertia matrix and its inverse for the quadrotor
inline Eigen::Matrix<float, 3, 3> getInertiaMatrix() {
    Eigen::Matrix<float, 3, 3> m;
    m << INERTIA_XX, 0, 0,
         0, INERTIA_YY, 0,
         0, 0, INERTIA_ZZ;
    return m;
}

inline Eigen::Matrix<float, 3, 3> getInertiaMatrixInverse() {
    Eigen::Matrix<float, 3, 3> m_inv;
    m_inv << 1.0f / INERTIA_XX, 0, 0,
             0, 1.0f / INERTIA_YY, 0,
             0, 0, 1.0f / INERTIA_ZZ;
    return m_inv;
}

#define K_THRUST 1.0f
#define K_TORQUE 1.0f

#endif

// Controller parameters
#define HORIZON 40
#define NUM_SAMPLES 4096
#define DT 0.1f
#define LAMBDA 10.0f

#define KERNEL_SIZE 512

// Helper function for clamping values (C++17 std::clamp alternative)
template <typename T>
T clamp(const T& value, const T& min_val, const T& max_val) {
  return std::max(min_val, std::min(value, max_val));
}

#ifdef USE_CUDA
#include <cuda_runtime.h>

#ifdef ACKERMANN_MODEL

struct CudaState {
  float x;
  float y;
  float theta;
};

struct CudaControl {
  float velocity;
  float steering_angle;
};
#endif

#ifdef QUADROTOR_MODEL
struct CudaState {
  float x;
  float y;
  float z;
  float qw;
  float qx;
  float qy;
  float qz;
  float vx;
  float vy;
  float vz;
  float wx;
  float wy;
  float wz;
};

struct CudaControl {
  float u1;
  float u2;
  float u3;
  float u4;
};
#endif

struct CudaTrajectory {
  CudaState states[HORIZON];
};

#endif
#endif
