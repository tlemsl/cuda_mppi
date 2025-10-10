#define USE_CUDA
#include <float.h>

#include <cmath>

#include "cuda_mppi_controller/config.h"
#include "cuda_mppi_controller/mppi_functions.cuh"

namespace mppi_controller {
#ifdef ACKERMANN_MODEL

State ToHostState(const CudaState& state) {
  State result;
  result[0] = state.x;
  result[1] = state.y;
  result[2] = state.theta;
  return result;
}
Control ToHostControl(const CudaControl& control) {
  Control result;
  result[0] = control.velocity;
  result[1] = control.steering_angle;
  return result;
}
CudaState ToCudaState(const State& state) {
  return CudaState{state[0], state[1], state[2]};
}
CudaControl ToCudaControl(const Control& control) {
  return CudaControl{control[0], control[1]};
}
Trajectory ToTrajectory(const CudaTrajectory& trajectory) {
  Trajectory result;
  result.reserve(HORIZON);
  for (int i = 0; i < HORIZON; i++) {
    result.push_back(ToHostState(trajectory.states[i]));
  }
  return result;
}
CudaTrajectory ToCudaTrajectory(const Trajectory& trajectory) {
  CudaTrajectory cuda_trajectory;
  for (int i = 0; i < HORIZON; i++) {
    cuda_trajectory.states[i] = ToCudaState(trajectory[i]);
  }
  return cuda_trajectory;
}
__device__ inline CudaState ForwardDynamics(const CudaState& current_state,
                                            const CudaControl& control) {
  float x = current_state.x;
  float y = current_state.y;
  float theta = current_state.theta;

  float velocity = control.velocity;
  float steering = control.steering_angle;

  CudaState next_state;
  next_state.x = x + velocity * __cosf(theta) * DT;
  next_state.y = y + velocity * __sinf(theta) * DT;
  next_state.theta =
      theta + (velocity * __fdividef(__tanf(steering), WHEELBASE)) * DT;

  return next_state;
}

__device__ inline void ComputeStateCost(const CudaState& state,
                                        const CudaControl& control,
                                        const CudaState& target_state,
                                        float* cost) {
  float x = state.x;
  float y = state.y;
  float theta = state.theta;

  float velocity = control.velocity;
  float steering = control.steering_angle;

  *cost = 0.0f;
  *cost += Q_X * (x - target_state.x) * (x - target_state.x);
  *cost += Q_Y * (y - target_state.y) * (y - target_state.y);
  *cost +=
      Q_THETA * (theta - target_state.theta) * (theta - target_state.theta);

  *cost += R_VEL * velocity * velocity;
  *cost += R_STEER * steering * steering;
}

__global__ void kernel_GeneratePerturbedControlsWithCuRAND(
    CudaControl* perturbed_controls, const CudaControl* base_controls,
    const float* random_numbers, int total_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_elements) {
    // Each control has 2 random numbers: velocity and steering
    float vel_noise =
        random_numbers[idx * 2] * MAX_VELOCITY * 0.5f;  // Scale noise
    float steering_noise = random_numbers[idx * 2 + 1] * MAX_STEERING * 0.5f;

    // Add perturbations to base control
    CudaControl perturbed = base_controls[idx];
    perturbed.velocity += vel_noise;
    perturbed.steering_angle += steering_noise;

    // Clamp to limits
    perturbed.velocity =
        fminf(fmaxf(perturbed.velocity, -MAX_VELOCITY), MAX_VELOCITY);
    perturbed.steering_angle =
        fminf(fmaxf(perturbed.steering_angle, -MAX_STEERING), MAX_STEERING);

    perturbed_controls[idx] = perturbed;
  }
}

__global__ void kernel_ComputeWeightedControls(
    const CudaControl* control_sequences, const float* weights,
    CudaControl* weighted_controls, float* total_weights, int num_samples,
    int horizon) {
  extern __shared__ float sdata[KERNEL_SIZE];

  int tid = threadIdx.x;
  int t = blockIdx.x;   // Time step
  int i = threadIdx.x;  // Sample index

  // Shared memory layout: [velocity_sum, steering_sum, weight_sum]
  float* vel_sum = sdata;
  float* steer_sum = sdata + blockDim.x;
  float* weight_sum = sdata + 2 * blockDim.x;

  // Initialize shared memory
  vel_sum[tid] = 0.0f;
  steer_sum[tid] = 0.0f;
  weight_sum[tid] = 0.0f;

  // Accumulate weighted controls for this time step
  for (int sample = i; sample < num_samples; sample += blockDim.x) {
    float weight = weights[sample];
    CudaControl control = control_sequences[sample * horizon + t];

    vel_sum[tid] += control.velocity * weight;
    steer_sum[tid] += control.steering_angle * weight;
    weight_sum[tid] += weight;
  }

  __syncthreads();

  // Parallel reduction
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      vel_sum[tid] += vel_sum[tid + s];
      steer_sum[tid] += steer_sum[tid + s];
      weight_sum[tid] += weight_sum[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    weighted_controls[t].velocity = vel_sum[0];
    weighted_controls[t].steering_angle = steer_sum[0];
    if (t == 0) {
      *total_weights = weight_sum[0];
    }
  }
}

__global__ void kernel_NormalizeControls(CudaControl* optimal_control_sequence,
                                         const CudaControl* weighted_controls,
                                         const float* total_weights,
                                         int horizon) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float total_weights_s;
  if (threadIdx.x == 0) {
    total_weights_s = *total_weights;
  }
  __syncthreads();
  if (t < horizon) {
    if (total_weights_s > 0.0f) {
      optimal_control_sequence[t].velocity =
          __fdividef(weighted_controls[t].velocity, total_weights_s);
      optimal_control_sequence[t].steering_angle =
          __fdividef(weighted_controls[t].steering_angle, total_weights_s);
    } else {
      optimal_control_sequence[t].velocity = 0.0f;
      optimal_control_sequence[t].steering_angle = 0.0f;
    }
  }
}
#endif

#ifdef QUADROTOR_MODEL


State ToHostState(const CudaState& state) {
  State result;
  result[0] = state.x;
  result[1] = state.y;
  result[2] = state.z;
  result[3] = state.qw;
  result[4] = state.qx;
  result[5] = state.qy;
  result[6] = state.qz;
  result[7] = state.vx;
  result[8] = state.vy;
  result[9] = state.vz;
  result[10] = state.wx;
  result[11] = state.wy;
  result[12] = state.wz;
  return result;
}
Control ToHostControl(const CudaControl& control) {
  Control result;
  result[0] = control.u1;
  result[1] = control.u2;
  result[2] = control.u3;
  result[3] = control.u4;
  return result;
}
CudaState ToCudaState(const State& state) {
  return CudaState{state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7], state[8], state[9], state[10], state[11], state[12]};
}
CudaControl ToCudaControl(const Control& control) {
  return CudaControl{control[0], control[1], control[2], control[3]};
}
Trajectory ToTrajectory(const CudaTrajectory& trajectory) {
  Trajectory result;
  result.reserve(HORIZON);
  for (int i = 0; i < HORIZON; i++) {
    result.push_back(ToHostState(trajectory.states[i]));
  }
  return result;
}
CudaTrajectory ToCudaTrajectory(const Trajectory& trajectory) {
  CudaTrajectory cuda_trajectory;
  for (int i = 0; i < HORIZON; i++) {
    cuda_trajectory.states[i] = ToCudaState(trajectory[i]);
  }
  return cuda_trajectory;
}

__device__ inline CudaState ForwardDynamics(const CudaState& current_state,
                                            const CudaControl& control) {
  float x = current_state.x;
  float y = current_state.y;
  float z = current_state.z;
  float qw = current_state.qw;
  float qx = current_state.qx;
  float qy = current_state.qy;
  float qz = current_state.qz;
  float vx = current_state.vx;
  float vy = current_state.vy;
  float vz = current_state.vz;
  float wx = current_state.wx;
  float wy = current_state.wy;
  float wz = current_state.wz;

  float u1 = control.u1;
  float u2 = control.u2;
  float u3 = control.u3;
  float u4 = control.u4;

  float f_z = K_THRUST * (u1 + u2 + u3 + u4);
}

__device__ inline void ComputeStateCost(const CudaState& state,
                                        const CudaControl& control,
                                        const CudaState& target_state,
                                        float* cost) {
  
  // cost variable should be in shared memory for fast access
  *cost = 0.0f;

  // Position cost
  *cost += Q_X * (state.x - target_state.x) * (state.x - target_state.x);
  *cost += Q_Y * (state.y - target_state.y) * (state.y - target_state.y);
  *cost += Q_Z * (state.z - target_state.z) * (state.z - target_state.z);
  // Orientation cost
  *cost += Q_QW * (state.qw - target_state.qw) * (state.qw - target_state.qw);
  *cost += Q_QX * (state.qx - target_state.qx) * (state.qx - target_state.qx);
  *cost += Q_QY * (state.qy - target_state.qy) * (state.qy - target_state.qy);
  *cost += Q_QZ * (state.qz - target_state.qz) * (state.qz - target_state.qz);
  // Velocity cost
  *cost += Q_VX * (state.vx - target_state.vx) * (state.vx - target_state.vx);
  *cost += Q_VY * (state.vy - target_state.vy) * (state.vy - target_state.vy);
  *cost += Q_VZ * (state.vz - target_state.vz) * (state.vz - target_state.vz);
  // Angular velocity cost
  *cost += Q_WX * (state.wx - target_state.wx) * (state.wx - target_state.wx);
  *cost += Q_WY * (state.wy - target_state.wy) * (state.wy - target_state.wy);
  *cost += Q_WZ * (state.wz - target_state.wz) * (state.wz - target_state.wz);

  // Control cost
  *cost += R_UN * (control.u1 - 0.5f) * (control.u1 - 0.5f);
  *cost += R_UN * (control.u2 - 0.5f) * (control.u2 - 0.5f);
  *cost += R_UN * (control.u3 - 0.5f) * (control.u3 - 0.5f);
  *cost += R_UN * (control.u4 - 0.5f) * (control.u4 - 0.5f);

}

__global__ void kernel_GeneratePerturbedControlsWithCuRAND(
    CudaControl* perturbed_controls, const CudaControl* base_controls,
    const float* random_numbers, int total_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_elements) {
    // Each control has 4 random numbers: u1, u2, u3, u4
    float u1_noise = random_numbers[idx * 4] * MAX_THRUST * 0.5f;
    float u2_noise = random_numbers[idx * 4 + 1] * MAX_THRUST * 0.5f;
    float u3_noise = random_numbers[idx * 4 + 2] * MAX_THRUST * 0.5f;
    float u4_noise = random_numbers[idx * 4 + 3] * MAX_THRUST * 0.5f;

    // Add perturbations to base control
    CudaControl perturbed = base_controls[idx];
    perturbed.u1 += u1_noise;
    perturbed.u2 += u2_noise;
    perturbed.u3 += u3_noise;
    perturbed.u4 += u4_noise;

    // Clamp to limits
    perturbed.u1 = fminf(fmaxf(perturbed.u1, -MAX_THRUST), MAX_THRUST);
    perturbed.u2 = fminf(fmaxf(perturbed.u2, -MAX_THRUST), MAX_THRUST);
    perturbed.u3 = fminf(fmaxf(perturbed.u3, -MAX_THRUST), MAX_THRUST);
    perturbed.u4 = fminf(fmaxf(perturbed.u4, -MAX_THRUST), MAX_THRUST);

    perturbed_controls[idx] = perturbed;
  }
}

__global__ void kernel_ComputeWeightedControls(
    const CudaControl* control_sequences, const float* weights,
    CudaControl* weighted_controls, float* total_weights, int num_samples,
    int horizon) {
  extern __shared__ float sdata[KERNEL_SIZE];

  int tid = threadIdx.x;
  int t = blockIdx.x;   // Time step
  int i = threadIdx.x;  // Sample index

  // Shared memory layout: [u1_sum, u2_sum, u3_sum, u4_sum, weight_sum]
  float* u1_sum = sdata;
  float* u2_sum = sdata + blockDim.x;
  float* u3_sum = sdata + 2 * blockDim.x;
  float* u4_sum = sdata + 3 * blockDim.x;
  float* weight_sum = sdata + 4 * blockDim.x; 

  // Initialize shared memory
  u1_sum[tid] = 0.0f;
  u2_sum[tid] = 0.0f;
  u3_sum[tid] = 0.0f;
  u4_sum[tid] = 0.0f;
  weight_sum[tid] = 0.0f;

  // Accumulate weighted controls for this time step
  for (int sample = i; sample < num_samples; sample += blockDim.x) {
    float weight = weights[sample];
    CudaControl control = control_sequences[sample * horizon + t];

    u1_sum[tid] += control.u1 * weight;
    u2_sum[tid] += control.u2 * weight;
    u3_sum[tid] += control.u3 * weight;
    u4_sum[tid] += control.u4 * weight;
    weight_sum[tid] += weight;
  }

  __syncthreads();

  // Parallel reduction
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      u1_sum[tid] += u1_sum[tid + s];
      u2_sum[tid] += u2_sum[tid + s];
      u3_sum[tid] += u3_sum[tid + s];
      u4_sum[tid] += u4_sum[tid + s];
      weight_sum[tid] += weight_sum[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    weighted_controls[t].u1 = u1_sum[0];
    weighted_controls[t].u2 = u2_sum[0];
    weighted_controls[t].u3 = u3_sum[0];
    weighted_controls[t].u4 = u4_sum[0];
    if (t == 0) {
      *total_weights = weight_sum[0];
    }
  }
}

__global__ void kernel_NormalizeControls(CudaControl* optimal_control_sequence,
                                         const CudaControl* weighted_controls,
                                         const float* total_weights,
                                         int horizon) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float total_weights_s;
  if (threadIdx.x == 0) {
    total_weights_s = *total_weights;
  }
  __syncthreads();
  if (t < horizon) {
    optimal_control_sequence[t].u1 =
        __fdividef(weighted_controls[t].u1, total_weights_s);
    optimal_control_sequence[t].u2 =
        __fdividef(weighted_controls[t].u2, total_weights_s);
    optimal_control_sequence[t].u3 =
        __fdividef(weighted_controls[t].u3, total_weights_s);
    optimal_control_sequence[t].u4 =
        __fdividef(weighted_controls[t].u4, total_weights_s);
  }
}

#endif

__global__ void kernel_GenerateTrajectoriesWithCost(
    CudaTrajectory* trajectories, float* trajectory_costs,
    const CudaState* current_state, const CudaControl* control_sequences,
    const CudaState* target_state) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float cost_s[KERNEL_SIZE];
  int kernel_idx = threadIdx.x;
  if (idx < NUM_SAMPLES) {
    // Each thread works on its own trajectory (no shared memory needed)
    CudaTrajectory trajectory;
    trajectory.states[0] = *current_state;
    cost_s[kernel_idx] = 0.0;

    for (int t = 1; t < HORIZON; ++t) {
      trajectory.states[t] = ForwardDynamics(
          trajectory.states[t - 1], control_sequences[idx * HORIZON + t]);
      ComputeStateCost(trajectory.states[t],
                       control_sequences[idx * HORIZON + t], *target_state,
                       cost_s + kernel_idx);
    }

    trajectory_costs[idx] = cost_s[kernel_idx];
    trajectories[idx] = trajectory;
  }
}

// Parallel reduction kernel to find minimum cost
__global__ void kernel_FindMinCost(const float* trajectory_costs,
                                   float* min_cost, int num_samples) {
  extern __shared__ float sdata[KERNEL_SIZE];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Load data into shared memory
  sdata[tid] = (i < num_samples) ? trajectory_costs[i] : FLT_MAX;
  __syncthreads();

  // Reduction in shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    min_cost[blockIdx.x] = sdata[0];
  }
}

__global__ void kernel_ComputeWeights(const float* trajectory_costs,
                                      float* weights, const float* min_cost,
                                      int num_samples) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float min_cost_s;
  if (threadIdx.x == 0) {
    min_cost_s = *min_cost;
  }
  __syncthreads();
  if (idx < num_samples) {
    weights[idx] = expf(-(trajectory_costs[idx] - min_cost_s) / LAMBDA);
  }
}

}  // namespace mppi_controller