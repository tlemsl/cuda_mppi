#define USE_CUDA
#include "cuda_mppi_controller/config.h"
#include "cuda_mppi_controller/mppi_functions.cuh"
#include <cmath>

namespace mppi_controller {
  State ToHostState(const CudaState& state){
    State result;
    result[0] = state.x;
    result[1] = state.y;
    result[2] = state.theta;
    return result;
}
Control ToHostControl(const CudaControl& control){
    Control result;
    result[0] = control.velocity;
    result[1] = control.steering_angle;
    return result;
}
CudaState ToCudaState(const State& state){
    return CudaState{state[0], state[1], state[2]};
}
CudaControl ToCudaControl(const Control& control){
    return CudaControl{control[0], control[1]};
}
Trajectory ToTrajectory(const CudaTrajectory& trajectory){
    Trajectory result;
    result.reserve(HORIZON);
    for (int i = 0; i < HORIZON; i++){
        result.push_back(ToHostState(trajectory.states[i]));
    }
    return result;
}
CudaTrajectory ToCudaTrajectory(const Trajectory& trajectory){
    CudaTrajectory cuda_trajectory;
    for (int i = 0; i < HORIZON; i++){
        cuda_trajectory.states[i] = ToCudaState(trajectory[i]);
    }
    return cuda_trajectory;
}
__device__ inline CudaState ForwardDynamics(const CudaState& current_state, const CudaControl& control) {
    float x = current_state.x;
    float y = current_state.y;
    float theta = current_state.theta;

    float velocity = control.velocity;
    float steering = control.steering_angle;

    CudaState next_state;
    next_state.x = x + velocity * __cosf(theta) * DT;
    next_state.y = y + velocity * __sinf(theta) * DT;
    next_state.theta = theta + (velocity * __tanf(steering) / WHEELBASE) * DT;

    return next_state;
}

__device__ inline float ComputeStateCost(const CudaState& state, const CudaControl& control, const CudaState& target_state) {
    float x = state.x;
    float y = state.y;
    float theta = state.theta;

    float velocity = control.velocity;
    float steering = control.steering_angle;

    float cost = 0.0;
    cost += Q_X * (x - target_state.x) * (x - target_state.x);
    cost += Q_Y * (y - target_state.y) * (y - target_state.y);
    cost += Q_THETA * (theta - target_state.theta) * (theta - target_state.theta);

    cost += R_VEL * velocity * velocity;
    cost += R_STEER * steering * steering;  

    return cost;
}

__global__ void kernel_GeneratePerturbedControls(
    CudaControl* perturbed_controls, const CudaControl* base_controls, const CudaControl* random_controls) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < NUM_SAMPLES * HORIZON) {
    // Generate random perturbations
    float vel_noise = random_controls[idx].velocity;
    float steering_noise = random_controls[idx].steering_angle;

    // Add perturbations to base control
    CudaControl perturbed = base_controls[idx];
    perturbed.velocity += vel_noise;
    perturbed.steering_angle += steering_noise;

    // Clamp to limits
    perturbed.velocity = fminf(fmaxf(perturbed.velocity, -MAX_VELOCITY), MAX_VELOCITY);
    perturbed.steering_angle = fminf(fmaxf(perturbed.steering_angle, -MAX_STEERING), MAX_STEERING);

    perturbed_controls[idx] = perturbed;
  }
}

__global__ void kernel_GenerateTrajectoriesWithCost(
    CudaTrajectory* trajectories, float* trajectory_costs, const CudaState* current_state,
    const CudaControl* control_sequences, const CudaState* target_state) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ CudaTrajectory trajectory;
  if (idx < NUM_SAMPLES) {
    trajectory.states[0] = *current_state;
    float total_cost = 0.0;
    for (int t = 1; t < HORIZON - 1; ++t) {
      trajectory.states[t] = ForwardDynamics(trajectory.states[t - 1], control_sequences[idx * HORIZON + t]);
      total_cost += ComputeStateCost(trajectory.states[t], control_sequences[idx * HORIZON + t], *target_state);
    }
    total_cost += ComputeStateCost(trajectory.states[HORIZON - 1], control_sequences[idx * HORIZON + HORIZON - 1], *target_state);
    trajectory_costs[idx] = total_cost;
    trajectories[idx] = trajectory;
  }
}


}  // namespace mppi_controller