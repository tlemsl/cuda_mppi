#include "cuda_mppi_controller/mppi_controller.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

#include "cuda_mppi_controller/mppi_functions.h"

namespace mppi_controller {
MPPIController::MPPIController() {
  // Initialize containers
  control_sequences_.resize(NUM_SAMPLES);
  trajectories_.resize(NUM_SAMPLES);
  trajectory_costs_.resize(NUM_SAMPLES);
  optimal_control_sequence_.resize(HORIZON);

  // Initialize states
  current_state_ = State::Zero();
  target_state_ = State::Zero();

  // Initialize random number generator
  generator_.seed(std::chrono::steady_clock::now().time_since_epoch().count());
  noise_dist_ = std::normal_distribution<float>(0.0, 0.5);
  std::cout << "CPP MPPI Controller (Ackermann) initialized with "
            << NUM_SAMPLES << " samples, horizon " << HORIZON << ", wheelbase "
            << WHEELBASE << "m, max steering " << MAX_STEERING << " rad"
            << std::endl;
}

MPPIController::~MPPIController() {
  // No special cleanup needed for CPU version
}

void MPPIController::SetCurrentState(const State& state) {
  current_state_ = state;
}

void MPPIController::SetTargetState(const State& target) {
  target_state_ = target;
}

// Generate perturbed control sequences
void MPPIController::GeneratePerturbedControls() {  
  for (int i = 0; i < NUM_SAMPLES; ++i) {
    control_sequences_[i].resize(HORIZON);

    for (int t = 0; t < HORIZON; ++t) {
      Control perturbed_control;

      // Generate random perturbations
      float vel_noise = noise_dist_(generator_) * MAX_VELOCITY;
      float steering_noise = noise_dist_(generator_) * MAX_STEERING;

      // If we have a previous optimal sequence, add perturbations to it
      perturbed_control[0] = optimal_control_sequence_[t][0] + vel_noise;
      perturbed_control[1] = optimal_control_sequence_[t][1] + steering_noise;
      // Apply control limits
      perturbed_control[0] =
          clamp<float>(perturbed_control[0], -MAX_VELOCITY, MAX_VELOCITY);
      perturbed_control[1] =
          clamp<float>(perturbed_control[1], -MAX_STEERING, MAX_STEERING);

      control_sequences_[i][t] = perturbed_control;
    }
  }
}

// Generate trajectories by forward simulation
void MPPIController::GenerateTrajectoriesWithCost() {
  auto start_time = std::chrono::high_resolution_clock::now();
  std::cout << "NUM_SAMPLES: " << NUM_SAMPLES << std::endl;
  std::cout << "HORIZON: " << HORIZON << std::endl;
  
  for (int i = 0; i < NUM_SAMPLES; ++i) {
    trajectories_[i].clear();
    trajectories_[i].push_back(current_state_);

    State current = current_state_;
    float total_cost = 0.0;

    for (int t = 1; t < HORIZON; ++t) {
      // Compute cost
      total_cost += ComputeStateCost(current, control_sequences_[i][t], target_state_);
      current = ForwardDynamics(current, control_sequences_[i][t]);
      trajectories_[i].push_back(current);
    }
    total_cost += ComputeStateCost(current, control_sequences_[i][HORIZON - 1], target_state_);

    trajectory_costs_[i] = total_cost;
  }
  
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  std::cout << "[CPU] GenerateTrajectoriesWithCost execution time: " << duration.count() << " microseconds" << std::endl;
}

Control MPPIController::ComputeOptimalControl() {
  auto start_time = std::chrono::high_resolution_clock::now();
  
  // Warm start the optimal control sequence
  auto warmstart_start = std::chrono::high_resolution_clock::now();
  if (optimal_control_sequence_.size() == HORIZON) {
    for (int t = 0; t < HORIZON - 1; ++t) {
      optimal_control_sequence_[t] = optimal_control_sequence_[t + 1];
    }
    optimal_control_sequence_[HORIZON - 1] = Control::Zero();
  }
  else {
    optimal_control_sequence_.resize(HORIZON);
    for (int t = 0; t < HORIZON; ++t) {
      optimal_control_sequence_[t] = Control::Zero();
    }
  }
  auto warmstart_end = std::chrono::high_resolution_clock::now();
  auto warmstart_duration = std::chrono::duration_cast<std::chrono::microseconds>(warmstart_end - warmstart_start);
  
  // Generate perturbed control sequences
  auto perturbed_start = std::chrono::high_resolution_clock::now();
  GeneratePerturbedControls();
  auto perturbed_end = std::chrono::high_resolution_clock::now();
  auto perturbed_duration = std::chrono::duration_cast<std::chrono::microseconds>(perturbed_end - perturbed_start);

  // Generate trajectories
  auto trajectories_start = std::chrono::high_resolution_clock::now();
  GenerateTrajectoriesWithCost();
  auto trajectories_end = std::chrono::high_resolution_clock::now();
  auto trajectories_duration = std::chrono::duration_cast<std::chrono::microseconds>(trajectories_end - trajectories_start);

  // Find minimum cost for normalization
  auto optimization_start = std::chrono::high_resolution_clock::now();
  float min_cost =
      *std::min_element(trajectory_costs_.begin(), trajectory_costs_.end());

  // Compute importance-weighted control
  std::vector<Control> weighted_controls(HORIZON, Control::Zero());
  float total_weights = 0.0;

  for (int i = 0; i < NUM_SAMPLES; ++i) {
    // std::cout << "Trajectory cost: " << trajectory_costs_[i] << std::endl;
    // std::cout << "Min cost: " << min_cost << std::endl;
    // std::cout << "Trajectory cost - min cost: " << trajectory_costs_[i] - min_cost
    //           << std::endl;
    float weight = std::exp(-(trajectory_costs_[i] - min_cost) / LAMBDA);
    // std::cout << "Weight: " << weight << std::endl;

    for (int t = 0; t < HORIZON; ++t) {
      weighted_controls[t] += control_sequences_[i][t] * weight;
    }
    total_weights += weight;

  }
  // std::cout << "Min cost: " << min_cost << std::endl;
  // std::cout << "Total weights: " << total_weights << std::endl;
  // Normalize and update optimal control sequence
  for (int t = 0; t < HORIZON; ++t) {
    if (total_weights > 0) {
      optimal_control_sequence_[t] = weighted_controls[t] / total_weights;
    } else {
      std::cout << "Abnormal total weights: " << total_weights << std::endl;
      optimal_control_sequence_[t] = Control::Zero();
    }
  }
  
  auto optimization_end = std::chrono::high_resolution_clock::now();
  auto optimization_duration = std::chrono::duration_cast<std::chrono::microseconds>(optimization_end - optimization_start);

  auto total_end_time = std::chrono::high_resolution_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end_time - start_time);
  std::cout << "--------------------------------" << std::endl;
  std::cout << "[CPU] Warm start time: " << warmstart_duration.count() << " microseconds" << std::endl;
  std::cout << "[CPU] Perturbed controls generation time: " << perturbed_duration.count() << " microseconds" << std::endl;
  std::cout << "[CPU] Trajectories generation time: " << trajectories_duration.count() << " microseconds" << std::endl;
  std::cout << "[CPU] Optimization time: " << optimization_duration.count() << " microseconds" << std::endl;
  std::cout << "[CPU] TOTAL ComputeOptimalControl execution time: " << total_duration.count() << " microseconds" << std::endl;
  std::cout << "--------------------------------" << std::endl;
  // Return first control action
  return optimal_control_sequence_[0];
}

// Get the optimal trajectory for visualization
const Trajectory& MPPIController::GetOptimalTrajectory() const {
  // Find the trajectory with minimum cost
  auto min_it =
      std::min_element(trajectory_costs_.begin(), trajectory_costs_.end());
  int min_idx = std::distance(trajectory_costs_.begin(), min_it);
  return trajectories_[min_idx];
}

// Get all sampled trajectories
const std::vector<Trajectory>& MPPIController::GetSampledTrajectories() const {
  return trajectories_;
}

void MPPIController::PrintStatus() const {
  std::cout << "Current state: [" << current_state_.transpose() << "]"
            << std::endl;
  std::cout << "Target state: [" << target_state_.transpose() << "]"
            << std::endl;
  if (!trajectory_costs_.empty()) {
    auto min_cost =
        *std::min_element(trajectory_costs_.begin(), trajectory_costs_.end());
    std::cout << "Minimum trajectory cost: " << min_cost << std::endl;
  }
}
};  // namespace mppi_controller
