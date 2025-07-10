#include "cuda_mppi_controller/mppi_controller.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

#include "cuda_mppi_controller/mppi_functions.h"

namespace mppi_controller {
MPPIController::MPPIController() {
#ifndef CUDAMODE
  // Initialize cost matrices (similar to original implementation)
  Q_ = Eigen::Matrix3d::Zero();
  Q_(0, 0) = Q_X;      // x position cost
  Q_(1, 1) = Q_Y;      // y position cost
  Q_(2, 2) = Q_THETA;  // heading cost

  R_ = Eigen::Matrix2d::Zero();
  R_(0, 0) = R_VEL;    // velocity cost
  R_(1, 1) = R_STEER;  // steering angle cost

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
  noise_dist_ = std::normal_distribution<double>(0.0, 0.5);
#endif
  std::cout << "Naive MPPI Controller (Ackermann) initialized with "
            << NUM_SAMPLES << " samples, horizon " << HORIZON << ", wheelbase "
            << WHEELBASE << "m, max steering " << MAX_STEERING << " rad"
            << std::endl;
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
      double vel_noise = noise_dist_(generator_) * MAX_VELOCITY;
      double steering_noise = noise_dist_(generator_) * MAX_STEERING;

      // If we have a previous optimal sequence, add perturbations to it
      perturbed_control[0] = optimal_control_sequence_[t][0] + vel_noise;
      perturbed_control[1] = optimal_control_sequence_[t][1] + steering_noise;
      // Apply control limits
      perturbed_control[0] =
          clamp(perturbed_control[0], -MAX_VELOCITY, MAX_VELOCITY);
      perturbed_control[1] =
          clamp(perturbed_control[1], -MAX_STEERING, MAX_STEERING);

      control_sequences_[i][t] = perturbed_control;
    }
  }
}

// Generate trajectories by forward simulation
void MPPIController::GenerateTrajectories() {
  for (int i = 0; i < NUM_SAMPLES; ++i) {
    trajectories_[i].clear();
    trajectories_[i].push_back(current_state_);

    State current = current_state_;
    double total_cost = 0.0;

    // Forward simulate trajectory
    for (int t = 1; t < HORIZON; ++t) {
      // Compute cost
      total_cost += ComputeStateCost(current, control_sequences_[i][t]);
      current = ForwardDynamics(current, control_sequences_[i][t]);
      trajectories_[i].push_back(current);
    }

    trajectory_costs_[i] = total_cost;
  }
}

// Compute state cost (similar to original implementation)
double MPPIController::ComputeStateCost(const State& state,
                                        const Control& control) {
  State error = state - target_state_;
  double state_cost = error.transpose() * Q_ * error;
  double control_cost = control.transpose() * R_ * control;
  return state_cost + control_cost;
}

// Compute optimal control using importance-weighted averaging
Control MPPIController::ComputeOptimalControl() {
  // Warm start the optimal control sequence
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
  // Generate perturbed control sequences
  GeneratePerturbedControls();

  // Generate trajectories
  GenerateTrajectories();

  // Find minimum cost for normalization
  double min_cost =
      *std::min_element(trajectory_costs_.begin(), trajectory_costs_.end());

  // Compute importance-weighted control
  std::vector<Control> weighted_controls(HORIZON, Control::Zero());
  double total_weights = 0.0;

  for (int i = 0; i < NUM_SAMPLES; ++i) {
    std::cout << "Trajectory cost: " << trajectory_costs_[i] << std::endl;
    std::cout << "Min cost: " << min_cost << std::endl;
    std::cout << "Trajectory cost - min cost: " << trajectory_costs_[i] - min_cost
              << std::endl;
    double weight = std::exp(-(trajectory_costs_[i] - min_cost) / LAMBDA);
    std::cout << "Weight: " << weight << std::endl;

    for (int t = 0; t < HORIZON; ++t) {
      weighted_controls[t] += control_sequences_[i][t] * weight;
    }
    total_weights += weight;

  }
  std::cout << "Min cost: " << min_cost << std::endl;
  std::cout << "Total weights: " << total_weights << std::endl;
  // Normalize and update optimal control sequence
  for (int t = 0; t < HORIZON; ++t) {
    if (total_weights > 0) {
      optimal_control_sequence_[t] = weighted_controls[t] / total_weights;
    } else {
      std::cout << "Abnormal total weights: " << total_weights << std::endl;
      optimal_control_sequence_[t] = Control::Zero();
    }
  }

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
