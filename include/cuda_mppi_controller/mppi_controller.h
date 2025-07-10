#ifndef MPPI_CONTROLLER_H
#define MPPI_CONTROLLER_H

#include <Eigen/Dense>
#include <random>
#include <vector>

#include "config.h"

namespace mppi_controller {


class MPPIController {
 private:


  #ifndef CUDAMODE
  // C++ implementation
  // Cost matrices
  Eigen::Matrix3d Q_;  ///< State cost matrix [x, y, theta]
  Eigen::Matrix2d R_;  ///< Control cost matrix [velocity, steering]

  // Current state and target
  State current_state_;  ///< Current robot state [x, y, theta]
  State target_state_;   ///< Target robot state [x, y, theta]

  // Control sequences and trajectories
  std::vector<std::vector<Control>>
      control_sequences_;                 ///< Sampled control sequences
  std::vector<Trajectory> trajectories_;  ///< Simulated trajectories
  std::vector<double> trajectory_costs_;  ///< Cost for each trajectory
  std::vector<Control>
      optimal_control_sequence_;  ///< Current optimal control sequence

  // Random number generation
  std::mt19937 generator_;  ///< Random number generator
  std::normal_distribution<double>
      noise_dist_;  ///< Noise distribution for perturbations
  #endif

  #ifdef CUDAMODE
  // CUDA implementation
  #endif

 public:
  MPPIController();

  /**
   * @brief Set the current state of the robot
   * @param state Current state [x, y, theta]
   */
  void SetCurrentState(const State& state);

  /**
   * @brief Set the target state for the robot
   * @param target Target state [x, y, theta]
   */
  void SetTargetState(const State& target);

  /**
   * @brief Compute the optimal control action using MPPI
   *
   * This is the main MPPI computation that:
   * 1. Generates perturbed control sequences
   * 2. Simulates trajectories using Ackermann dynamics
   * 3. Evaluates trajectory costs
   * 4. Computes importance-weighted optimal control
   *
   * @return Control Optimal control [velocity, steering_angle]
   */
  Control ComputeOptimalControl();

  /**
   * @brief Get the optimal trajectory for visualization
   * @return const Trajectory& Reference to the best trajectory
   */
  const Trajectory& GetOptimalTrajectory() const;

  /**
   * @brief Get all sampled trajectories for visualization
   * @return const std::vector<Trajectory>& Reference to all trajectories
   */
  const std::vector<Trajectory>& GetSampledTrajectories() const;

  /**
   * @brief Print current controller status
   */
  void PrintStatus() const;

 private:
  /**
   * @brief Generate perturbed control sequences around the optimal sequence
   */
  void GeneratePerturbedControls();

  /**
   * @brief Generate trajectories by forward simulation
   */
  void GenerateTrajectories();

  /**
   * @brief Compute the cost for a given state and control
   *
   * Uses quadratic cost function:
   * cost = (state - target)^T * Q * (state - target) + control^T * R * control
   *
   * @param state Current state
   * @param control Applied control
   * @return double Total cost
   */
  double ComputeStateCost(const State& state, const Control& control);
};

}  // namespace mppi_controller

#endif  // MPPI_CONTROLLER_H
