#ifndef MPPI_CONTROLLER_H
#define MPPI_CONTROLLER_H

#include <Eigen/Dense>
#include <random>
#include <vector>
#include <thread>
#include <functional>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include "config.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <curand.h>
#include <thrust/copy.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#endif

namespace mppi_controller {

class MPPIController {
 private:
  // C++ implementation
  // Cost matrices
  Eigen::Matrix3f Q_;  ///< State cost matrix [x, y, theta]
  Eigen::Matrix2f R_;  ///< Control cost matrix [velocity, steering]

  // Current state and target
  State current_state_;  ///< Current robot state [x, y, theta]
  State target_state_;   ///< Target robot state [x, y, theta]

  // Control sequences and trajectories
  std::vector<std::vector<Control>>
      control_sequences_;                 ///< Sampled control sequences
  std::vector<Trajectory> trajectories_;  ///< Simulated trajectories
  std::vector<float> trajectory_costs_;  ///< Cost for each trajectory
  std::vector<Control>
      optimal_control_sequence_;  ///< Current optimal control sequence

#ifdef USE_CUDA
  // CUDA implementation
  CudaState* current_state_d_;
  CudaState* target_state_d_;

  CudaControl* control_sequences_d_;
  CudaTrajectory* trajectories_d_;
  float* trajectory_costs_d_;
  CudaControl* optimal_control_sequence_d_;
  CudaControl* random_controls_d_;
  
  // CUDA implementation
  dim3 block_size_;
  dim3 thread_size_;

  curandGenerator_t generator_;
  float* random_numbers_d_;
  
  // CUDA events for timing
  cudaEvent_t start_event_;
  cudaEvent_t end_event_;
#else
  // Random number generation for CPU version
  std::mt19937 generator_;
  std::normal_distribution<float> noise_dist_;
#endif
 public:
  MPPIController();
  ~MPPIController();

  void SetCurrentState(const State& state);

  void SetTargetState(const State& target);

  Control ComputeOptimalControl();

  const Trajectory& GetOptimalTrajectory() const;

  const std::vector<Trajectory>& GetSampledTrajectories() const;

  void PrintStatus() const;

  // Benchmarking methods
  void BenchmarkGeneratePerturbedControls(int iterations = 100);
  void BenchmarkGenerateTrajectoriesWithCost(int iterations = 100);
  void BenchmarkComputeOptimalControl(int iterations = 100);
  void RunFullBenchmark(int iterations = 100);

 private:
  void GeneratePerturbedControls();

  void GenerateTrajectoriesWithCost();
};

}  // namespace mppi_controller

#endif  // MPPI_CONTROLLER_H
