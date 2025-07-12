#define USE_CUDA
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

#include "cuda_mppi_controller/config.h"
#include "cuda_mppi_controller/mppi_controller.h"
#include "cuda_mppi_controller/mppi_functions.cuh"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <curand.h>
#include <thrust/copy.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#endif

namespace mppi_controller {
MPPIController::MPPIController() {
  // Initialize containers
  control_sequences_.resize(NUM_SAMPLES);
  for (int i = 0; i < NUM_SAMPLES; ++i) {
    control_sequences_[i].resize(HORIZON);
  }
  trajectories_.resize(NUM_SAMPLES);
  trajectory_costs_.resize(NUM_SAMPLES);
  optimal_control_sequence_.resize(HORIZON);

  // Initialize states
  current_state_ = State::Zero();
  target_state_ = State::Zero();

  // Allocate device memory (use cudaMalloc, not cudaMallocHost)
  cudaMalloc(&current_state_d_, sizeof(CudaState));
  cudaMalloc(&target_state_d_, sizeof(CudaState));
  cudaMalloc(&control_sequences_d_, sizeof(CudaControl) * NUM_SAMPLES * HORIZON);
  cudaMalloc(&trajectories_d_, sizeof(CudaTrajectory) * NUM_SAMPLES);
  cudaMalloc(&trajectory_costs_d_, sizeof(float) * NUM_SAMPLES);
  cudaMalloc(&optimal_control_sequence_d_, sizeof(CudaControl) * HORIZON);

  // Create CUDA events for timing
  cudaEventCreate(&start_event_);
  cudaEventCreate(&end_event_);

  std::cout << "CUDA MPPI Controller (Ackermann) initialized with "
            << NUM_SAMPLES << " samples, horizon " << HORIZON << ", wheelbase "
            << WHEELBASE << "m, max steering " << MAX_STEERING << " rad"
            << std::endl;
}

MPPIController::~MPPIController() {
  // Free device memory
  cudaFree(current_state_d_);
  cudaFree(target_state_d_);
  cudaFree(control_sequences_d_);
  cudaFree(trajectories_d_);
  cudaFree(trajectory_costs_d_);
  cudaFree(optimal_control_sequence_d_);
  
  // Destroy CUDA events
  cudaEventDestroy(start_event_);
  cudaEventDestroy(end_event_);
}

void MPPIController::SetCurrentState(const State& state) {
  current_state_ = state;

  CudaState cuda_state = ToCudaState(state);
  cudaMemcpy(current_state_d_, &cuda_state, sizeof(CudaState),
             cudaMemcpyHostToDevice);
}

void MPPIController::SetTargetState(const State& target) {
  target_state_ = target;

  CudaState cuda_target = ToCudaState(target);
  cudaMemcpy(target_state_d_, &cuda_target, sizeof(CudaState),
             cudaMemcpyHostToDevice);
}

void MPPIController::GeneratePerturbedControls() {
  // Generate random numbers using cuRAND
  curandGenerator_t generator;
  
  // Create cuRAND generator
  curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
  
  // Set seed based on current time
  curandSetPseudoRandomGeneratorSeed(generator, 
    std::chrono::steady_clock::now().time_since_epoch().count());
  
  // Generate random numbers for velocity and steering perturbations
  int total_elements = NUM_SAMPLES * HORIZON * 2; // 2 for velocity and steering
  float* random_numbers_d;
  cudaMalloc(&random_numbers_d, total_elements * sizeof(float));
  
  // Generate normally distributed random numbers with mean=0, stddev=1
  curandGenerateNormal(generator, random_numbers_d, total_elements, 0.0f, 1.0f);
  
  // Convert base control sequence to device memory
  std::vector<CudaControl> base_controls_host(NUM_SAMPLES * HORIZON);
  for (int i = 0; i < NUM_SAMPLES; ++i) {
    for (int t = 0; t < HORIZON; ++t) {
      base_controls_host[i * HORIZON + t] = ToCudaControl(optimal_control_sequence_[t]);
    }
  }
  
  CudaControl* base_controls_d;
  cudaMalloc(&base_controls_d, NUM_SAMPLES * HORIZON * sizeof(CudaControl));
  cudaMemcpy(base_controls_d, base_controls_host.data(),
             NUM_SAMPLES * HORIZON * sizeof(CudaControl),
             cudaMemcpyHostToDevice);
  
  // Launch CUDA kernel to generate perturbed controls
  int threads_per_block = 256;
  int blocks = (NUM_SAMPLES * HORIZON + threads_per_block - 1) / threads_per_block;
  
  kernel_GeneratePerturbedControlsWithCuRAND<<<blocks, threads_per_block>>>(
      control_sequences_d_, base_controls_d, random_numbers_d, NUM_SAMPLES * HORIZON);
  
  cudaDeviceSynchronize();
  
  // Copy back to host for compatibility with existing code
  std::vector<CudaControl> flat_control_sequences(NUM_SAMPLES * HORIZON);
  cudaMemcpy(flat_control_sequences.data(), control_sequences_d_,
             sizeof(CudaControl) * NUM_SAMPLES * HORIZON,
             cudaMemcpyDeviceToHost);
  
  // Update host control sequences
  for (int i = 0; i < NUM_SAMPLES; ++i) {
    control_sequences_[i].resize(HORIZON);
    for (int t = 0; t < HORIZON; ++t) {
      control_sequences_[i][t] = ToHostControl(flat_control_sequences[i * HORIZON + t]);
    }
  }
  
  // Cleanup
  cudaFree(random_numbers_d);
  cudaFree(base_controls_d);
  curandDestroyGenerator(generator);
}

void MPPIController::GenerateTrajectoriesWithCost() {
  auto host_start_time = std::chrono::high_resolution_clock::now();
  
  // Optimize kernel launch configuration
  int total_threads = NUM_SAMPLES;
  int threads_per_block = KERNEL_SIZE;
  int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
  
  thread_size_ = dim3(threads_per_block, 1);
  block_size_ = dim3(blocks, 1);

  // Time CUDA kernel execution
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  cudaEventRecord(start);
  kernel_GenerateTrajectoriesWithCost<<<block_size_, thread_size_>>>(
      trajectories_d_, trajectory_costs_d_, current_state_d_, control_sequences_d_, target_state_d_);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  
  // Copy back trajectory costs
  cudaMemcpy(trajectory_costs_.data(), trajectory_costs_d_,
             sizeof(float) * NUM_SAMPLES, cudaMemcpyDeviceToHost);
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
  } else {
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
  float min_cost = *std::min_element(trajectory_costs_.begin(), trajectory_costs_.end());

  // Compute importance-weighted control
  std::vector<Control> weighted_controls(HORIZON, Control::Zero());
  float total_weights = 0.0;

  for (int i = 0; i < NUM_SAMPLES; ++i) {
    float weight = std::exp(-(trajectory_costs_[i] - min_cost) / LAMBDA);
    for (int t = 0; t < HORIZON; ++t) {
      weighted_controls[t] += control_sequences_[i][t] * weight;
    }
    total_weights += weight;
  }
  
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


  // Return first control action
  return optimal_control_sequence_[0];
}

// Get the optimal trajectory for visualization
const Trajectory& MPPIController::GetOptimalTrajectory() const {
  // Find the trajectory with minimum cost
  auto min_it = std::min_element(trajectory_costs_.begin(), trajectory_costs_.end());
  int min_idx = std::distance(trajectory_costs_.begin(), min_it);
  return trajectories_[min_idx];
}

// Get all sampled trajectories
const std::vector<Trajectory>& MPPIController::GetSampledTrajectories() const {
  return trajectories_;
}

void MPPIController::PrintStatus() const {
  std::cout << "Current state: [" << current_state_.transpose() << "]" << std::endl;
  std::cout << "Target state: [" << target_state_.transpose() << "]" << std::endl;
  if (!trajectory_costs_.empty()) {
    auto min_cost = *std::min_element(trajectory_costs_.begin(), trajectory_costs_.end());
    std::cout << "Minimum trajectory cost: " << min_cost << std::endl;
  }
}

// Benchmarking methods
void MPPIController::BenchmarkGeneratePerturbedControls(int iterations) {
  std::cout << "\n=== Benchmarking GeneratePerturbedControls (CUDA) ===" << std::endl;
  std::cout << "Running " << iterations << " iterations..." << std::endl;
  
  std::vector<double> times;
  times.reserve(iterations);
  
  // Warm up
  for (int i = 0; i < 5; ++i) {
    GeneratePerturbedControls();
  }
  
  // Benchmark
  for (int i = 0; i < iterations; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    GeneratePerturbedControls();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    times.push_back(duration.count());
  }
  
  // Calculate statistics
  double sum = 0.0;
  for (double time : times) {
    sum += time;
  }
  double avg = sum / iterations;
  
  std::sort(times.begin(), times.end());
  double median = times[iterations / 2];
  double min_time = times[0];
  double max_time = times[iterations - 1];
  
  std::cout << "Average time: " << avg << " microseconds" << std::endl;
  std::cout << "Median time: " << median << " microseconds" << std::endl;
  std::cout << "Min time: " << min_time << " microseconds" << std::endl;
  std::cout << "Max time: " << max_time << " microseconds" << std::endl;
}

void MPPIController::BenchmarkGenerateTrajectoriesWithCost(int iterations) {
  std::cout << "\n=== Benchmarking GenerateTrajectoriesWithCost (CUDA) ===" << std::endl;
  std::cout << "Running " << iterations << " iterations..." << std::endl;
  
  std::vector<double> times;
  times.reserve(iterations);
  
  // Ensure we have control sequences
  GeneratePerturbedControls();
  
  // Warm up
  for (int i = 0; i < 5; ++i) {
    GenerateTrajectoriesWithCost();
  }
  
  // Benchmark
  for (int i = 0; i < iterations; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    GenerateTrajectoriesWithCost();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    times.push_back(duration.count());
  }
  
  // Calculate statistics
  double sum = 0.0;
  for (double time : times) {
    sum += time;
  }
  double avg = sum / iterations;
  
  std::sort(times.begin(), times.end());
  double median = times[iterations / 2];
  double min_time = times[0];
  double max_time = times[iterations - 1];
  
  std::cout << "Average time: " << avg << " microseconds" << std::endl;
  std::cout << "Median time: " << median << " microseconds" << std::endl;
  std::cout << "Min time: " << min_time << " microseconds" << std::endl;
  std::cout << "Max time: " << max_time << " microseconds" << std::endl;
}

void MPPIController::BenchmarkComputeOptimalControl(int iterations) {
  std::cout << "\n=== Benchmarking ComputeOptimalControl (CUDA) ===" << std::endl;
  std::cout << "Running " << iterations << " iterations..." << std::endl;
  
  std::vector<double> times;
  times.reserve(iterations);
  
  // Warm up
  for (int i = 0; i < 5; ++i) {
    ComputeOptimalControl();
  }
  
  // Benchmark
  for (int i = 0; i < iterations; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    ComputeOptimalControl();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    times.push_back(duration.count());
  }
  
  // Calculate statistics
  double sum = 0.0;
  for (double time : times) {
    sum += time;
  }
  double avg = sum / iterations;
  
  std::sort(times.begin(), times.end());
  double median = times[iterations / 2];
  double min_time = times[0];
  double max_time = times[iterations - 1];
  
  std::cout << "Average time: " << avg << " microseconds" << std::endl;
  std::cout << "Median time: " << median << " microseconds" << std::endl;
  std::cout << "Min time: " << min_time << " microseconds" << std::endl;
  std::cout << "Max time: " << max_time << " microseconds" << std::endl;
}

void MPPIController::RunFullBenchmark(int iterations) {
  std::cout << "\n======================================" << std::endl;
  std::cout << "CUDA MPPI Controller Benchmark" << std::endl;
  std::cout << "NUM_SAMPLES: " << NUM_SAMPLES << std::endl;
  std::cout << "HORIZON: " << HORIZON << std::endl;
  std::cout << "======================================" << std::endl;
  
  BenchmarkGeneratePerturbedControls(iterations);
  BenchmarkGenerateTrajectoriesWithCost(iterations);
  BenchmarkComputeOptimalControl(iterations);
  
  std::cout << "\n======================================" << std::endl;
  std::cout << "Benchmark Complete" << std::endl;
  std::cout << "======================================" << std::endl;
}
};  // namespace mppi_controller
