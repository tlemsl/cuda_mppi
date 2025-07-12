#define USE_CUDA
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

#include "cuda_mppi_controller/config.h"
#include "cuda_mppi_controller/mppi_controller.h"
#include "cuda_mppi_controller/mppi_functions.cuh"

#include <cuda_runtime.h>
#include <thrust/random.h>

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
  cudaMalloc(&random_controls_d_, sizeof(CudaControl) * NUM_SAMPLES * HORIZON);

  // Create CUDA events for timing
  cudaEventCreate(&start_event_);
  cudaEventCreate(&end_event_);

  // Initialize random number generator
  generator_.seed(std::chrono::steady_clock::now().time_since_epoch().count());
  noise_dist_ = thrust::normal_distribution<float>(0.0f, 1.0f);
  
  // Initialize std random number generator for TBB
  std_generator_.seed(std::chrono::steady_clock::now().time_since_epoch().count());
  std_noise_dist_ = std::normal_distribution<float>(0.0f, 0.5f);

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
  cudaFree(random_controls_d_);
  
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

// // Generate perturbed control sequences
// void MPPIController::GeneratePerturbedControls() {
//   // auto start_time = std::chrono::high_resolution_clock::now();
  
//   // Generate perturbed control sequences
//   int total_thread = NUM_SAMPLES * HORIZON;
//   thread_size_ = dim3(HORIZON * 2, 1);
//   int total_block = (total_thread + thread_size_.x - 1) / thread_size_.x;
//   block_size_ = dim3(total_block, 1);

//   // auto random_gen_start = std::chrono::high_resolution_clock::now();
//   thrust::generate(random_controls_.begin(), random_controls_.end(),
//                    [this]() { 
//                      CudaControl control;
//                      control.velocity = noise_dist_(generator_) * MAX_VELOCITY;
//                      control.steering_angle = noise_dist_(generator_) * MAX_STEERING;
//                      return control;
//                    });
//   // auto random_gen_end = std::chrono::high_resolution_clock::now();
//   // auto random_gen_duration = std::chrono::duration_cast<std::chrono::microseconds>(random_gen_end - random_gen_start);
//   // std::cout << "[CUDA] Random generation time: " << random_gen_duration.count() << " microseconds" << std::endl;

//   // Convert optimal_control_sequence_ to flat array for device transfer
//   // auto memory_prep_start = std::chrono::high_resolution_clock::now();
//   std::vector<CudaControl> flat_optimal_controls(NUM_SAMPLES * HORIZON);
//   for (int i = 0; i < NUM_SAMPLES; ++i) {
//     for (int t = 0; t < HORIZON; ++t) {
//       flat_optimal_controls[i * HORIZON + t] = ToCudaControl(optimal_control_sequence_[t]);
//     }
//   }

//   cudaMemcpyAsync(optimal_control_sequence_d_, flat_optimal_controls.data(),
//              sizeof(CudaControl) * NUM_SAMPLES * HORIZON,
//              cudaMemcpyHostToDevice);

//   cudaMemcpyAsync(random_controls_d_, random_controls_.data(),
//              sizeof(CudaControl) * NUM_SAMPLES * HORIZON,
//              cudaMemcpyHostToDevice);
//   // auto memory_prep_end = std::chrono::high_resolution_clock::now();
//   // auto memory_prep_duration = std::chrono::duration_cast<std::chrono::microseconds>(memory_prep_end - memory_prep_start);
//   // std::cout << "[CUDA] Memory preparation time: " << memory_prep_duration.count() << " microseconds" << std::endl;

//   // Time CUDA kernel execution
//   cudaEventRecord(start_event_);
//   kernel_GeneratePerturbedControls<<<block_size_, thread_size_>>>(
//       control_sequences_d_, optimal_control_sequence_d_, random_controls_d_);
//   cudaDeviceSynchronize();
//   cudaEventRecord(end_event_);
  
//   float kernel_time;
//   cudaEventElapsedTime(&kernel_time, start_event_, end_event_);
//   std::cout << "[CUDA] GeneratePerturbedControls kernel time: " << kernel_time * 1000 << " microseconds" << std::endl;
  
//   // Copy back and convert to host format
//   // auto memory_copy_start = std::chrono::high_resolution_clock::now();
//   std::vector<CudaControl> flat_control_sequences(NUM_SAMPLES * HORIZON);
//   cudaMemcpyAsync(flat_control_sequences.data(), control_sequences_d_,
//              sizeof(CudaControl) * NUM_SAMPLES * HORIZON,
//              cudaMemcpyDeviceToHost);
             
//   // Convert flat array back to vector of vectors
//   for (int i = 0; i < NUM_SAMPLES; ++i) {
//     for (int t = 0; t < HORIZON; ++t) {
//       control_sequences_[i][t] = ToHostControl(flat_control_sequences[i * HORIZON + t]);
//     }
//   }
//   // auto memory_copy_end = std::chrono::high_resolution_clock::now();
//   // auto memory_copy_duration = std::chrono::duration_cast<std::chrono::microseconds>(memory_copy_end - memory_copy_start);
//   // std::cout << "[CUDA] Memory copy back time: " << memory_copy_duration.count() << " microseconds" << std::endl;
  
//   // auto host_end_time = std::chrono::high_resolution_clock::now();
//   // auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(host_end_time - host_start_time);
//   // std::cout << "[CUDA] GeneratePerturbedControls TOTAL time: " << total_duration.count() << " microseconds" << std::endl;
// }
void MPPIController::GeneratePerturbedControls() {  
  // Create a functor class for TBB parallel execution
  class PerturbedControlsFunctor {
   private:
    std::vector<std::vector<Control>>& control_sequences_;
    const std::vector<Control>& optimal_control_sequence_;
    std::mt19937& generator_;
    std::normal_distribution<float>& noise_dist_;
    
   public:
    PerturbedControlsFunctor(
        std::vector<std::vector<Control>>& control_sequences,
        const std::vector<Control>& optimal_control_sequence,
        std::mt19937& generator,
        std::normal_distribution<float>& noise_dist)
        : control_sequences_(control_sequences),
          optimal_control_sequence_(optimal_control_sequence),
          generator_(generator),
          noise_dist_(noise_dist) {}

    void operator()(const tbb::blocked_range<int>& range) const {
      // Each thread gets its own random number generator to avoid contention
      std::mt19937 local_generator = generator_;
      std::normal_distribution<float> local_noise_dist = noise_dist_;
      
      // Seed the local generator with a different seed for each thread
      local_generator.seed(std::chrono::steady_clock::now().time_since_epoch().count() + 
                          std::hash<std::thread::id>{}(std::this_thread::get_id()));
      
      for (int i = range.begin(); i != range.end(); ++i) {
        control_sequences_[i].resize(HORIZON);

        for (int t = 0; t < HORIZON; ++t) {
          Control perturbed_control;

          // Generate random perturbations using local generator
          float vel_noise = local_noise_dist(local_generator) * MAX_VELOCITY;
          float steering_noise = local_noise_dist(local_generator) * MAX_STEERING;

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
  };
  
  // Execute parallel generation using TBB
  tbb::parallel_for(tbb::blocked_range<int>(0, NUM_SAMPLES), PerturbedControlsFunctor(
      control_sequences_, optimal_control_sequence_, std_generator_, std_noise_dist_));
  
  // Transfer control sequences to GPU
  std::vector<CudaControl> flat_control_sequences(NUM_SAMPLES * HORIZON);
  for (int i = 0; i < NUM_SAMPLES; ++i) {
    for (int t = 0; t < HORIZON; ++t) {
      flat_control_sequences[i * HORIZON + t] = ToCudaControl(control_sequences_[i][t]);
    }
  }
  
  cudaMemcpy(control_sequences_d_, flat_control_sequences.data(),
             sizeof(CudaControl) * NUM_SAMPLES * HORIZON,
             cudaMemcpyHostToDevice);
}

void MPPIController::GenerateTrajectoriesWithCost() {
  auto host_start_time = std::chrono::high_resolution_clock::now();
  
  // Optimize kernel launch configuration
  int total_threads = NUM_SAMPLES;
  int threads_per_block = 256;  // Better utilization than 32
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
