#ifndef MPPI_FUNCTIONS_H
#define MPPI_FUNCTIONS_H

#include <cuda_runtime.h>

#include "cuda_mppi_controller/config.h"

namespace mppi_controller {

CudaState ToCudaState(const State& state);
State ToHostState(const CudaState& state);
CudaControl ToCudaControl(const Control& control);
Control ToHostControl(const CudaControl& control);
Trajectory ToTrajectory(const CudaTrajectory& trajectory);
CudaTrajectory ToCudaTrajectory(const Trajectory& trajectory);
__device__ inline CudaState ForwardDynamics(const CudaState& current_state,
                                            const CudaControl& control);
__device__ inline float ComputeStateCost(const CudaState& state,
                                         const CudaControl& control,
                                         const CudaState& target_state);

__global__ void kernel_GeneratePerturbedControls(
    CudaControl* perturbed_controls, const CudaControl* base_controls,
    const CudaControl* random_controls);

__global__ void kernel_GenerateTrajectoriesWithCost(
    CudaTrajectory* trajectories, float* trajectory_costs,
    const CudaState* current_state, const CudaControl* control_sequences,
    const CudaState* target_state);

__global__ void kernel_ComputeOptimalControl(
    CudaControl* optimal_control_sequence, const CudaTrajectory* trajectories,
    const float* trajectory_costs, const int num_samples, const int horizon);

}  // namespace mppi_controller

#endif  // MPPI_FUNCTIONS_H
