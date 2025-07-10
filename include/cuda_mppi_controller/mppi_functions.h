#ifndef MPPI_FUNCTIONS_H
#define MPPI_FUNCTIONS_H

#include "cuda_mppi_controller/config.h"

namespace mppi_controller {

State ForwardDynamics(const State& current_state, const Control& control);

}  // namespace mppi_controller

#endif  // MPPI_FUNCTIONS_H
