#include "cuda_mppi_controller/mppi_functions.h"
#include <cmath>

namespace mppi_controller {

State ForwardDynamics(const State& current_state, const Control& control) {
    double x = current_state[0];
    double y = current_state[1];
    double theta = current_state[2];

    double velocity = control[0];
    double steering = control[1];

    State next_state;
    next_state[0] = x + velocity * std::cos(theta) * DT;
    next_state[1] = y + velocity * std::sin(theta) * DT;
    next_state[2] = theta + (velocity * std::tan(steering) / WHEELBASE) * DT;

    return next_state;
}

}  // namespace mppi_controller