#include "cuda_mppi_controller/mppi_functions.h"
#include <cmath>

namespace mppi_controller {

State ForwardDynamics(const State& current_state, const Control& control) {
    float x = current_state[0];
    float y = current_state[1];
    float theta = current_state[2];

    float velocity = control[0];
    float steering = control[1];

    State next_state;
    next_state[0] = x + velocity * std::cos(theta) * DT;
    next_state[1] = y + velocity * std::sin(theta) * DT;
    next_state[2] = theta + (velocity * std::tan(steering) / WHEELBASE) * DT;

    return next_state;
}

float ComputeStateCost(const State& state, const Control& control, const State& target_state) {
    float x = state[0];
    float y = state[1];
    float theta = state[2];

    float velocity = control[0];
    float steering = control[1];

    float target_x = target_state[0];
    float target_y = target_state[1];
    float target_theta = target_state[2];

    float cost = 0.0;
    cost += Q_X * (x - target_x) * (x - target_x);
    cost += Q_Y * (y - target_y) * (y - target_y);
    cost += Q_THETA * (theta - target_theta) * (theta - target_theta);

    cost += R_VEL * velocity * velocity;
    cost += R_STEER * steering * steering;  

    return cost;
}

}  // namespace mppi_controller