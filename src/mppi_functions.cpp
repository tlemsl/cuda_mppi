#include "cuda_mppi_controller/mppi_functions.h"
#include <cmath>

namespace mppi_controller {

#ifdef ACKERMANN_MODEL
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
#endif

#ifdef QUADROTOR_MODEL

Eigen::Matrix<float, 3, 3> RotationMatrix(const Eigen::Quaternionf& q) {
    Eigen::Matrix<float, 3, 3> R;
    R << 1 - 2 * q.y() * q.y() - 2 * q.z() * q.z(), 2 * q.x() * q.y() - 2 * q.z() * q.w(), 2 * q.x() * q.z() + 2 * q.y() * q.w(),
         2 * q.x() * q.y() + 2 * q.z() * q.w(), 1 - 2 * q.x() * q.x() - 2 * q.z() * q.z(), 2 * q.y() * q.z() - 2 * q.x() * q.w(),
         2 * q.x() * q.z() - 2 * q.y() * q.w(), 2 * q.y() * q.z() + 2 * q.x() * q.w(), 1 - 2 * q.x() * q.x() - 2 * q.y() * q.y();
    return R;
}

Eigen::Vector4f QuaternionHat(const Eigen::Vector3f& omega) {
    Eigen::Vector4f hat;
    hat << 0, omega[0], omega[1], omega[2];
    return hat;
}


State ForwardDynamics(const State& current_state, const Control& control) {
    float x = current_state[0];
    float y = current_state[1];
    float z = current_state[2];
    float qw = current_state[3];
    float qx = current_state[4];
    float qy = current_state[5];
    float qz = current_state[6];
    float vx = current_state[7];
    float vy = current_state[8];
    float vz = current_state[9];
    float wx = current_state[10];
    float wy = current_state[11];
    float wz = current_state[12];

    float f_z = K_THRUST * (control[0] + control[1] + control[2] + control[3]);  
    // Nova configuration
    // 
    
    float tau_x = control[1];
    float tau_y = control[2];
    float tau_z = control[3];

    Eigen::Quaternionf q(qx, qy, qz, qw);
    Eigen::Matrix<float, 3, 3> R = RotationMatrix(q);

    Eigen::Vector3f velocity_world(vx, vy, vz);

    Eigen::Vector3f omega_body(wx, wy, wz);

    Eigen::Vector3f force_body(0, 0, f_z);
    Eigen::Vector3f torque_body(tau_x, tau_y, tau_z);
    Eigen::Vector3f acceleration_body = (force_body + torque_body) / MASS;
    // Eigen::Vector3f omega_dot_body = (torque_body - QuaternionHat(omega_body) * omega_body) / INERTIA_XX;


    float qw_dot = 0.5 * (qw * wx - qx * wy - qy * wz + qz * wx);
    float qx_dot = 0.5 * (qw * wy + qx * wz - qy * wx - qz * wy);
    float qy_dot = 0.5 * (qw * wz + qx * wx - qy * wy - qz * wz);
    float qz_dot = 0.5 * (-qw * wx + qx * wy + qy * wz - qz * wx);

    State next_state;

    next_state[0] = x + vx * DT;
    next_state[1] = y + vy * DT;
    next_state[2] = z + vz * DT;

    next_state[3] = qw + qw_dot * DT;
    next_state[4] = qx + qx_dot * DT;
    next_state[5] = qy + qy_dot * DT;
    next_state[6] = qz + qz_dot * DT;

    next_state[7] = vx + acceleration_body[0] * DT;
    next_state[8] = vy + acceleration_body[1] * DT;
    next_state[9] = vz + acceleration_body[2] * DT;

    next_state[10] = wx + omega_body[0] * DT;
    next_state[11] = wy + omega_body[1] * DT;
    next_state[12] = wz + omega_body[2] * DT;
    return current_state;
}

float ComputeStateCost(const State& state, const Control& control, const State& target_state) {
    // TODO: Implement Quadrotor model
    return 0.0;
}
#endif
}  // namespace mppi_controller