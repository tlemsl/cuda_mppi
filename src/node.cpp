#include <ackermann_msgs/AckermannDriveStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <ros/ros.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <visualization_msgs/MarkerArray.h>

#include "cuda_mppi_controller/config.h"
#include "cuda_mppi_controller/mppi_controller.h"

namespace mppi_controller {

// Global controller instance for callback access
MPPIController* g_mppi_controller = nullptr;

// Callback function to handle current state updates
void currentStateCallback(const geometry_msgs::PoseStampedConstPtr& msg) {
  // ROS_INFO("Current state updated");

  if (g_mppi_controller) {
    State state;
    state[0] = msg->pose.position.x;
    state[1] = msg->pose.position.y;
    tf2::Quaternion q;
    tf2::fromMsg(msg->pose.orientation, q);
    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
    state[2] = yaw;
    g_mppi_controller->SetCurrentState(state);
  }
}

// Callback function to handle target state updates
void targetStateCallback(const geometry_msgs::PoseStampedConstPtr& msg) {
  if (g_mppi_controller) {
    State target;
    target[0] = msg->pose.position.x;
    target[1] = msg->pose.position.y;
    tf2::Quaternion q;
    tf2::fromMsg(msg->pose.orientation, q);
    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
    target[2] = yaw;
    g_mppi_controller->SetTargetState(target);
  }
  ROS_INFO("Target state updated");
}

}  // namespace mppi_controller

int main(int argc, char** argv) {
  ros::init(argc, argv, "mppi_controller");
  ros::NodeHandle nh;

  mppi_controller::MPPIController mppi_controller;
  mppi_controller::g_mppi_controller = &mppi_controller;

  ros::Subscriber state_sub =
      nh.subscribe("/mushr_mujoco_ros/buddy/pose", 10,
                   mppi_controller::currentStateCallback);
  ROS_INFO("Subscribed to current state");
  ros::Subscriber target_sub = nh.subscribe(
      "/move_base_simple/goal", 10, mppi_controller::targetStateCallback);
  ROS_INFO("Subscribed to target state");
  ros::Publisher cmd_vel_pub =
      nh.advertise<ackermann_msgs::AckermannDriveStamped>(
          "/mushr_mujoco_ros/buddy/control", 1);

  ros::Publisher trajectory_pub = nh.advertise<geometry_msgs::PoseArray>(
      "/mppi_controller/trajectories", 1);
  ros::Publisher optimal_trajectory_pub =
      nh.advertise<geometry_msgs::PoseArray>(
          "/mppi_controller/optimal_trajectory", 1);

  ros::Rate rate(1 / DT);
  Control optimal_control;
  std::vector<Trajectory> trajectories;
  Trajectory optimal_trajectory;

  ackermann_msgs::AckermannDriveStamped cmd_vel_msg;
  geometry_msgs::PoseArray trajectory_markers;
  geometry_msgs::PoseArray optimal_trajectory_markers;

  while (ros::ok()) {
    optimal_control = mppi_controller.ComputeOptimalControl();
    optimal_trajectory = mppi_controller.GetOptimalTrajectory();
    trajectories = mppi_controller.GetSampledTrajectories();
    ros::Time current_time = ros::Time::now();
    // mppi_controller.PrintStatus();

    // Publish trajectories
    trajectory_markers.header.frame_id = "map";
    trajectory_markers.header.stamp = current_time;
    trajectory_markers.poses.clear();
    for (const auto& trajectory : trajectories) {
      for (const auto& state : trajectory) {
        geometry_msgs::Pose pose;
        pose.orientation.w = 1.0;
        pose.position.x = state[0];
        pose.position.y = state[1];
        tf2::Quaternion q;
        q.setRPY(0, 0, state[2]);
        pose.orientation = tf2::toMsg(q);
        trajectory_markers.poses.push_back(pose);
      }
    }
    trajectory_pub.publish(trajectory_markers);

    // Publish optimal trajectory
    optimal_trajectory_markers.header.frame_id = "map";
    optimal_trajectory_markers.header.stamp = current_time;
    optimal_trajectory_markers.poses.clear();
    for (int i = 0; i < optimal_trajectory.size(); i++) {
      geometry_msgs::Pose pose;
      pose.position.x = optimal_trajectory[i][0];
      pose.position.y = optimal_trajectory[i][1];
      tf2::Quaternion q;
      q.setRPY(0, 0, optimal_trajectory[i][2]);
      pose.orientation = tf2::toMsg(q);
      optimal_trajectory_markers.poses.push_back(pose);
    }
    optimal_trajectory_pub.publish(optimal_trajectory_markers);

    // Publish optimal control
    cmd_vel_msg.header.stamp = current_time;
    cmd_vel_msg.header.frame_id = "base_link";
    cmd_vel_msg.drive.speed = optimal_control[0];
    cmd_vel_msg.drive.steering_angle = optimal_control[1];
    cmd_vel_pub.publish(cmd_vel_msg);

    // std::cout << "Optimal control: " << optimal_control[0] << ", "
    //           << optimal_control[1] << std::endl;
    rate.sleep();
    ros::spinOnce();
  }

  return 0;
}
