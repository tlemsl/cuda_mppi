find_package(ackermann_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  tf
  geometry_msgs
  visualization_msgs
  ackermann_msgs
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS
)

include_directories(
  include ${catkin_INCLUDE_DIRS}
)


# Create executable
add_executable(c_mppi_controller src/node.cpp)
add_executable(cuda_mppi_controller src/node.cpp)

# Link libraries
target_link_libraries(c_mppi_controller ${catkin_LIBRARIES} c_mppi_controller_lib Eigen3::Eigen TBB::tbb)
target_link_libraries(cuda_mppi_controller ${catkin_LIBRARIES} cuda_mppi_controller_lib Eigen3::Eigen TBB::tbb)
target_link_libraries(cuda_mppi_controller curand)

