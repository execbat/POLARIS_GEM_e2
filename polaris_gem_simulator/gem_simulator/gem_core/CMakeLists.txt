cmake_minimum_required(VERSION 3.0.2)
project(gem_core)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  diagnostic_msgs
  ackermann_msgs
  message_generation
)

add_message_files(
  FILES
  RobotState.msg
  Health.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs geometry_msgs diagnostic_msgs 
)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs geometry_msgs sensor_msgs diagnostic_msgs ackermann_msgs message_runtime
)

catkin_install_python(PROGRAMS
  scripts/supervisor.py
  scripts/controller_mux.py
  scripts/health_sim.py
  scripts/pure_pursuit.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
