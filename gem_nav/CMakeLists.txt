cmake_minimum_required(VERSION 3.0.2)
project(gem_nav)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  nav_msgs
  geometry_msgs
  ackermann_msgs
  tf2_ros
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs nav_msgs geometry_msgs ackermann_msgs tf2_ros
)

catkin_install_python(PROGRAMS
  scripts/task_planner.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

