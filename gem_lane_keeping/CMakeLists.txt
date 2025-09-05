cmake_minimum_required(VERSION 3.0.2)
project(gem_lane_keeping)
find_package(catkin REQUIRED COMPONENTS rospy std_msgs sensor_msgs ackermann_msgs image_transport cv_bridge)
catkin_package()
catkin_install_python(PROGRAMS scripts/vision_lka.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
