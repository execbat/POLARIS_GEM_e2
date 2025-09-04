#!/usr/bin/env python3

# ================================================================
# File name: gem_control.py
# Description: Low level Ackermann steering for GEM in Gazebo
# Author: Hang Cui
# Email: hangcui3@illinois.edu
# Date created: 06/05/2021
# Date last modified: 10/03/2021
# Version: 0.1
# Usage: roslaunch gem_gazebo gem_gazebo_rviz.launch
# Python version: 3.8
# ================================================================

import math
import sys

import numpy as np
import threading

import tf2_ros
import rospy

from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import Float64, Bool, Int32, String, Float32
from controller_manager_msgs.srv import ListControllers
from sensor_msgs.msg import BatteryState, Temperature



PI = 3.141592653589739


def get_steer_angle(phi):
    if phi >= 0.0:
        return (PI / 2) - phi
    return (-PI / 2) - phi


def _get_front_wheel_params(prefix):
    steer_link = rospy.get_param(prefix + "steering_link_name")
    steer_controller = rospy.get_param(prefix + "steering_controller_name")
    wheel_controller = rospy.get_param(prefix + "axle_controller_name")
    diameter = float(rospy.get_param(prefix + "diameter"))
    return steer_link, steer_controller, wheel_controller, 1 / (PI * diameter)


def _get_rear_wheel_params(prefix):
    link = rospy.get_param(prefix + "link_name")
    wheel_controller = rospy.get_param(prefix + "axle_controller_name")
    diameter = float(rospy.get_param(prefix + "diameter"))
    return link, wheel_controller, 1 / (PI * diameter)


class GEMController(object):

    def __init__(self,
                 left_front_inv_circ, right_front_inv_circ,
                 left_rear_inv_circ, right_rear_inv_circ,
                 right_rear_link,
                 steer_joint_dist: float,
                 wheel_base: float,
                 left_steer_pub: rospy.Publisher, right_steer_pub: rospy.Publisher,
                 left_front_wheel_pub: rospy.Publisher, right_front_wheel_pub: rospy.Publisher,
                 left_rear_wheel_pub: rospy.Publisher, right_rear_wheel_pub: rospy.Publisher,
                 cmd_timeout: float = 0.5):
        self.left_front_inv_circ = left_front_inv_circ
        self.right_front_inv_circ = right_front_inv_circ
        self.left_rear_inv_circ = left_rear_inv_circ
        self.right_rear_inv_circ = right_rear_inv_circ

        self.right_rear_link = right_rear_link

        self.cmd_timeout = cmd_timeout

        self.prev_t = rospy.get_time()
        self.last_cmd_time = rospy.get_time()

        self.steer_joint_dist_div_2 = steer_joint_dist/2.0

        self.wheelbase = wheel_base

        self.wheelbase_inv = 1 / (self.wheelbase*1.0)

        self.wheelbase_sqr = self.wheelbase**2

        self.ackermann_cmd_lock = threading.Lock()

        self.steer_ang             = 0.0
        self.steer_ang_vel         = 0.0
        self.speed                 = 0.0
        self.accel                 = 0.0
        self.last_steer_ang        = 0.0
        self.theta_left            = 0.0
        self.theta_left_old        = 0.0
        self.theta_right           = 0.0
        self.theta_right_old       = 0.0
        self.last_speed            = 0.0
        self.last_accel_limit      = 0.0
        self.left_front_ang_vel    = 0.0
        self.right_front_ang_vel   = 0.0
        self.left_rear_ang_vel     = 0.0
        self.right_rear_ang_vel    = 0.0

        self.left_steer_pub        = left_steer_pub
        self.right_steer_pub       = right_steer_pub
        self.left_front_wheel_pub  = left_front_wheel_pub
        self.right_front_wheel_pub = right_front_wheel_pub
        self.left_rear_wheel_pub   = left_rear_wheel_pub
        self.right_rear_wheel_pub  = right_rear_wheel_pub
        
        # added by Evgenii
        self.safety_stop = False
                
        # --- vehicle state and tresholds ---
        self.state = "IDLE"
        self.state_pub = rospy.Publisher("vehicle_state", String, queue_size=1, latch=True)
        self.state_pub.publish(String(self.state))

        self.batt_min_pct       = float(rospy.get_param("~battery_min_pct", 50.0))   # %
        self.temp_max_c         = float(rospy.get_param("~temp_max_c", 55.0))        # temperature
        self.gps_acc_thresh_mm  = float(rospy.get_param("~gps_acc_thresh_mm", 200.0))# mm
        self.gps_bad_timeout    = float(rospy.get_param("~gps_bad_timeout", 15.0))   # sec
        self.net_timeout_none   = float(rospy.get_param("~net_timeout_none", 10.0))  # sec (0 = not connected)
        self.net_timeout_low    = float(rospy.get_param("~net_timeout_low", 20.0))   # sec (2 = low)
        self.idle_speed_eps     = float(rospy.get_param("~idle_speed_eps", 0.05))    # m/sec

        # sensor values and "last time of the GOOD state"
        self.battery_pct  = None        # %
        self.temp_c       = None        # temperature
        self.gps_acc_mm   = None        # mm
        self.net_signal   = 1           # 0/1/2
        self.last_good_gps = rospy.get_time()
        self.last_good_net = rospy.get_time()

        # Topics 
        e_stop_topic      = rospy.get_param("~e_stop_topic", "/safety/stop")
        battery_topic     = rospy.get_param("~battery_topic", "/battery_state")
        temp_topic        = rospy.get_param("~temp_topic", "/temperature")
        gps_acc_topic     = rospy.get_param("~gps_acc_topic", "/gps/hacc_mm")
        net_signal_topic  = rospy.get_param("~net_signal_topic", "/net/signal")

        # Subscribes
        rospy.Subscriber(e_stop_topic, Bool, self._on_stop, queue_size=1)
        rospy.Subscriber(battery_topic,  BatteryState,  self._on_battery, queue_size=1)
        rospy.Subscriber(temp_topic,     Temperature,   self._on_temp,    queue_size=1)
        rospy.Subscriber(gps_acc_topic,  Float32,       self._on_gps_acc, queue_size=1)
        rospy.Subscriber(net_signal_topic, Int32,       self._on_signal,  queue_size=1)

    def loop_body(self, curr_t):
        delta_t = curr_t - self.prev_t  # Calculate delta_t first
        self.prev_t = curr_t        

        # updating state, and with ERROR state, force everything to stop
        if self._update_state_and_check_error(curr_t):
            # forced stop - publishing zeros
            self.theta_left = 0.0
            self.theta_right = 0.0
            self.left_front_ang_vel = 0.0
            self.right_front_ang_vel = 0.0
            self.left_rear_ang_vel = 0.0
            self.right_rear_ang_vel = 0.0
        else:   
            if self.cmd_timeout > 0.0 and (curr_t - self.last_cmd_time > self.cmd_timeout):
                steer_ang_changed, center_y = self.control_steering(self.last_steer_ang, 0.0, 0.001)
                self.control_wheels(0.0, 0.0, 0.0, steer_ang_changed, center_y)

            elif delta_t > 0.0:
                with self.ackermann_cmd_lock:
                    steer_ang     = self.steer_ang
                    steer_ang_vel = self.steer_ang_vel
                    speed         = self.speed
                    accel         = self.accel

                steer_ang_changed, center_y = self.control_steering(steer_ang, steer_ang_vel, delta_t)
                self.control_wheels(speed, accel, delta_t, steer_ang_changed, center_y)

        self.left_steer_pub.publish(self.theta_left)
        self.right_steer_pub.publish(self.theta_right)

        if self.left_front_wheel_pub:
            self.left_front_wheel_pub.publish(self.left_front_ang_vel)

        if self.right_front_wheel_pub:
            self.right_front_wheel_pub.publish(self.right_front_ang_vel)

        if self.left_rear_wheel_pub:
            self.left_rear_wheel_pub.publish(self.left_rear_ang_vel)

        if self.right_rear_wheel_pub:
            self.right_rear_wheel_pub.publish(self.right_rear_ang_vel)

    def ackermann_callback(self, ackermann_cmd):
        self.last_cmd_time = rospy.get_time()
        
        if self.state == "ERROR" or self.safety_stop:
            return
            
        with self.ackermann_cmd_lock:
            self.steer_ang = ackermann_cmd.steering_angle
            self.steer_ang_vel = ackermann_cmd.steering_angle_velocity
            self.speed = ackermann_cmd.speed
            self.accel = ackermann_cmd.acceleration

    def control_steering(self, steer_ang, steer_ang_vel_limit, delta_t):

        if steer_ang_vel_limit > 0.0:
            ang_vel = (steer_ang - self.last_steer_ang) / delta_t
            ang_vel = max(-steer_ang_vel_limit, min(ang_vel, steer_ang_vel_limit))
            theta = self.last_steer_ang + ang_vel * delta_t
        else:
            theta = steer_ang

        center_y = self.wheelbase * math.tan((PI/2)-theta)

        steer_ang_changed = theta != self.last_steer_ang

        if steer_ang_changed:
            self.last_steer_ang = theta
            self.theta_left = get_steer_angle(math.atan(self.wheelbase_inv * (center_y - self.steer_joint_dist_div_2)))
            self.theta_right = get_steer_angle(math.atan(self.wheelbase_inv * (center_y + self.steer_joint_dist_div_2)))

        return steer_ang_changed, center_y

    def control_wheels(self, speed, accel_limit, delta_t, steer_ang_changed, center_y):
        eps = 1e-6
    

        if accel_limit > 0.0:
            self.last_accel_limit = accel_limit
            accel = (speed - self.last_speed) / delta_t
            accel = max(-accel_limit, min(accel, accel_limit))
            veh_speed = self.last_speed + accel * delta_t
        else:
            self.last_accel_limit = accel_limit
            veh_speed = speed

        if veh_speed != self.last_speed or steer_ang_changed:
            self.last_speed = veh_speed
            left_dist = center_y - self.steer_joint_dist_div_2
            right_dist = center_y + self.steer_joint_dist_div_2
            gain = (2 * PI) * veh_speed / max(abs(center_y), eps)
            r = math.sqrt(left_dist ** 2 + self.wheelbase_sqr)
            self.left_front_ang_vel = gain * r * self.left_front_inv_circ
            r = math.sqrt(right_dist ** 2 + self.wheelbase_sqr)
            self.right_front_ang_vel = gain * r * self.right_front_inv_circ
            gain = (2 * PI) * veh_speed / max(center_y, eps if center_y >=0 else -eps)
            self.left_rear_ang_vel = gain * left_dist * self.left_rear_inv_circ
            self.right_rear_ang_vel = gain * right_dist * self.right_rear_inv_circ
            
    # added by Evgenii        
    def _on_stop(self, msg: Bool):
        self.safety_stop = bool(msg.data)

    def _on_battery(self, msg: BatteryState):
        # BatteryState.percentage 
        pct = float(msg.percentage)
        self.battery_pct = pct*100.0 if pct <= 1.0 else pct

    def _on_temp(self, msg: Temperature):
        self.temp_c = float(msg.temperature)

    def _on_gps_acc(self, msg: Float32):
        # horizontal accuracy in mm
        self.gps_acc_mm = float(msg.data)

    def _on_signal(self, msg: Int32):
        # 0 = not connected, 1 = connected, 2 = low
        self.net_signal = int(msg.data)
         
    def _update_state_and_check_error(self, curr_t: float) -> bool:
        error = False

        # e-stop 
        if self.safety_stop:
            error = True

        # battery
        if self.battery_pct is not None and self.battery_pct <= self.batt_min_pct:
            error = True
    
        # temperature
        if self.temp_c is not None and self.temp_c >= self.temp_max_c:
            error = True

        # GPS:  "good" if accuracy <= theshold; else going into ERROR state after timeout
        if self.gps_acc_mm is not None:
            if self.gps_acc_mm <= self.gps_acc_thresh_mm:
                self.last_good_gps = curr_t
            elif (curr_t - self.last_good_gps) >= self.gps_bad_timeout:
                error = True

        # Internet: 1 — good. 0 и 2 — calc with timeouts
        if self.net_signal == 1:
            self.last_good_net = curr_t
        elif self.net_signal == 0:
            if (curr_t - self.last_good_net) >= self.net_timeout_none:
                error = True
        elif self.net_signal == 2:
            if (curr_t - self.last_good_net) >= self.net_timeout_low:
                error = True

        # calc new state
        new_state = self.state
        if error:
            new_state = "ERROR"
        else:
            idle = abs(self.last_speed) <= self.idle_speed_eps and (curr_t - self.last_cmd_time) > 1.0
            new_state = "IDLE" if idle else "RUNNING"

        if new_state != self.state:
            self.state = new_state
            self.state_pub.publish(String(self.state))

        return error

def _vector3_to_numpy(msg) -> np.ndarray:
    return np.array([msg.x, msg.y, msg.z])


def main(argv):
    rospy.init_node("gem_ackermann_controller")
    (left_steer_link, left_steer_controller, left_front_wheel_controller, left_front_inv_circ) = \
        _get_front_wheel_params("~left_front_wheel/")
    (right_steer_link, right_steer_controller, right_front_wheel_controller, right_front_inv_circ) = \
        _get_front_wheel_params("~right_front_wheel/")
    (left_rear_link, left_rear_wheel_controller, left_rear_inv_circ) = \
        _get_rear_wheel_params("~left_rear_wheel/")
    (right_rear_link, right_rear_wheel_controller, right_rear_inv_circ) = \
        _get_rear_wheel_params("~right_rear_wheel/")
    publishing_frequency = rospy.get_param("~publishing_frequency", 30.0)
    cmd_timeout = rospy.get_param("~cmd_timeout", 0.5)

    # Look up frame translations w.r.t right rear wheel
    tf_buffer = tf2_ros.Buffer()
    _ = tf2_ros.TransformListener(tf_buffer)
    t_point = rospy.Time(0)  # Time point, 0 means get the latest
    timeout = rospy.Duration(secs=30)
    try:
        tf_buffer.can_transform(left_steer_link, right_rear_link, t_point, timeout)
        trans = tf_buffer.lookup_transform(left_steer_link, right_rear_link, t_point)
        ls = _vector3_to_numpy(trans.transform.translation)

        tf_buffer.can_transform(right_steer_link, right_rear_link, t_point, timeout)
        trans = tf_buffer.lookup_transform(right_steer_link, right_rear_link, t_point)
        rs = _vector3_to_numpy(trans.transform.translation)

        tf_buffer.can_transform(left_rear_link, right_rear_link, t_point, timeout)
        trans = tf_buffer.lookup_transform(left_rear_link, right_rear_link, t_point)
        lrw = _vector3_to_numpy(trans.transform.translation)
        rrw = np.array([0.0] * 3)

        steer_joint_dist = np.linalg.norm(ls - rs)
        wheel_base = np.linalg.norm((ls + rs) / 2.0 - (lrw + rrw) / 2.0)
        rospy.logdebug("Wheel base: %.3f m, Steer joint Distance: %.3f m" % (wheel_base, steer_joint_dist))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        rospy.logerr("Cannot lookup frames using TF2 after %.1f seconds. Shutting down." % timeout.to_sec())
        return

    list_controllers = rospy.ServiceProxy("controller_manager/list_controllers", ListControllers)
    list_controllers.wait_for_service()
    list_controllers.close()

    # Explicitly create Publisher and Subscriber in main function
    controller = GEMController(
        left_front_inv_circ, right_front_inv_circ,
        left_rear_inv_circ, right_rear_inv_circ,
        right_rear_link,
        steer_joint_dist=steer_joint_dist,
        wheel_base=wheel_base,
        left_steer_pub=rospy.Publisher(left_steer_controller + "/command", Float64, queue_size=1),
        right_steer_pub=rospy.Publisher(right_steer_controller + "/command", Float64, queue_size=1),
        left_front_wheel_pub=rospy.Publisher(left_front_wheel_controller + "/command", Float64, queue_size=1),
        right_front_wheel_pub=rospy.Publisher(right_front_wheel_controller + "/command", Float64, queue_size=1),
        left_rear_wheel_pub=rospy.Publisher(left_rear_wheel_controller + "/command", Float64, queue_size=1),
        right_rear_wheel_pub=rospy.Publisher(right_rear_wheel_controller + "/command", Float64, queue_size=1),
        cmd_timeout=cmd_timeout
    )
    _ = rospy.Subscriber("ackermann_cmd", AckermannDrive, controller.ackermann_callback, queue_size=1)

    try:
        rate = rospy.Rate(hz=publishing_frequency)
        while not rospy.is_shutdown():
            curr_t = rospy.get_time()
            controller.loop_body(curr_t)
            rate.sleep()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down GEM Ackermann Controller")


if __name__ == "__main__":
    sys.exit(main(rospy.myargv()) or 0)
