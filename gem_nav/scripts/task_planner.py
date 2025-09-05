#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import math
import rospy
from std_msgs.msg import Bool, String, UInt8
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped

# planner states
IDLE, READY, NAVIGATING, PAUSED, DONE, ABORTED = range(6)

class TaskPlanner(object):
    def __init__(self):
        # --- parameters
        self.frame_id           = rospy.get_param("~frame_id", "odom")
        self.pose_topic         = rospy.get_param("~pose_topic", "odom")  # относительный => внутри ns станет /gem/odom
        self.estop_topic        = rospy.get_param("~estop_topic", "/gem/safety/stop")  # абсолютный
        self.start_radius       = float(rospy.get_param("~start_radius", 2.0))
        self.goal_radius        = float(rospy.get_param("~goal_radius", 2.0))
        self.start_speed_thresh = float(rospy.get_param("~start_speed_thresh", 0.2))
        self.stop_at_goal       = bool(rospy.get_param("~stop_at_goal", True))
        self.publish_rate       = float(rospy.get_param("~publish_rate", 10.0))
        self.waypoints_file     = rospy.get_param("~waypoints_file", "")

        # --- state
        self.state = IDLE
        self.current_pos = None
        self.current_speed = 0.0
        self.estop = False
        self.path_msg = None

        # --- publishers
        self.pub_path   = rospy.Publisher("/pure_pursuit/path", Path, queue_size=1, latch=True)
        self.pub_enable = rospy.Publisher("/pure_pursuit/enable", Bool, queue_size=1)
        self.pub_state  = rospy.Publisher("task_planner/state", String, queue_size=10, latch=True)

        # --- subscribers
        rospy.Subscriber(self.pose_topic, Odometry, self.on_odom)
        rospy.Subscriber(self.estop_topic, Bool, self.on_estop)
        rospy.Subscriber("/task_planner/waypoints", Path, self.on_waypoints)  # можно присылать путь «на лету»

        # --- load waypoints from file - if available
        if self.waypoints_file:
            try:
                self.path_msg = self.load_path_from_csv(self.waypoints_file)
                rospy.loginfo("Planner: loaded %d waypoints from %s",
                              len(self.path_msg.poses), self.waypoints_file)
                if len(self.path_msg.poses) > 0:
                    self.state = READY
            except Exception as e:
                rospy.logwarn("Planner: cannot load waypoints_file=%s: %s",
                              self.waypoints_file, e)

        self.publish_state()

        # --- cycle of making decisions
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.step)

    # ===== callbacks

    def on_odom(self, msg):
        self.current_pos = msg.pose.pose.position
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.current_speed = math.hypot(vx, vy)

    def on_estop(self, msg):
        self.estop = bool(msg.data)
        if self.estop:
            # suspend decision immediately
            self.pub_enable.publish(Bool(False))
            self.set_state(PAUSED)

    def on_waypoints(self, path_msg):
        # assign path from topic
        self.path_msg = Path()
        self.path_msg.header.frame_id = self.frame_id
        # nirmalize frame and orientation
        for ps in path_msg.poses:
            p = PoseStamped()
            p.header.frame_id = self.frame_id
            p.pose.position = ps.pose.position
            p.pose.orientation = ps.pose.orientation
            if abs(p.pose.orientation.x) < 1e-9 and abs(p.pose.orientation.y) < 1e-9 \
               and abs(p.pose.orientation.z) < 1e-9 and abs(p.pose.orientation.w) < 1e-9:
                p.pose.orientation.w = 1.0
            self.path_msg.poses.append(p)
        self.set_state(READY if len(self.path_msg.poses) > 0 else IDLE)

    # ===== helpers

    def load_path_from_csv(self, fname):
        
        fname = os.path.expanduser(os.path.expandvars(fname))
        if not os.path.isabs(fname):            
            fname = os.path.join(os.getcwd(), fname)

        if not os.path.isfile(fname):
            raise IOError("Waypoints file not found: %s" % fname)

        path = Path()
        path.header.frame_id = self.frame_id
        with open(fname) as f:
            r = csv.reader(f)
            for row in r:
                if len(row) < 2:
                    continue
                try:
                    x = float(row[0]); y = float(row[1])
                except ValueError:
                    continue
                ps = PoseStamped()
                ps.header.frame_id = self.frame_id
                ps.pose.position.x = x
                ps.pose.position.y = y
                ps.pose.orientation.w = 1.0
                path.poses.append(ps)
        return path

    def dist(self, p, q):
        return math.hypot(p.x - q.x, p.y - q.y)

    def set_state(self, new_state):
        if self.state != new_state:
            self.state = new_state
            self.publish_state()

    def publish_state(self):
        names = ["IDLE","READY","NAVIGATING","PAUSED","DONE","ABORTED"]
        self.pub_state.publish(String(names[self.state]))

    # ===== main loop

    def step(self, _evt):
        # no waypoints — do nothing (waiting for file/topic)
        if not self.path_msg or len(self.path_msg.poses) == 0:
            self.set_state(IDLE)
            return

        # no pose - waiting fo odometry
        if self.current_pos is None:
            return

        # safety
        if self.estop:
            self.pub_enable.publish(Bool(False))
            self.set_state(PAUSED)
            return

        start = self.path_msg.poses[0].pose.position
        goal  = self.path_msg.poses[-1].pose.position

        if self.state in (IDLE, READY, PAUSED, DONE, ABORTED):
            # start condition
            if self.dist(self.current_pos, start) <= self.start_radius and \
               self.current_speed <= self.start_speed_thresh:
                # publishing Path (latched) и enabling of the controller
                self.path_msg.header.stamp = rospy.Time.now()
                for ps in self.path_msg.poses:
                    ps.header.stamp = self.path_msg.header.stamp
                self.pub_path.publish(self.path_msg)
                self.pub_enable.publish(Bool(True))
                self.set_state(NAVIGATING)
            else:
                self.set_state(READY)
            return

        if self.state == NAVIGATING:
            # check if finished
            if self.dist(self.current_pos, goal) <= self.goal_radius:
                if self.stop_at_goal:
                    self.pub_enable.publish(Bool(False))
                self.set_state(DONE)
            # just wait, pure_pursuit paves the way
            return


def main():
    rospy.init_node("task_planner")
    TaskPlanner()
    rospy.spin()

if __name__ == "__main__":
    main()

