#!/usr/bin/env python3
import rospy
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
import tf
import sys

class ModelStatesToOdom:
    def __init__(self):
        self.model_name   = rospy.get_param("~model_name", "gem")
        self.frame_id     = rospy.get_param("~frame_id", "odom")
        self.child_frame  = rospy.get_param("~child_frame", "base_link")
        topic_out         = rospy.get_param("~odom_topic", "odom")  

        self.pub = rospy.Publisher(topic_out, Odometry, queue_size=10)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.cb)

    def cb(self, msg):
        try:
            idx = msg.name.index(self.model_name)
        except ValueError:
            return
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = self.child_frame
        odom.pose.pose = msg.pose[idx]
        odom.twist.twist = msg.twist[idx]
        self.pub.publish(odom)

if __name__ == "__main__":
    rospy.init_node("model_states_to_odom")
    ModelStatesToOdom()
    rospy.spin()
