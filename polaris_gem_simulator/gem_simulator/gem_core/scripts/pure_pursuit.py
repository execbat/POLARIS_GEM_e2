#!/usr/bin/env python3
import rospy
from std_msgs.msg import Bool, UInt8
from nav_msgs.msg import Path
from ackermann_msgs.msg import AckermannDriveStamped

IDLE, RUNNING, PAUSED, DONE, ABORTED = range(5)

class PurePursuitMinimal:
    def __init__(self):
        self.enabled = False
        self.has_path = False

        # relative names — they are mapped on  /pure_pursuit/* inside of launch  
        self.pub_cmd    = rospy.Publisher("cmd", AckermannDriveStamped, queue_size=10)
        self.pub_status = rospy.Publisher("status", UInt8, queue_size=1, latch=True)

        rospy.Subscriber("enable", Bool, self.on_enable)
        rospy.Subscriber("path", Path, self.on_path)

        self.state = IDLE
        self.publish_status()

        self.timer = rospy.Timer(rospy.Duration(0.05), self.step)  # 20 Гц

    def on_enable(self, msg):
        self.enabled = bool(msg.data)
        self._update_state()

    def on_path(self, msg):
        self.has_path = len(msg.poses) > 0
        self._update_state()

    def _update_state(self):
        if not self.has_path:
            self.state = IDLE
        elif not self.enabled:
            self.state = PAUSED
        else:
            self.state = RUNNING
        self.publish_status()

    def step(self, _):
        if self.state != RUNNING:
            # public zero cmd
            cmd = AckermannDriveStamped()
            cmd.drive.speed = 0.0
            cmd.drive.steering_angle = 0.0
            self.pub_cmd.publish(cmd)
            return

        # minimal controller
        # postind a slow speed
        cmd = AckermannDriveStamped()
        cmd.drive.speed = rospy.get_param("~cruise_speed", 1.0)
        cmd.drive.steering_angle = 0.0
        self.pub_cmd.publish(cmd)

    def publish_status(self):
        self.pub_status.publish(UInt8(self.state))

if __name__ == "__main__":
    rospy.init_node("pure_pursuit")
    PurePursuitMinimal()
    rospy.spin()
