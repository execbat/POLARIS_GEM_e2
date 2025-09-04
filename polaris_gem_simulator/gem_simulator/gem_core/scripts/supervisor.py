import rospy
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Twist
from ackermann_msgs.msg import AckermannDrive
from gem_core.msg import RobotState, Health
import time

# error bits
ERR_BATTERY    = 1 << 0
ERR_TEMP       = 1 << 1
ERR_GPS        = 1 << 2
ERR_INTERNET   = 1 << 3
ERR_ESTOP      = 1 << 4

class Supervisor:
    def __init__(self):
        # theshold values 
        self.battery_thresh = rospy.get_param("~battery_thresh_pct", 50.0)   # <=50% -> ERROR
        self.temp_thresh    = rospy.get_param("~temp_thresh_c", 55.0)        # >=55C -> ERROR
        self.gps_thresh_mm  = rospy.get_param("~gps_hacc_thresh_mm", 200.0)  # <200mm -> ERROR 
        self.net_timeout0_s = rospy.get_param("~net_timeout_disconnected_s", 10.0)
        self.net_timeout2_s = rospy.get_param("~net_timeout_low_s", 20.0)
        self.gps_timeout_s  = rospy.get_param("~gps_timeout_bad_s", 15.0)

        # Inspection of the cmd to detect IDLE/RUNNING
        self.cmd_topic = rospy.get_param("~cmd_activity_topic", "/gem/ackermann_cmd")
        self.cmd_activity_window_s = rospy.get_param("~cmd_activity_window_s", 1.0)

        # Subscriptions
        self.health_sub = rospy.Subscriber("health", Health, self.on_health, queue_size=1)
        # Optionally receiveng the commands
        self.cmd_sub1 = rospy.Subscriber(self.cmd_topic, AckermannDrive, self.on_cmd, queue_size=1)
        self.cmd_sub2 = rospy.Subscriber("/cmd_vel", Twist, self.on_cmd, queue_size=1)

        # Publishers
        self.state_pub  = rospy.Publisher("robot_state", RobotState, queue_size=1, latch=True)
        self.stop_pub   = rospy.Publisher("safety/stop", Bool, queue_size=1, latch=True)
        self.reason_pub = rospy.Publisher("robot_state_reason", String, queue_size=1, latch=True)

        # Current states
        self.health = Health()
        self.last_cmd_time = rospy.Time(0)

        # Timers of "Bad" conditions. Used for timeouts
        self.bad_since = {
            "gps": None,        # condition: gps_hacc_mm < 200
            "net0": None,       # internet_status == 0
            "net2": None,       # internet_status == 2
        }

        self.timer = rospy.Timer(rospy.Duration(0.1), self.tick)

    def on_health(self, msg: Health):
        self.health = msg
        # Update of the Timers of "Bad" conditions
        now = rospy.Time.now()

        # GPS condition — "if < 200 mm => ERROR after 15sec"
        if msg.gps_hacc_mm < self.gps_thresh_mm:
            if self.bad_since["gps"] is None:
                self.bad_since["gps"] = now
        else:
            self.bad_since["gps"] = None

        # Internet
        if msg.internet_status == 0:
            if self.bad_since["net0"] is None:
                self.bad_since["net0"] = now
            self.bad_since["net2"] = None
        elif msg.internet_status == 2:
            if self.bad_since["net2"] is None:
                self.bad_since["net2"] = now
            self.bad_since["net0"] = None
        else:  # 1 = OK
            self.bad_since["net0"] = None
            self.bad_since["net2"] = None

    def on_cmd(self, *_):
        self.last_cmd_time = rospy.Time.now()

    def is_timeout(self, key, threshold_s):
        t = self.bad_since[key]
        if t is None:
            return False
        return (rospy.Time.now() - t).to_sec() >= threshold_s

    def tick(self, _):
        # Collecting of the errors
        mask = 0
        reasons = []

        # Instant changes
        if self.health.battery_pct <= self.battery_thresh:
            mask |= ERR_BATTERY; reasons.append("battery<=%.1f%%" % self.battery_thresh)
        if self.health.temperature_c >= self.temp_thresh:
            mask |= ERR_TEMP; reasons.append("temp>=%.1fC" % self.temp_thresh)
        if self.health.estop:
            mask |= ERR_ESTOP; reasons.append("estop=true")

        # Delayed changes
        if self.is_timeout("gps", self.gps_timeout_s):
            mask |= ERR_GPS; reasons.append("gps_hacc<%.0fmm for %.0fs" % (self.gps_thresh_mm, self.gps_timeout_s))
        if self.is_timeout("net0", self.net_timeout0_s):
            mask |= ERR_INTERNET; reasons.append("internet=0 for %.0fs" % self.net_timeout0_s)
        if self.is_timeout("net2", self.net_timeout2_s):
            mask |= ERR_INTERNET; reasons.append("internet=2 for %.0fs" % self.net_timeout2_s)

        # Condition
        state = RobotState.RUNNING if (rospy.Time.now() - self.last_cmd_time).to_sec() < self.cmd_activity_window_s else RobotState.IDLE
        reason = ""
        if mask != 0:
            state = RobotState.ERROR
            reason = ", ".join(reasons)

        # Publications
        msg = RobotState(state=state, error_mask=mask, reason=reason)
        self.state_pub.publish(msg)
        self.stop_pub.publish(Bool(data=(state == RobotState.ERROR)))
        if reason:
            self.reason_pub.publish(String(data=reason))

def main():
    rospy.init_node("gem_supervisor")
    Supervisor()
    rospy.spin()

if __name__ == "__main__":
    main()

