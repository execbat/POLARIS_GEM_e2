import rospy
from std_msgs.msg import String
from ackermann_msgs.msg import AckermannDrive

class ControllerMux:
    def __init__(self):
        self.output_topic = rospy.get_param("~output_cmd_topic", "/gem/ackermann_cmd")
        self.inputs = rospy.get_param("~inputs", {
            "pure_pursuit": "/pure_pursuit/cmd",
            "stanley":      "/stanley/cmd",
        })
        self.active = rospy.get_param("~default_controller", "pure_pursuit")

        self.pub = rospy.Publisher(self.output_topic, AckermannDrive, queue_size=10)
        self.active_pub = rospy.Publisher("~active", String, queue_size=1, latch=True)
        self.select_sub = rospy.Subscriber("~select", String, self.on_select, queue_size=1)

        self.subs = {}
        for name, topic in self.inputs.items():
            self.subs[name] = rospy.Subscriber(topic, AckermannDrive, self.make_cb(name), queue_size=10)

        self.active_pub.publish(String(self.active))
        rospy.loginfo("controller_mux active=%s -> %s", self.active, self.output_topic)

    def on_select(self, msg: String):
        name = msg.data.strip()
        if name in self.inputs:
            self.active = name
            self.active_pub.publish(String(self.active))
            rospy.loginfo("controller_mux switched to %s", name)
        else:
            rospy.logwarn("controller_mux: unknown controller '%s'", name)

    def make_cb(self, name):
        def cb(m: AckermannDrive):
            if name == self.active:
                self.pub.publish(m)
        return cb

def main():
    rospy.init_node("controller_mux")
    ControllerMux()
    rospy.spin()

if __name__ == "__main__":
    main()

