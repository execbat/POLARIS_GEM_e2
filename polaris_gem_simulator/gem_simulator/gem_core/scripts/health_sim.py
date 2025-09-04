import rospy
from gem_core.msg import Health

def main():
    rospy.init_node("health_sim")
    pub = rospy.Publisher("/gem/health", Health, queue_size=1, latch=True)
    rate = rospy.Rate(10)
    h = Health()
    h.battery_pct = 100.0
    h.temperature_c = 35.0
    h.gps_hacc_mm = 500.0
    h.internet_status = 1
    h.estop = False
    while not rospy.is_shutdown():
        pub.publish(h)
        rate.sleep()

if __name__ == "__main__":
    main()

