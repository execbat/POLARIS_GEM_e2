#!/usr/bin/env python3
import rospy, cv2, numpy as np
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import Float32
from cv_bridge import CvBridge

class VisionLKA:
    def __init__(self):
        self.bridge = CvBridge()
        # relative name  "image"
        self.img_sub = rospy.Subscriber("image", Image, self.on_image, queue_size=1, buff_size=2**24)
        self.pub_cmd = rospy.Publisher("/vision_lka/cmd", AckermannDrive, queue_size=10)
        self.pub_err = rospy.Publisher("/vision_lka/lateral_error", Float32, queue_size=10)
        self.pub_dbg = rospy.Publisher("/vision_lka/debug", Image, queue_size=1)

        # params
        self.target_speed = rospy.get_param("~target_speed", 1.5)
        self.kp = rospy.get_param("~kp", 0.015)
        self.kd = rospy.get_param("~kd", 0.05)
        self.k_heading = rospy.get_param("~k_heading", 0.005)
        self.steer_limit = rospy.get_param("~steer_limit", 0.5)
        self.s_thresh = rospy.get_param("~s_thresh", 60)
        self.v_thresh = rospy.get_param("~v_thresh", 50)

        self.prev_err = 0.0
        self.prev_t = rospy.Time.now()
        
        self.estop = False
        rospy.Subscriber("/gem/safety/stop", Bool, self.on_stop, queue_size=1)
        
        rospy.loginfo("vision_lka params: ts=%.2f kp=%.3f kd=%.3f kh=%.3f steer_lim=%.2f s=%d v=%d",
              self.target_speed, self.kp, self.kd, self.k_heading, self.steer_limit,
              self.s_thresh, self.v_thresh)

    def on_image(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "cv_bridge err: %s", e); return

        h, w = img.shape[:2]
        roi = img[int(h*0.55):, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        
        road = cv2.inRange(hsv, (0,0,self.v_thresh), (179,self.s_thresh,255))
        
        road = cv2.medianBlur(road, 5)
        road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))

        y_scan = int(roi.shape[0]*0.75)
        scan = road[y_scan, :]

        nz = np.where(scan>0)[0]
        if nz.size < 10:
            # can't see the road — soft stop
            self.publish_cmd(0.0, 0.0)
            return

        left_idx  = nz[0]
        right_idx = nz[-1]
        lane_center_px = (left_idx + right_idx) / 2.0
        img_center_px  = w / 2.0
        lateral_err_px = (img_center_px - lane_center_px)

        # additional angular error of CoM  
        m = cv2.moments(road, binaryImage=True)
        heading_err = 0.0
        if m['m00'] > 1e3:
            cx = int(m['m10']/m['m00'])
            heading_err = (img_center_px - cx)

        now = rospy.Time.now()
        dt = (now - self.prev_t).to_sec() or 1e-3
        d_err = (lateral_err_px - self.prev_err) / dt
        steer = self.kp*lateral_err_px + self.kd*d_err + self.k_heading*heading_err
        steer = float(np.clip(steer, -self.steer_limit, self.steer_limit))
        self.prev_err, self.prev_t = lateral_err_px, now

        self.pub_err.publish(float(lateral_err_px))
        self.publish_cmd(self.target_speed, steer)

        # debug 
        dbg = cv2.cvtColor(road, cv2.COLOR_GRAY2BGR)
        cv2.line(dbg, (int(lane_center_px), y_scan), (int(lane_center_px), max(0, y_scan-40)), (0,0,255), 2)
        cv2.line(dbg, (int(img_center_px),   y_scan), (int(img_center_px),   max(0, y_scan-40)), (255,0,0), 2)
        self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(cv2.resize(dbg,(w,h//2)), encoding="bgr8"))

    def publish_cmd(self, speed, steer):
        if self.estop:
            speed = 0.0
            steer = 0.0
        cmd = AckermannDrive()
        cmd.speed = float(speed)
        cmd.steering_angle = float(steer)
        self.pub_cmd.publish(cmd)
        
    def on_stop(self, msg: Bool):
        self.estop = bool(msg.data)        

if __name__ == "__main__":
    rospy.init_node("vision_lka")
    VisionLKA()
    rospy.spin()
