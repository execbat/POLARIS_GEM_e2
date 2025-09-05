#!/usr/bin/env python3
import rospy, cv2, numpy as np, math
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge

def clamp(v, lo, hi): return max(lo, min(hi, v))

class VisionLKA:
    def __init__(self):
        rospy.init_node("vision_lka")
        self.bridge = CvBridge()

       
        self.target_speed = rospy.get_param("~target_speed", 1.5)
        self.kp           = rospy.get_param("~kp", 0.015)     
        self.kd           = rospy.get_param("~kd", 0.05)      
        self.k_heading    = rospy.get_param("~k_heading", 0.005)
        self.steer_limit  = rospy.get_param("~steer_limit", 0.5)  
        self.s_thresh     = rospy.get_param("~s_thresh", 60)       
        self.v_thresh     = rospy.get_param("~v_thresh", 50)       
        self.roi_top      = rospy.get_param("~roi_top", 0.55)      
        self.scan_pos     = rospy.get_param("~scan_pos", 0.75)    
        self.min_mask_px  = rospy.get_param("~min_mask_px", 300)   
        self.min_lane_w   = rospy.get_param("~min_lane_width_px", 40) 

        
        self.pub_cmd = rospy.Publisher("cmd", AckermannDrive, queue_size=10)
        self.pub_err = rospy.Publisher("lateral_error", Float32, queue_size=10)
        self.pub_dbg = rospy.Publisher("debug", Image, queue_size=1)
        rospy.Subscriber("image", Image, self.on_image, queue_size=1)
        rospy.Subscriber("/gem/safety/stop", Bool, self.on_stop, queue_size=1)

        self.estop = False
        self.prev_err = 0.0
        self.prev_t = rospy.get_time()

        rospy.loginfo("vision_lka params: ts=%.2f kp=%.3f kd=%.3f kh=%.3f steer_lim=%.2f s=%d v=%d roi=%.2f scan=%.2f",
                      self.target_speed, self.kp, self.kd, self.k_heading, self.steer_limit,
                      self.s_thresh, self.v_thresh, self.roi_top, self.scan_pos)

    def on_stop(self, msg: Bool):
        self.estop = bool(msg.data)

    def publish_cmd(self, speed, steer):
        if self.estop:
            speed = 0.0
            steer = 0.0
        cmd = AckermannDrive()
        cmd.speed = float(speed)
        cmd.steering_angle = float(clamp(steer, -self.steer_limit, self.steer_limit))
        self.pub_cmd.publish(cmd)

    def on_image(self, msg: Image):
        # 1) BGR и ROI
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "cv_bridge err: %s", e); return

        h, w = bgr.shape[:2]
        y0 = int(h * self.roi_top)
        roi = bgr[y0:h, :]

        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white  = cv2.inRange(hsv, (0,   0,   self.v_thresh), (179, self.s_thresh, 255))
        yellow = cv2.inRange(hsv, (15, 80,   80),            (35,  255,          255))
        mask = cv2.bitwise_or(white, yellow)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        mask = cv2.medianBlur(mask, 5)

        nz_count = int(np.count_nonzero(mask))

        
        if nz_count < self.min_mask_px:
            self._publish_debug(mask, w, h, y0, label="NO LANE")
            self.pub_err.publish(0.0)
            self.publish_cmd(0.0, 0.0)
            return

        
        ys = mask.shape[0]
        y_scan = int(ys * self.scan_pos)
        scan = mask[y_scan, :]
        xs = np.where(scan > 0)[0]
        if xs.size < 2:
            self._publish_debug(mask, w, h, y0, label="SCAN EMPTY")
            self.pub_err.publish(0.0)
            self.publish_cmd(0.0, 0.0)
            return

        left_idx, right_idx = int(xs[0]), int(xs[-1])
        if (right_idx - left_idx) < self.min_lane_w:
            self._publish_debug(mask, w, h, y0, label="LANE TOO NARROW")
            self.pub_err.publish(0.0)
            self.publish_cmd(0.0, 0.0)
            return

        lane_center_px = 0.5 * (left_idx + right_idx)
        img_center_px  = 0.5 * w
        err_px = (img_center_px - lane_center_px)    # >0: линия правее центра → рулить вправо

        
        err = float(err_px) / (w/2.0)           
        t  = rospy.get_time()
        dt = max(1e-3, t - self.prev_t)
        d_err = (err - self.prev_err) / dt
        self.prev_err, self.prev_t = err, t

        
        m = cv2.moments(mask, binaryImage=True)
        heading = 0.0
        if m['m00'] > 1e3:
            cx = m['m10']/m['m00']  
            heading = - ( (cx - img_center_px) / (w/2.0) ) 

        
        steer = self.kp*err + self.kd*d_err + self.k_heading*heading
        speed = self.target_speed

        
        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.line(dbg, (int(lane_center_px), y_scan), (int(lane_center_px), max(0, y_scan-40)), (0,0,255), 2)
        cv2.line(dbg, (int(img_center_px),  y_scan), (int(img_center_px),  max(0, y_scan-40)), (255,0,0), 1)
        txt = f"nz={nz_count} err={err:+.2f} de={d_err:+.2f} hd={heading:+.2f} st={steer:+.2f} v={speed:.2f}"
        cv2.putText(dbg, txt, (10, dbg.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
        self._publish_debug(dbg, w, h, y0)

        
        self.pub_err.publish(float(err))
        self.publish_cmd(speed, steer)

    def _publish_debug(self, roi_bgr_or_mask, w, h, y0, label=None):
        """Вклеиваем ROI обратно в холст исходного размера и шлём /vision_lka/debug"""
        if len(roi_bgr_or_mask.shape) == 2:
            roi_bgr = cv2.cvtColor(roi_bgr_or_mask, cv2.COLOR_GRAY2BGR)
        else:
            roi_bgr = roi_bgr_or_mask
        canv = np.zeros((h, w, 3), dtype=np.uint8)
        canv[y0:h, :] = cv2.resize(roi_bgr, (w, h - y0))
        if label:
            cv2.putText(canv, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)
        try:
            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(canv, encoding='bgr8'))
        except Exception as e:
            rospy.logwarn_throttle(2.0, "debug publish failed: %s", e)

if __name__ == "__main__":
    VisionLKA()
    rospy.spin()

