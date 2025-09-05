#!/usr/bin/env python3
import rospy, cv2, numpy as np, math, time
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool
from ackermann_msgs.msg import AckermannDrive
from cv_bridge import CvBridge

def clamp(v, lo, hi): return max(lo, min(hi, v))

class VisionLKA:
    def __init__(self):
        rospy.init_node("vision_lka")
        
        self.target_speed = rospy.get_param("~target_speed", 1.2)
        self.min_speed    = rospy.get_param("~min_speed", 0.5)       # speed при fallback
        self.kp           = rospy.get_param("~kp", 0.015)             # по боковой ошибке
        self.kd           = rospy.get_param("~kd", 0.06)              # демпфирование
        self.k_heading    = rospy.get_param("~k_heading", 0.005)      # за угол линии
        self.steer_limit  = rospy.get_param("~steer_limit", 0.5)      # рад
        self.s_thresh     = rospy.get_param("~s_thresh", 80)          # для белого (низкая S)
        self.v_thresh     = rospy.get_param("~v_thresh", 40)          # для белого (высокая V)
        self.roi_top      = rospy.get_param("~roi_top", 0.55)         # доля от высоты кадра
        self.min_mask_px  = rospy.get_param("~min_mask_px", 300)      # минимум пикс. в маске

        self.bridge = CvBridge()
        self.pub_cmd   = rospy.Publisher("cmd",   AckermannDrive, queue_size=10)
        self.pub_dbg   = rospy.Publisher("debug", Image,          queue_size=1)
        self.pub_error = rospy.Publisher("lateral_error", Float32, queue_size=1)

        rospy.Subscriber("image", Image, self.on_image, queue_size=1)
        rospy.Subscriber("/gem/safety/stop", Bool, self.on_stop, queue_size=1)

        self.estop = False
        self.prev_err = 0.0
        self.prev_t   = rospy.get_time()

        rospy.loginfo("vision_lka: ready (ROI top=%.2f, S<%d, V>%d)", self.roi_top, self.s_thresh, self.v_thresh)

    def on_stop(self, msg: Bool):
        self.estop = bool(msg.data)

    def publish_cmd(self, speed, steer):
        if self.estop:
            speed = 0.0; steer = 0.0
        cmd = AckermannDrive()
        cmd.speed = float(speed)
        cmd.steering_angle = float(clamp(steer, -self.steer_limit, self.steer_limit))
        self.pub_cmd.publish(cmd)

    def on_image(self, msg: Image):
        
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "vision_lka: cv_bridge fail: %s", e); return

        h, w = bgr.shape[:2]
        y0 = int(h * self.roi_top)
        roi = bgr[y0:h, :]

      
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    
        white = cv2.inRange(hsv, (0, 0, self.v_thresh), (179, self.s_thresh, 255))

 
        yellow = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))

        mask = cv2.bitwise_or(white, yellow)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        mask = cv2.medianBlur(mask, 5)

        nz = np.column_stack(np.nonzero(mask))  # (y,x) точки
        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

  
        #if nz.shape[0] < self.min_mask_px:
        #    
        #    cv2.putText(dbg, "FALLBACK", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)
        #    self._publish_debug(dbg, y0, h, w)
        #    self.publish_cmd(max(self.min_speed, self.target_speed*0.5), 0.0)
        #    self.pub_error.publish(Float32(data=0.0))
        #    return

        
        M = cv2.moments(mask)
        cx = int(M['m10']/M['m00']) if M['m00'] != 0 else w//2
        err_px = cx - (w//2)
        err = float(err_px) / (w/2) 

        # derection
        vx, vy, x0f, y0f = cv2.fitLine(nz.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
       
        heading = -math.atan2(float(vx), float(vy))  

        # PD 
        t = rospy.get_time()
        dt = max(1e-3, t - self.prev_t)
        d_err = (err - self.prev_err) / dt
        steer = self.kp*err + self.kd*d_err + self.k_heading*heading
        self.prev_err, self.prev_t = err, t

        # speed adjust
        speed = self.target_speed * (1.0 - 0.4*min(1.0, abs(err)))

      
        cv2.line(dbg, (w//2, 0), (w//2, dbg.shape[0]-1), (0,0,255), 1)      
        cv2.line(dbg, (cx, 0), (cx, dbg.shape[0]-1), (255,0,0), 2)          
        
        x1 = int(x0f - 1000*vx); y1 = int(y0f - 1000*vy)
        x2 = int(x0f + 1000*vx); y2 = int(y0f + 1000*vy)
        cv2.line(dbg, (x1,y1), (x2,y2), (0,255,0), 2)
        txt = f"err={err:+.2f} hd={heading:+.2f} steer={steer:+.2f} spd={speed:.2f}"
        cv2.putText(dbg, txt, (10, dbg.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,255,255),2,cv2.LINE_AA)

        self._publish_debug(dbg, y0, h, w)
        self.pub_error.publish(Float32(data=err))
        self.publish_cmd(speed, steer)

    def _publish_debug(self, roi_bgr, y0, h, w):
        
        canv = np.zeros((h, w, 3), dtype=np.uint8)
        canv[y0:h, :] = roi_bgr
        try:
            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(canv, encoding='bgr8'))
        except Exception as e:
            rospy.logwarn_throttle(2.0, "vision_lka: debug publish failed: %s", e)

if __name__ == "__main__":
    VisionLKA()
    rospy.spin()

