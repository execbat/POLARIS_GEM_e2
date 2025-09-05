#!/usr/bin/env python3
import rospy, cv2, numpy as np
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

class VisionLKA:
    def __init__(self):
        rospy.init_node("vision_lka")
        self.bridge = CvBridge()

        # --- control params ---
        self.target_speed = rospy.get_param("~target_speed", 1.5)
        self.kp           = rospy.get_param("~kp", 0.020)
        self.kd           = rospy.get_param("~kd", 0.060)
        self.k_heading    = rospy.get_param("~k_heading", 0.012)
        self.steer_limit  = rospy.get_param("~steer_limit", 0.40)

        # --- color/mask ---
        # HSV-white 
        self.s_thresh     = rospy.get_param("~s_thresh", 100)
        self.v_thresh     = rospy.get_param("~v_thresh", 35)
        # HLS-white 
        self.hls_L_min    = rospy.get_param("~hls_L_min", 190)
        # LAB-yellow 
        self.lab_b_min    = rospy.get_param("~lab_b_min", 140)

        # --- scan geometry ---
        self.roi_top      = rospy.get_param("~roi_top", 0.62)               # верх ROI (доля высоты кадра)
        self.scan_rows    = rospy.get_param("~scan_rows", [0.60, 0.70, 0.80, 0.90])  # scan scales by height
        self.min_mask_px  = rospy.get_param("~min_mask_px", 150)            # min masked square
        self.min_lane_w   = rospy.get_param("~min_lane_width_px", 22)       # min wide in pixels
        self.min_valid_rows = rospy.get_param("~min_valid_rows", 2)         # min number of valid lines
        self.hold_bad_ms  = rospy.get_param("~hold_bad_ms", 400)            # hold last cmd

        # --- состояние ---
        self.estop        = False
        self.prev_err     = 0.0
        self.prev_t       = rospy.get_time()
        self.last_ok_time = rospy.Time(0.0)
        self.last_cmd     = AckermannDrive()

        # --- I/O ---
        self.pub_cmd = rospy.Publisher("cmd", AckermannDrive, queue_size=10)
        self.pub_err = rospy.Publisher("lateral_error", Float32, queue_size=10)
        self.pub_dbg = rospy.Publisher("debug", Image, queue_size=1)
        rospy.Subscriber("image", Image, self.on_image, queue_size=1)
        rospy.Subscriber("/gem/safety/stop", Bool, self.on_stop, queue_size=1)

        rospy.loginfo("vision_lka params: ts=%.2f kp=%.3f kd=%.3f kh=%.3f steer=%.2f "
                      "HSV(S<=%d,V>=%d) HLS(L>=%d) LAB(b>=%d) roi=%.2f scans=%s "
                      "min_mask=%d min_w=%d min_rows=%d hold=%dms",
                      self.target_speed, self.kp, self.kd, self.k_heading, self.steer_limit,
                      self.s_thresh, self.v_thresh, self.hls_L_min, self.lab_b_min,
                      self.roi_top, str(self.scan_rows),
                      self.min_mask_px, self.min_lane_w, self.min_valid_rows, self.hold_bad_ms)

    # -------------------- callbacks --------------------

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
            rospy.logwarn_throttle(2.0, "cv_bridge err: %s", e)
            return

        h, w = bgr.shape[:2]
        y0 = int(h * self.roi_top)
        y0 = max(0, min(h - 2, y0))
        roi = bgr[y0:h, :]

        # masks: HSV-white OR HLS-white (L high И S low), AND LAB-yellow
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

        white_hsv = cv2.inRange(hsv, (0, 0, self.v_thresh), (179, self.s_thresh, 255))

        L = hls[:, :, 1]
        S = hls[:, :, 2]
        white_hls_L = cv2.inRange(L, self.hls_L_min, 255)
        white_hls_S = cv2.inRange(S, 0, 60)               # низкая насыщенность
        white_hls = cv2.bitwise_and(white_hls_L, white_hls_S)

        yellow_lab = cv2.inRange(lab[:, :, 2], self.lab_b_min, 255)

        mask = cv2.bitwise_or(white_hsv, white_hls)
        mask = cv2.bitwise_or(mask, yellow_lab)

        # control of mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        mask = cv2.medianBlur(mask, 3)

        nz_count = int(np.count_nonzero(mask))
        if nz_count < self.min_mask_px:
            # hold last cmd, stop after
            if (rospy.Time.now() - self.last_ok_time).to_sec() * 1000.0 < self.hold_bad_ms:
                self.publish_cmd(self.last_cmd.speed, self.last_cmd.steering_angle)
            else:
                self.pub_err.publish(0.0)
                self.publish_cmd(0.0, 0.0)
            self._debug_and_stop(mask, w, h, y0, "NO LANE (area)")
            return

       # multiscan
        ys, xs_w = mask.shape[0], mask.shape[1]
        centers, used_y = [], []
        mid = xs_w // 2
        for r in self.scan_rows:
            y_scan = int(ys * float(r))
            y_scan = np.clip(y_scan, 0, ys - 1)
            scan = mask[y_scan, :]
    
            left_xs  = np.where(scan[:mid] > 0)[0]     # pixels from left
            right_xs = np.where(scan[mid:] > 0)[0]     # pixels from right

            if left_xs.size > 0 and right_xs.size > 0:
                left_idx  = int(left_xs[-1])           # right edge on left
                right_idx = int(mid + right_xs[0])     # left edge on right
                if (right_idx - left_idx) >= self.min_lane_w:
                    centers.append(0.5 * (left_idx + right_idx))
                    used_y.append(y_scan)

        if len(centers) < self.min_valid_rows:
            if (rospy.Time.now() - self.last_ok_time).to_sec() * 1000.0 < self.hold_bad_ms:
                self.publish_cmd(self.last_cmd.speed, self.last_cmd.steering_angle)
            else:
                self.pub_err.publish(0.0)
                self.publish_cmd(0.0, 0.0)
            self._debug_and_stop(mask, w, h, y0, "NO LANE (rows)")
            return

        # Lateral errors
        lane_center_px = float(np.mean(centers[-max(1, len(centers)//2):]))
        img_center_px  = 0.5 * w
        err = (img_center_px - lane_center_px) / (w * 0.5)

        heading = 0.0
        if len(centers) >= 2:
            z = np.polyfit(np.array(used_y, dtype=np.float32),
                           np.array(centers, dtype=np.float32), 1)
            slope = z[0]
            heading = float(slope) / (w * 0.5)

        # PD + heading
        t = rospy.get_time()
        dt = max(1e-3, t - self.prev_t)
        d_err = (err - self.prev_err) / dt
        self.prev_err, self.prev_t = err, t

        steer = self.kp * err + self.kd * d_err + self.k_heading * heading
        speed = self.target_speed
    
        # publishers and memorisina last cmd
        self.pub_err.publish(float(err))
        self.publish_cmd(speed, steer)
        self.last_ok_time = rospy.Time.now()
        self.last_cmd = AckermannDrive(speed=float(speed),
                                       steering_angle=float(clamp(steer, -self.steer_limit, self.steer_limit)))

        # debug
        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for y_scan, cx in zip(used_y, centers):
            cv2.line(dbg, (int(cx), y_scan), (int(cx), max(0, y_scan - 30)), (0, 0, 255), 2)
        cv2.line(dbg, (int(img_center_px), ys - 5), (int(img_center_px), ys - 50), (255, 0, 0), 1)
        txt = f"nz={nz_count} err={err:+.2f} de={d_err:+.2f} hd={heading:+.3f} st={steer:+.2f} v={speed:.2f}"
        cv2.putText(dbg, txt, (10, dbg.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        self._publish_debug(dbg, w, h, y0)


    # -------------------- helpers --------------------

    def _debug_and_stop(self, mask, w, h, y0, label):
        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(dbg, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        self._publish_debug(dbg, w, h, y0)
        

    def _publish_debug(self, roi_bgr, w, h, y0):
        canv = np.zeros((h, w, 3), dtype=np.uint8)
        canv[y0:h, :] = cv2.resize(roi_bgr, (w, h - y0))
        try:
            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(canv, encoding='bgr8'))
        except Exception as e:
            rospy.logwarn_throttle(2.0, "vision_lka: debug publish failed: %s", e)

if __name__ == "__main__":
    VisionLKA()
    rospy.spin()

