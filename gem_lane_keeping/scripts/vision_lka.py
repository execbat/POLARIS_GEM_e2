#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision-based lane keeping for GEM simulator.
- Robust color mask (HSV/HLS white + LAB yellow)
- Multi-row scan of lane edges → lane center
- PID (P+I+D) on lateral error + heading feed-forward from slope
- Leaky integral with anti-windup
- Steering low-pass smoothing
- Speed reduction on curves & large steering
- Safety stop respected
Publishes:
  - cmd  : ackermann_msgs/AckermannDrive
  - debug: sensor_msgs/Image (overlay)
  - lateral_error: std_msgs/Float32
Subscribes:
  - image: sensor_msgs/Image (remap to /gem/front_single_camera/image_raw)
  - /gem/safety/stop: std_msgs/Bool
"""
import rospy, cv2, numpy as np
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge

def clamp(v, lo, hi): return lo if v < lo else hi if v > hi else v

class VisionLKA:
    def __init__(self):
        rospy.init_node("vision_lka")
        self.bridge = CvBridge()

        # ===== Control gains (PID) =====
        self.kp        = rospy.get_param("~kp",        0.025)  # P
        self.kd        = rospy.get_param("~kd",        0.030)  # D (keep modest)
        self.ki        = rospy.get_param("~ki",        0.004)  # I (NEW)
        self.i_decay   = rospy.get_param("~i_decay",   0.02)   # 0..0.1, leaky integral
        self.k_heading = rospy.get_param("~k_heading", 0.050)  # feed-forward from slope

        # ===== Steering shaping =====
        self.steer_limit  = rospy.get_param("~steer_limit", 0.90)
        self.steer_alpha  = rospy.get_param("~steer_alpha", 0.35)  # 0..1 EMA for steering
        self.steer_sign   = rospy.get_param("~steer_sign",  1.0)   # set -1 if steering is flipped

        # ===== Speed profile =====
        self.target_speed   = rospy.get_param("~target_speed", 1.2)
        self.min_speed      = rospy.get_param("~min_speed",    0.7)
        self.steer_slowdown = rospy.get_param("~steer_slowdown", 0.15)  # scale with |steer|
        self.k_curve_speed  = rospy.get_param("~k_curve_speed", 1.8)    # scale with curvature

        # ===== Color/mask params =====
        self.s_thresh   = rospy.get_param("~s_thresh", 100)  # HSV S upper bound for white
        self.v_thresh   = rospy.get_param("~v_thresh",  35)  # HSV V lower bound for white
        self.hls_L_min  = rospy.get_param("~hls_L_min", 190) # HLS L lower bound (white)
        self.lab_b_min  = rospy.get_param("~lab_b_min", 140) # LAB b lower bound (yellow)

        # ===== Geometry / scans =====
        self.roi_top     = rospy.get_param("~roi_top", 0.62)                 # ROI starts here (fraction of image height)
        self.scan_rows   = rospy.get_param("~scan_rows", [0.60, 0.70, 0.80, 0.90])
        self.min_mask_px = rospy.get_param("~min_mask_px", 150)
        self.min_lane_w  = rospy.get_param("~min_lane_width_px", 22)
        self.min_valid_rows = rospy.get_param("~min_valid_rows", 2)
        self.hold_bad_ms = rospy.get_param("~hold_bad_ms", 400)              # hold last cmd when lane lost

        # ===== State =====
        self.estop        = False
        self.prev_err     = 0.0
        self.prev_t       = rospy.get_time()
        self.int_err      = 0.0        # integral state
        self.steer_filt   = 0.0        # smoothed steering
        self.last_ok_time = rospy.Time(0.0)
        self.last_cmd     = AckermannDrive()

        # ===== I/O =====
        self.pub_cmd = rospy.Publisher("cmd", AckermannDrive, queue_size=10)
        self.pub_err = rospy.Publisher("lateral_error", Float32, queue_size=10)
        self.pub_dbg = rospy.Publisher("debug", Image, queue_size=1)
        rospy.Subscriber("image", Image, self.on_image, queue_size=1)
        rospy.Subscriber("/gem/safety/stop", Bool, self.on_stop, queue_size=1)

        rospy.loginfo("vision_lka: PID P=%.3f I=%.3f D=%.3f leak=%.3f, k_heading=%.3f, "
                      "steer_lim=%.2f alpha=%.2f sign=%.1f, v=%.2f..%.2f curve=%.2f slow=%.2f",
                      self.kp, self.ki, self.kd, self.i_decay, self.k_heading,
                      self.steer_limit, self.steer_alpha, self.steer_sign,
                      self.min_speed, self.target_speed, self.k_curve_speed, self.steer_slowdown)

    # ---------- callbacks ----------
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
        # 1) image → ROI
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "cv_bridge: %s", e); return
        h, w = bgr.shape[:2]
        y0 = int(h * self.roi_top); y0 = min(max(y0, 0), h-2)
        roi = bgr[y0:h, :]

        # 2) robust mask (white & yellow)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

        white_hsv  = cv2.inRange(hsv, (0, 0, self.v_thresh), (179, self.s_thresh, 255))
        white_hls  = cv2.inRange(hls[:, :, 1], self.hls_L_min, 255)
        yellow_lab = cv2.inRange(lab[:, :, 2], self.lab_b_min, 255)

        mask = cv2.bitwise_or(white_hsv, white_hls)
        mask = cv2.bitwise_or(mask, yellow_lab)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.medianBlur(mask, 5)

        nz_count = int(np.count_nonzero(mask))
        if nz_count < self.min_mask_px:
            self._lost_and_hold(mask, w, h, y0, "NO LANE (area)")
            return

        # 3) multi-row scan → lane centers
        ys = mask.shape[0]
        centers, used_y = [], []
        for r in self.scan_rows:
            y_scan = int(ys * float(r))
            y_scan = 0 if y_scan < 0 else ys-1 if y_scan >= ys else y_scan
            xs = np.where(mask[y_scan, :] > 0)[0]
            if xs.size >= self.min_lane_w:
                left_idx, right_idx = int(xs[0]), int(xs[-1])
                if (right_idx - left_idx) >= self.min_lane_w:
                    centers.append(0.5 * (left_idx + right_idx))
                    used_y.append(y_scan)
        if len(centers) < self.min_valid_rows:
            self._lost_and_hold(mask, w, h, y0, "NO LANE (rows)")
            return

        # 4) lateral error (normalized) + heading from slope
        img_center_px  = 0.5 * w
        lane_center_px = float(np.mean(centers[-max(1, len(centers)//2):]))  # bottom half scans
        err = (img_center_px - lane_center_px) / (w * 0.5)  # >0 -> steer left

        z = np.polyfit(np.array(used_y, dtype=np.float32),
                       np.array(centers, dtype=np.float32), 1)
        slope = float(z[0])                   # px per row
        heading = slope / (w * 0.5)           # normalize to ~[-1..1]

        # 5) PID + heading
        t  = rospy.get_time()
        dt = max(1e-3, t - self.prev_t)
        d_err = (err - self.prev_err) / dt
        self.prev_err, self.prev_t = err, t

        # Leaky integrator with anti-windup
        self.int_err = (1.0 - self.i_decay) * self.int_err + err * dt
        self.int_err = clamp(self.int_err, -0.6, 0.6)

        steer_cmd = (self.kp * err) + (self.kd * d_err) + (self.ki * self.int_err) + (self.k_heading * heading)
        steer_cmd = clamp(steer_cmd, -self.steer_limit, self.steer_limit)
        steer_cmd *= self.steer_sign

        # Smooth steering (EMA)
        self.steer_filt = (1.0 - self.steer_alpha) * self.steer_filt + self.steer_alpha * steer_cmd

        # 6) speed schedule: slow down on curves / big steering
        curve = abs(heading)
        v_curve = 1.0 / (1.0 + self.k_curve_speed * curve)
        v_steer = 1.0 - self.steer_slowdown * abs(self.steer_filt)
        v_scale = max(0.25, min(v_curve, v_steer))
        speed   = clamp(self.target_speed * v_scale, self.min_speed, self.target_speed)

        # 7) publish
        self.pub_err.publish(float(err))
        self.publish_cmd(speed, self.steer_filt)
        self.last_ok_time = rospy.Time.now()
        self.last_cmd.speed = float(speed)
        self.last_cmd.steering_angle = float(self.steer_filt)

        # 8) debug overlay
        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for y_scan, cx in zip(used_y, centers):
            cv2.line(dbg, (int(cx), y_scan), (int(cx), max(0, y_scan - 30)), (0, 0, 255), 2)
        cv2.line(dbg, (int(img_center_px), ys - 5), (int(img_center_px), ys - 55), (255, 0, 0), 1)
        txt = f"err={err:+.2f} de={d_err:+.2f} I={self.int_err:+.2f} hd={heading:+.2f} st={self.steer_filt:+.2f} v={speed:.2f}"
        cv2.putText(dbg, txt, (10, dbg.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        self._publish_debug(dbg, w, h, y0)

    # ---------- helpers ----------
    def _lost_and_hold(self, mask, w, h, y0, label):
        # Hold last good command for a short time, then stop softly
        if (rospy.Time.now() - self.last_ok_time).to_sec() * 1000.0 < self.hold_bad_ms:
            self.publish_cmd(self.last_cmd.speed, self.last_cmd.steering_angle)
        else:
            self.publish_cmd(0.0, 0.0)
        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(dbg, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        self._publish_debug(dbg, w, h, y0)

    def _publish_debug(self, roi_bgr, w, h, y0):
        canv = np.zeros((h, w, 3), dtype=np.uint8)
        canv[y0:h, :] = cv2.resize(roi_bgr, (w, h - y0))
        try:
            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(canv, encoding='bgr8'))
        except Exception as e:
            rospy.logwarn_throttle(2.0, "debug publish failed: %s", e)

if __name__ == "__main__":
    VisionLKA()
    rospy.spin()

