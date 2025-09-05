#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grass-balance Lane Keeping (ROS1 / Noetic)

Idea:
  - Segment "grass" (green shoulders) using HSV + LAB.
  - Compute weighted grass area LEFT vs RIGHT in a bottom ROI.
  - Use PD(+I) controller to steer such that areas are equal (vehicle centered).
  - Slow down when steering is large or confidence is low.
  - If segmentation confidence is bad, hold the last cmd briefly, then soft-stop.
  - Respect /gem/safety/stop (estop).

Topics (REMAPPED in launch):
  Sub:   image  (sensor_msgs/Image)
  Pub:   cmd    (ackermann_msgs/AckermannDrive)
         lateral_error (std_msgs/Float32)  # area_right - area_left (normalized)
         debug  (sensor_msgs/Image)        # visualization of ROI+mask

Author: you + ChatGPT
"""

import rospy
import cv2
import numpy as np
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

        # ---------------- Control & speed governor ----------------
        self.target_speed = rospy.get_param("~target_speed", 1.6)
        self.min_speed    = rospy.get_param("~min_speed",   0.6)
        self.steer_limit  = rospy.get_param("~steer_limit", 0.55)

        # PD(+I) on area-difference error (right - left)
        self.kp = rospy.get_param("~kp",  1.10)     # main proportional gain
        self.kd = rospy.get_param("~kd",  0.20)     # derivative (on error)
        self.ki = rospy.get_param("~ki",  0.00)     # optional integral (keep small)
        self.deadband = rospy.get_param("~deadband", 0.02)  # ignore tiny error

        # Speed scaling based on steering magnitude and confidence
        self.steer_slowdown = rospy.get_param("~steer_slowdown", 0.85)  # 0..1 multiplier at max steer
        self.conf_floor     = rospy.get_param("~conf_floor",     0.55)  # min confidence speed factor (0..1)

        # ---------------- ROI & weighting ----------------
        self.roi_top   = rospy.get_param("~roi_top",   0.56)  # fraction of image height
        self.roi_bot   = rospy.get_param("~roi_bot",   1.00)  # 1.0 = bottom
        self.gamma_y   = rospy.get_param("~gamma_y",   1.6)   # emphasize lower rows (near-field)
        # Optional inner "ignore strip" around the center to reduce central bias (0..0.2)
        self.center_ignore_frac = rospy.get_param("~center_ignore_frac", 0.00)

        # ---------------- Grass segmentation (HSV + LAB) ----------------
        # HSV green band (typical grass)
        self.h_lo = rospy.get_param("~h_lo", 35)
        self.h_hi = rospy.get_param("~h_hi", 95)
        self.s_lo = rospy.get_param("~s_lo", 50)
        self.v_lo = rospy.get_param("~v_lo", 40)

        # LAB a-channel (green has smaller 'a'); we keep pixels with a <= a_max
        self.lab_a_max = rospy.get_param("~lab_a_max", 140)

        # Morphology
        self.close_kernel = rospy.get_param("~close_kernel", 7)  # closing
        self.open_kernel  = rospy.get_param("~open_kernel",  3)  # opening
        self.blur_ksize   = rospy.get_param("~blur_ksize",   5)  # median blur

        # ---------------- Confidence & safety ----------------
        self.min_grass_ratio = rospy.get_param("~min_grass_ratio", 0.02)  # < 2% = too little
        self.max_grass_ratio = rospy.get_param("~max_grass_ratio", 0.70)  # >70% = too much
        self.hold_bad_ms     = rospy.get_param("~hold_bad_ms", 600)        # hold last cmd this long

        # EMA smoothing on area-diff (stabilize)
        self.diff_alpha   = rospy.get_param("~diff_alpha", 0.35)  # 0..1 (higher = less smoothing)
        self.diff_ema     = 0.0

        # ---------------- State ----------------
        self.estop     = False
        self.prev_err  = 0.0
        self.i_err     = 0.0
        self.prev_t    = rospy.get_time()
        self.last_ok_t = rospy.Time(0.0)
        self.last_cmd  = AckermannDrive()

        # ---------------- IO ----------------
        self.pub_cmd = rospy.Publisher("cmd", AckermannDrive, queue_size=10)
        self.pub_err = rospy.Publisher("lateral_error", Float32, queue_size=10)
        self.pub_dbg = rospy.Publisher("debug", Image, queue_size=1)
        rospy.Subscriber("image", Image, self.on_image, queue_size=1)           # remap in launch
        rospy.Subscriber("/gem/safety/stop", Bool, self.on_stop, queue_size=1)

        rospy.loginfo(
            "Grass-LKA params: v=%.2f..%.2f steer_lim=%.2f kp=%.2f kd=%.2f ki=%.3f "
            "roi=[%.2f..%.2f] gamma=%.2f HSV(H:%d..%d,S>=%d,V>=%d) LAB(a<=%d) "
            "kernels(close=%d,open=%d,blur=%d) grass_ratio[%.2f..%.2f] hold=%dms",
            self.min_speed, self.target_speed, self.steer_limit, self.kp, self.kd, self.ki,
            self.roi_top, self.roi_bot, self.gamma_y,
            self.h_lo, self.h_hi, self.s_lo, self.v_lo, self.lab_a_max,
            self.close_kernel, self.open_kernel, self.blur_ksize,
            self.min_grass_ratio, self.max_grass_ratio, self.hold_bad_ms
        )

    # ---------------- Callbacks ----------------
    def on_stop(self, msg: Bool):
        self.estop = bool(msg.data)

    def on_image(self, msg: Image):
        # ---- Convert and crop ROI ----
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "cv_bridge error: %s", e)
            return

        H, W = bgr.shape[:2]
        y0 = int(np.clip(self.roi_top * H, 0, H - 2))
        y1 = int(np.clip(self.roi_bot * H, y0 + 2, H))
        roi = bgr[y0:y1, :]
        h, w = roi.shape[:2]

        # ---- Grass mask (HSV + LAB) ----
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

        mask_h = cv2.inRange(hsv[:, :, 0], self.h_lo, self.h_hi)
        mask_s = cv2.inRange(hsv[:, :, 1], self.s_lo, 255)
        mask_v = cv2.inRange(hsv[:, :, 2], self.v_lo, 255)
        mask_hsv = cv2.bitwise_and(mask_h, cv2.bitwise_and(mask_s, mask_v))

        mask_lab = cv2.inRange(lab[:, :, 1], 0, self.lab_a_max)  # a-channel

        grass = cv2.bitwise_or(mask_hsv, mask_lab)

        # Morphology: close → open → blur
        if self.close_kernel > 1:
            k = np.ones((self.close_kernel, self.close_kernel), np.uint8)
            grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, k)
        if self.open_kernel > 1:
            k = np.ones((self.open_kernel, self.open_kernel), np.uint8)
            grass = cv2.morphologyEx(grass, cv2.MORPH_OPEN, k)
        if self.blur_ksize >= 3 and self.blur_ksize % 2 == 1:
            grass = cv2.medianBlur(grass, self.blur_ksize)

        # ---- Confidence check ----
        grass_ratio = float(np.count_nonzero(grass)) / float(max(1, grass.size))
        conf_ok = (self.min_grass_ratio <= grass_ratio <= self.max_grass_ratio)

        # Build vertical weights to emphasize near field (bottom of ROI)
        yy = (np.arange(h).astype(np.float32) / max(1.0, h - 1))  # 0..1
        weights_y = np.power(yy, self.gamma_y).reshape(h, 1)

        # Optional: ignore a narrow center strip to reduce "center bias"
        center_ignore = np.zeros((1, w), np.uint8)
        if self.center_ignore_frac > 1e-3:
            cx = int(0.5 * w)
            half = int(0.5 * self.center_ignore_frac * w)
            center_ignore[:, max(0, cx - half):min(w, cx + half)] = 1
        # Apply center ignore by zeroing mask in that band (optional)
        if self.center_ignore_frac > 1e-3:
            grass = np.where(center_ignore, 0, grass)

        # ---- Weighted areas left vs right ----
        left_half  = np.s_[:, :w // 2]
        right_half = np.s_[:, w // 2:]

        # Convert grass to {0,1} float to apply weights
        g = (grass > 0).astype(np.float32)
        g_w = g * weights_y  # apply vertical weights

        area_left  = float(np.sum(g_w[left_half]))
        area_right = float(np.sum(g_w[right_half]))

        total = area_left + area_right
        if total <= 1e-6 or not conf_ok:
            # Vision unreliable: hold last cmd briefly, then stop
            self._hold_or_stop("LOW CONF")
            self._publish_debug(self._make_debug(bgr, y0, y1, grass, area_left, area_right, None, 0.0, 0.0, 0.0), H, W)
            return

        # ---- Error = normalized (right - left) ----
        diff = (area_right - area_left) / total  # ∈ [-1..1]
        # EMA smoothing to reduce flicker
        self.diff_ema = self.diff_alpha * diff + (1.0 - self.diff_alpha) * self.diff_ema
        err = self.diff_ema

        # Deadband
        if abs(err) < self.deadband:
            err = 0.0

        # ---- PD(+I) steering ----
        t = rospy.get_time()
        dt = max(1e-3, t - self.prev_t)
        d_err = (err - self.prev_err) / dt
        self.prev_err, self.prev_t = err, t

        # Integrator (optional; reset when confidence is marginal)
        if self.ki > 0.0 and conf_ok:
            self.i_err = float(np.clip(self.i_err + err * dt, -0.8, 0.8))
        else:
            self.i_err = 0.0

        steer = self.kp * err + self.kd * d_err + self.ki * self.i_err
        steer = clamp(steer, -self.steer_limit, self.steer_limit)

        # ---- Speed governor ----
        steer_ratio = abs(steer) / max(1e-6, self.steer_limit)  # 0..1
        # confidence factor → closer to 1.0 when grass_ratio in mid range
        conf_span = (self.max_grass_ratio - self.min_grass_ratio)
        conf_pos  = np.clip((grass_ratio - self.min_grass_ratio) / max(1e-6, conf_span), 0.0, 1.0)
        conf_factor = self.conf_floor + (1.0 - self.conf_floor) * conf_pos  # [conf_floor..1]

        speed = self.target_speed * (1.0 - self.steer_slowdown * steer_ratio) * conf_factor
        speed = clamp(speed, self.min_speed, self.target_speed)

        # Respect estop
        if self.estop:
            speed = 0.0
            steer = 0.0

        # Publish
        cmd = AckermannDrive()
        cmd.speed = float(speed)
        cmd.steering_angle = float(steer)
        self.pub_cmd.publish(cmd)
        self.pub_err.publish(float(err))
        self.last_cmd = cmd
        self.last_ok_t = rospy.Time.now()

        # Debug image
        dbg = self._make_debug(bgr, y0, y1, grass, area_left, area_right, err, steer, speed, grass_ratio)
        self._publish_debug(dbg, H, W)

    # ---------------- Helpers ----------------
    def _hold_or_stop(self, reason: str):
        # Hold the last command for hold_bad_ms; then stop softly
        if (rospy.Time.now() - self.last_ok_t).to_sec() * 1000.0 < self.hold_bad_ms:
            self.pub_cmd.publish(self.last_cmd)
        else:
            stop = AckermannDrive()
            self.pub_cmd.publish(stop)

    def _make_debug(self, bgr, y0, y1, grass_mask, area_L, area_R, err, steer, speed, ratio):
        """Build a full-frame debug canvas with ROI overlay and annotations."""
        H, W = bgr.shape[:2]
        canv = bgr.copy()

        # Colorize grass mask (green overlay) inside ROI
        roi_h, roi_w = grass_mask.shape[:2]
        mask_color = np.zeros((roi_h, roi_w, 3), np.uint8)
        mask_color[:, :, 1] = grass_mask  # G channel
        overlay = cv2.addWeighted(canv[y0:y1, :], 1.0, mask_color, 0.45, 0)
        canv[y0:y1, :] = overlay

        # Draw split line at image center
        cx = W // 2
        cv2.line(canv, (cx, y0), (cx, y1), (255, 0, 0), 1)

        # Optional center ignore strip
        if self.center_ignore_frac > 1e-3:
            half = int(0.5 * self.center_ignore_frac * W)
            cv2.rectangle(canv, (cx - half, y0), (cx + half, y1), (255, 0, 255), 1)

        # Text annotations
        ytxt = 24
        def put(s):  # helper
            nonlocal ytxt
            cv2.putText(canv, s, (10, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            ytxt += 22

        put("Grass-LKA (area-balance)")
        put(f"areas L={area_L:.0f}  R={area_R:.0f}  ratio={ratio:.2f}")
        if err is not None:
            put(f"err={err:+.3f}  steer={steer:+.2f}  v={speed:.2f} m/s")

        return canv

    def _publish_debug(self, bgr_full, H, W):
        try:
            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(bgr_full, encoding='bgr8'))
        except Exception as e:
            rospy.logwarn_throttle(2.0, "debug publish failed: %s", e)


if __name__ == "__main__":
    VisionLKA()
    rospy.spin()

