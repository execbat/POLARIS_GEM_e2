#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grass-balance Lane Keeping (fixed gating)

Key changes vs previous:
  - Only LOWER confidence bound is enforced (min_grass_ratio). Upper bound no longer stops the car.
  - Wider default HSV/LAB for grass.
  - Extra telemetry with rospy.loginfo_throttle for ratio/areas/error.
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

        # ---------------- Control ----------------
        self.target_speed = rospy.get_param("~target_speed", 1.6)
        self.min_speed    = rospy.get_param("~min_speed",   0.6)
        self.steer_limit  = rospy.get_param("~steer_limit", 0.55)

        # PD(+I) on area-difference error (right - left)
        self.kp = rospy.get_param("~kp",  1.10)
        self.kd = rospy.get_param("~kd",  0.20)
        self.ki = rospy.get_param("~ki",  0.00)
        self.deadband = rospy.get_param("~deadband", 0.02)

        # Speed scaling with steering + confidence
        self.steer_slowdown = rospy.get_param("~steer_slowdown", 0.85)
        self.conf_floor     = rospy.get_param("~conf_floor",     0.55)

        # ---------------- ROI & weights ----------------
        self.roi_top   = rospy.get_param("~roi_top", 0.54)
        self.roi_bot   = rospy.get_param("~roi_bot", 1.00)
        self.gamma_y   = rospy.get_param("~gamma_y", 1.6)
        self.center_ignore_frac = rospy.get_param("~center_ignore_frac", 0.00)

        # ---------------- Grass segmentation ----------------
        # Wider defaults
        self.h_lo = rospy.get_param("~h_lo", 28)
        self.h_hi = rospy.get_param("~h_hi", 100)
        self.s_lo = rospy.get_param("~s_lo", 32)
        self.v_lo = rospy.get_param("~v_lo", 28)
        self.lab_a_max = rospy.get_param("~lab_a_max", 155)

        self.close_kernel = rospy.get_param("~close_kernel", 9)
        self.open_kernel  = rospy.get_param("~open_kernel",  3)
        self.blur_ksize   = rospy.get_param("~blur_ksize",   5)

        # ---------------- Confidence ----------------
        # Only the LOWER bound is enforced now
        self.min_grass_ratio = rospy.get_param("~min_grass_ratio", 0.004)  # 0.4%
        self.hold_bad_ms     = rospy.get_param("~hold_bad_ms", 600)

        # EMA smoothing
        self.diff_alpha   = rospy.get_param("~diff_alpha", 0.35)
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
        rospy.Subscriber("image", Image, self.on_image, queue_size=1)
        rospy.Subscriber("/gem/safety/stop", Bool, self.on_stop, queue_size=1)

        rospy.loginfo(
            "Grass-LKA fixed: v=%.2f..%.2f steer_lim=%.2f kp=%.2f kd=%.2f ki=%.3f "
            "roi=[%.2f..%.2f] gamma=%.2f HSV(H:%d..%d,S>=%d,V>=%d) LAB(a<=%d) "
            "kernels(close=%d,open=%d,blur=%d) min_grass_ratio=%.4f hold=%dms",
            self.min_speed, self.target_speed, self.steer_limit, self.kp, self.kd, self.ki,
            self.roi_top, self.roi_bot, self.gamma_y,
            self.h_lo, self.h_hi, self.s_lo, self.v_lo, self.lab_a_max,
            self.close_kernel, self.open_kernel, self.blur_ksize,
            self.min_grass_ratio, self.hold_bad_ms
        )

    # --------------- Callbacks ---------------
    def on_stop(self, msg: Bool):
        self.estop = bool(msg.data)

    def on_image(self, msg: Image):
        # Convert + ROI
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

        # Grass mask: HSV AND (S,V) + LAB a<=th, then OR
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

        mask_h = cv2.inRange(hsv[:, :, 0], self.h_lo, self.h_hi)
        mask_s = cv2.inRange(hsv[:, :, 1], self.s_lo, 255)
        mask_v = cv2.inRange(hsv[:, :, 2], self.v_lo, 255)
        mask_hsv = cv2.bitwise_and(mask_h, cv2.bitwise_and(mask_s, mask_v))

        mask_lab = cv2.inRange(lab[:, :, 1], 0, self.lab_a_max)

        grass = cv2.bitwise_or(mask_hsv, mask_lab)

        # Morphology
        if self.close_kernel > 1:
            k = np.ones((self.close_kernel, self.close_kernel), np.uint8)
            grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, k)
        if self.open_kernel > 1:
            k = np.ones((self.open_kernel, self.open_kernel), np.uint8)
            grass = cv2.morphologyEx(grass, cv2.MORPH_OPEN, k)
        if self.blur_ksize >= 3 and self.blur_ksize % 2 == 1:
            grass = cv2.medianBlur(grass, self.blur_ksize)

        # Confidence: only lower bound
        ratio = float(np.count_nonzero(grass)) / float(max(1, grass.size))
        if ratio < self.min_grass_ratio:
            # Not enough signal → hold then stop
            self._hold_or_stop("LOW RATIO")
            dbg = self._make_debug(bgr, y0, y1, grass, 0.0, 0.0, None, 0.0, 0.0, ratio)
            self._publish_debug(dbg, H, W)
            rospy.loginfo_throttle(1.0, "[grass-lka] ratio=%.3f (LOW) -> stop/hold", ratio)
            return

        # Build vertical weights
        yy = (np.arange(h).astype(np.float32) / max(1.0, h - 1))
        weights_y = np.power(yy, self.gamma_y).reshape(h, 1)

        # Optional ignore band around center
        if self.center_ignore_frac > 1e-3:
            cx = int(0.5 * w)
            half = int(0.5 * self.center_ignore_frac * w)
            grass[:, max(0, cx - half):min(w, cx + half)] = 0

        # Weighted areas left vs right
        g = (grass > 0).astype(np.float32)
        g_w = g * weights_y

        area_left  = float(np.sum(g_w[:, :w // 2]))
        area_right = float(np.sum(g_w[:, w // 2:]))
        total = area_left + area_right
        if total <= 1e-6:
            self._hold_or_stop("ZERO TOTAL")
            dbg = self._make_debug(bgr, y0, y1, grass, 0.0, 0.0, None, 0.0, 0.0, ratio)
            self._publish_debug(dbg, H, W)
            rospy.loginfo_throttle(1.0, "[grass-lka] total=0 -> stop/hold")
            return

        diff = (area_right - area_left) / total
        self.diff_ema = self.diff_alpha * diff + (1.0 - self.diff_alpha) * self.diff_ema
        err = self.diff_ema
        if abs(err) < self.deadband:
            err = 0.0

        # PD(+I)
        t = rospy.get_time()
        dt = max(1e-3, t - self.prev_t)
        d_err = (err - self.prev_err) / dt
        self.prev_err, self.prev_t = err, t

        if self.ki > 0.0:
            self.i_err = float(np.clip(self.i_err + err * dt, -0.8, 0.8))
        else:
            self.i_err = 0.0

        steer = self.kp * err + self.kd * d_err + self.ki * self.i_err
        steer = clamp(steer, -self.steer_limit, self.steer_limit)

        # Speed governor
        steer_ratio = abs(steer) / max(1e-6, self.steer_limit)
        # confidence factor grows with ratio (bounded [conf_floor..1])
        conf_factor = self.conf_floor + (1.0 - self.conf_floor) * np.clip(ratio / 0.35, 0.0, 1.0)
        speed = self.target_speed * (1.0 - self.steer_slowdown * steer_ratio) * conf_factor
        speed = clamp(speed, self.min_speed, self.target_speed)

        if self.estop:
            speed = 0.0
            steer = 0.0

        cmd = AckermannDrive()
        cmd.speed = float(speed)
        cmd.steering_angle = float(steer)
        self.pub_cmd.publish(cmd)
        self.pub_err.publish(float(err))
        self.last_cmd = cmd
        self.last_ok_t = rospy.Time.now()

        rospy.loginfo_throttle(1.0,
            "[grass-lka] ratio=%.3f L=%.0f R=%.0f err=%+.3f de=%+.3f steer=%+.2f v=%.2f",
            ratio, area_left, area_right, err, d_err, steer, speed)

        dbg = self._make_debug(bgr, y0, y1, grass, area_left, area_right, err, steer, speed, ratio)
        self._publish_debug(dbg, H, W)

    # --------------- Helpers ---------------
    def _hold_or_stop(self, _reason: str):
        if (rospy.Time.now() - self.last_ok_t).to_sec() * 1000.0 < self.hold_bad_ms:
            self.pub_cmd.publish(self.last_cmd)
        else:
            self.pub_cmd.publish(AckermannDrive())

    def _make_debug(self, bgr, y0, y1, grass_mask, area_L, area_R, err, steer, speed, ratio):
        H, W = bgr.shape[:2]
        canv = bgr.copy()
        roi_h, roi_w = grass_mask.shape[:2]
        mask_color = np.zeros((roi_h, roi_w, 3), np.uint8)
        mask_color[:, :, 1] = grass_mask  # green overlay
        canv[y0:y1, :] = cv2.addWeighted(canv[y0:y1, :], 1.0, mask_color, 0.45, 0)

        cx = W // 2
        cv2.line(canv, (cx, y0), (cx, y1), (255, 0, 0), 1)

        ytxt = 24
        def put(s):
            nonlocal ytxt
            cv2.putText(canv, s, (10, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            ytxt += 22

        put("Grass-LKA (area-balance)")
        put(f"ratio={ratio:.3f} L={area_L:.0f} R={area_R:.0f}")
        if err is not None:
            put(f"err={err:+.3f} steer={steer:+.2f} v={speed:.2f} m/s")
        return canv

    def _publish_debug(self, bgr_full, H, W):
        try:
            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(bgr_full, encoding='bgr8'))
        except Exception as e:
            rospy.logwarn_throttle(2.0, "debug publish failed: %s", e)


if __name__ == "__main__":
    VisionLKA()
    rospy.spin()

