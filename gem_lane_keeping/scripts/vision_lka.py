#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision-based lane keeping for GEM simulator.

Inputs
------
- "image" (sensor_msgs/Image)    : camera image (remap to /gem/front_single_camera/image_raw)
- "/gem/safety/stop" (std_msgs/Bool) : safety stop from supervisor

Outputs
-------
- "cmd" (ackermann_msgs/AckermannDrive) : steering + speed command
- "lateral_error" (std_msgs/Float32)    : normalized lateral error (debug)
- "debug" (sensor_msgs/Image)           : visualization of ROI/masks/scans

Tuning (ROS params, namespace ~)
--------------------------------
target_speed        : nominal forward speed [m/s]
kp, kd, k_heading   : PD + heading gains
steer_limit         : abs steering clamp [rad]
roi_y_top / roi_top : top of ROI as fraction of image height (0..1)
scan_rows           : list of scan row positions inside ROI (0..1)
min_valid_rows      : minimal number of valid scan rows to accept
min_mask_px         : minimal count of "white" pixels in ROI mask
min_lane_width_px   : minimal distance between left/right borders
morph_kernel        : morphology kernel (odd int, e.g. 5/7/9)

white_s_max         : HSV S <= white_s_max
white_v_min         : HSV V >= white_v_min
hls_L_min           : HLS L >= hls_L_min
lab_b_min           : LAB b >= lab_b_min (yellow edge)

canny_low, canny_high : Canny optional; set 0/0 to disable
scan_margin         : ignore this many pixels at left/right of each scan
hold_bad_ms         : hold last good command this many ms before stopping
invert_steer        : if True, flips steering sign

alpha_steer         : low-pass smoothing [0..1] for steering (larger = smoother)
steer_rate_limit    : max change per frame [rad]
k_slow              : how much to slow on curves (0=no slow, 0.5=up to 50%)
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class VisionLKA:
    def __init__(self):
        rospy.init_node("vision_lka")
        self.bridge = CvBridge()

        # ---- Control / behavior params ----
        self.target_speed   = float(rospy.get_param("~target_speed", 1.5))
        self.kp             = float(rospy.get_param("~kp", 0.015))
        self.kd             = float(rospy.get_param("~kd", 0.050))
        self.k_heading      = float(rospy.get_param("~k_heading", 0.005))
        self.steer_limit    = float(rospy.get_param("~steer_limit", 0.40))

        # Smoothing & dynamics
        self.alpha_steer    = float(rospy.get_param("~alpha_steer", 0.35))  # 0..1; larger -> more smoothing
        self.steer_rate_lim = float(rospy.get_param("~steer_rate_limit", 0.06))  # max Δsteer per frame [rad]
        self.k_slow         = float(rospy.get_param("~k_slow", 0.45))  # slow down on curves (0..1)

        # ---- ROI & scan geometry ----
        self.roi_top        = float(rospy.get_param("~roi_y_top",
                               rospy.get_param("~roi_top", 0.55)))  # keep backward-compat param name
        # Scan rows are fractions of ROI height (0..1, bottom ~1.0)
        self.scan_rows      = list(rospy.get_param("~scan_rows", [0.86, 0.90, 0.94, 0.97]))
        self.min_valid_rows = int(rospy.get_param("~min_valid_rows", 1))
        self.min_mask_px    = int(rospy.get_param("~min_mask_px", 120))
        self.min_lane_w     = int(rospy.get_param("~min_lane_width_px", 12))
        self.scan_margin    = int(rospy.get_param("~scan_margin", 14))
        self.hold_bad_ms    = int(rospy.get_param("~hold_bad_ms", 600))
        self.invert_steer   = bool(rospy.get_param("~invert_steer", True))

        # ---- Color / edge thresholds ----
        self.white_s_max    = int(rospy.get_param("~white_s_max", 90))
        self.white_v_min    = int(rospy.get_param("~white_v_min", 150))
        self.hls_L_min      = int(rospy.get_param("~hls_L_min", 190))
        self.lab_b_min      = int(rospy.get_param("~lab_b_min", 140))
        self.canny_low      = int(rospy.get_param("~canny_low", 0))     # 0 disables
        self.canny_high     = int(rospy.get_param("~canny_high", 0))    # 0 disables
        self.morph_kernel   = int(rospy.get_param("~morph_kernel", 7))  # 5..9 good starting range

        # ---- State ----
        self.estop          = False
        self.prev_err       = 0.0
        self.prev_t         = rospy.get_time()
        self.prev_steer     = 0.0
        self.last_ok_time   = rospy.Time(0)
        self.last_cmd       = AckermannDrive()

        # ---- I/O ----
        self.pub_cmd = rospy.Publisher("cmd", AckermannDrive, queue_size=10)
        self.pub_err = rospy.Publisher("lateral_error", Float32, queue_size=10)
        self.pub_dbg = rospy.Publisher("debug", Image, queue_size=1)
        rospy.Subscriber("image", Image, self.on_image, queue_size=1, buff_size=2**24)
        rospy.Subscriber("/gem/safety/stop", Bool, self.on_stop, queue_size=1)

        rospy.loginfo(
            "vision_lka params: v=%.2f kp=%.3f kd=%.3f kh=%.3f steer_lim=%.2f "
            "roi_top=%.2f scans=%s min_rows=%d min_mask=%d min_w=%d "
            "HSV(S<=%d,V>=%d) HLS(L>=%d) LAB(b>=%d) canny=[%d,%d] kernel=%d "
            "alpha=%.2f dSteer<=%.3f k_slow=%.2f",
            self.target_speed, self.kp, self.kd, self.k_heading, self.steer_limit,
            self.roi_top, str(self.scan_rows), self.min_valid_rows,
            self.min_mask_px, self.min_lane_w,
            self.white_s_max, self.white_v_min, self.hls_L_min, self.lab_b_min,
            self.canny_low, self.canny_high, self.morph_kernel,
            self.alpha_steer, self.steer_rate_lim, self.k_slow
        )

    # ---------- Callbacks ----------

    def on_stop(self, msg: Bool):
        self.estop = bool(msg.data)

    def publish_cmd(self, speed: float, steer: float):
        """E-stop gating and publish AckermannDrive."""
        if self.estop:
            speed, steer = 0.0, 0.0
        if self.invert_steer:
            steer = -steer

        # steering low-pass + rate limit for stability
        raw = clamp(steer, -self.steer_limit, self.steer_limit)
        filt = (1.0 - self.alpha_steer) * self.prev_steer + self.alpha_steer * raw
        d = clamp(filt - self.prev_steer, -self.steer_rate_lim, self.steer_rate_lim)
        steer_out = self.prev_steer + d
        self.prev_steer = steer_out

        cmd = AckermannDrive()
        cmd.speed = float(max(0.0, speed))
        cmd.steering_angle = float(clamp(steer_out, -self.steer_limit, self.steer_limit))
        self.pub_cmd.publish(cmd)
        self.last_cmd = cmd

    def on_image(self, msg: Image):
        # --- Convert & ROI ---
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "cv_bridge error: %s", str(e))
            return

        h, w = bgr.shape[:2]
        y0 = int(h * clamp(self.roi_top, 0.0, 0.98))
        roi = bgr[y0:h, :]

        # --- Build lane mask (robust to lighting) ---
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # White in HSV: low saturation & high value
        white_hsv = cv2.inRange(hsv,
                                (0, 0, self.white_v_min),
                                (179, self.white_s_max, 255))
        # White in HLS: high lightness L
        L = hls[:, :, 1]
        _, white_hls = cv2.threshold(L, self.hls_L_min, 255, cv2.THRESH_BINARY)

        # Yellow edge (outer line) in LAB:b channel
        bch = lab[:, :, 2]
        _, yellow_lab = cv2.threshold(bch, self.lab_b_min, 255, cv2.THRESH_BINARY)

        # Combine white + yellow
        mask = cv2.bitwise_or(white_hsv, white_hls)
        mask = cv2.bitwise_or(mask, yellow_lab)

        # Optional Canny: only bright edges; remove ROI borders to avoid false hits
        if self.canny_high > 0:
            edges = cv2.Canny(gray, self.canny_low, self.canny_high)
            bright = cv2.inRange(gray, 140, 255)
            edges = cv2.bitwise_and(edges, bright)
            # wipe borders (10 px from each side + top)
            edges[:, :10] = 0
            edges[:, -10:] = 0
            edges[:10, :] = 0
            mask = cv2.bitwise_or(mask, edges)

        # Morphology to thicken thin lines and bridge small gaps
        k = max(3, int(self.morph_kernel) | 1)  # ensure odd
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        mask = cv2.medianBlur(mask, 5)

        nz = int(np.count_nonzero(mask))
        if nz < self.min_mask_px:
            self._maybe_hold_then_stop(mask, w, h, y0, "NO LANE (area)")
            return

        # --- Multi-row scanning along ROI height ---
        ys = mask.shape[0]
        centers, used_y = [], []

        margin = max(0, int(self.scan_margin))
        for r in self.scan_rows:
            y_scan = int(clamp(r, 0.0, 0.999) * ys)
            row = mask[y_scan, :]

            # ignore left/right margins to avoid ROI edges and FOV distortions
            row_crop = row[margin: row.shape[0] - margin]
            xs = np.flatnonzero(row_crop)

            if xs.size >= self.min_lane_w:
                left_idx  = int(xs[0]) + margin
                right_idx = int(xs[-1]) + margin
                if (right_idx - left_idx) >= self.min_lane_w:
                    centers.append(0.5 * (left_idx + right_idx))
                    used_y.append(y_scan)

        if len(centers) < self.min_valid_rows:
            self._maybe_hold_then_stop(mask, w, h, y0, "NO LANE (rows)")
            return

        # --- Lateral error (center of lane vs image center) ---
        lane_center_px = float(np.mean(centers[-max(1, len(centers)//2):]))
        img_center_px  = 0.5 * w
        err = (img_center_px - lane_center_px) / (w * 0.5)  # normalized [-1..1] (right positive)

        # --- Heading error (trend of centers vs height) ---
        heading = 0.0
        if len(centers) >= 2:
            ys_arr = np.asarray(used_y, dtype=np.float32)
            cx_arr = np.asarray(centers, dtype=np.float32)
            z = np.polyfit(ys_arr, cx_arr, 1)  # slope px/row
            slope = z[0]
            # Normalize slope to [-1..1]; minus sign tends to steer into the curve
            heading = -float(slope) / (w * 0.5)

        # --- PD control + heading compensation ---
        t = rospy.get_time()
        dt = max(1e-3, t - self.prev_t)
        d_err = (err - self.prev_err) / dt
        self.prev_err, self.prev_t = err, t

        steer_raw = self.kp * err + self.kd * d_err + self.k_heading * heading
        steer_raw = clamp(steer_raw, -self.steer_limit, self.steer_limit)

        # Speed adaptation: slow down on sharp curves (based on raw steer)
        curvature = min(1.0, abs(steer_raw) / max(1e-6, self.steer_limit))
        speed = self.target_speed * (1.0 - self.k_slow * curvature)

        # Publish + remember OK time
        self.pub_err.publish(float(err))
        self.publish_cmd(speed, steer_raw)
        self.last_ok_time = rospy.Time.now()

        # --- Debug canvas ---
        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for y_scan, cx in zip(used_y, centers):
            cv2.line(dbg, (int(cx), y_scan), (int(cx), max(0, y_scan - 28)), (0, 0, 255), 2)
        cv2.line(dbg, (int(img_center_px), ys - 6), (int(img_center_px), ys - 42), (255, 0, 0), 1)
        txt = f"nz={nz} err={err:+.2f} dE={d_err:+.2f} hd={heading:+.3f} steer={steer_raw:+.2f} v={speed:.2f}"
        cv2.putText(dbg, txt, (10, dbg.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        self._publish_debug(dbg, w, h, y0)

    # ---------- Helpers ----------

    def _maybe_hold_then_stop(self, mask, w, h, y0, label: str):
        """Hold last good command for hold_bad_ms; then stop."""
        # First, publish debug banner
        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(dbg, label, (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2, cv2.LINE_AA)
        self._publish_debug(dbg, w, h, y0)

        # If we recently had a valid solution, keep last command briefly
        if (rospy.Time.now() - self.last_ok_time).to_sec() * 1000.0 < self.hold_bad_ms:
            self.publish_cmd(self.last_cmd.speed, self.last_cmd.steering_angle)
        else:
            self.pub_err.publish(0.0)
            self.publish_cmd(0.0, 0.0)

    def _publish_debug(self, roi_bgr, w, h, y0):
        """Paste ROI back into a full-size black canvas and publish."""
        canv = np.zeros((h, w, 3), dtype=np.uint8)
        if roi_bgr.ndim == 2:
            roi_bgr = cv2.cvtColor(roi_bgr, cv2.COLOR_GRAY2BGR)
        canv[y0:h, :] = cv2.resize(roi_bgr, (w, h - y0))
        try:
            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(canv, encoding='bgr8'))
        except Exception as e:
            rospy.logwarn_throttle(2.0, "debug publish failed: %s", str(e))


if __name__ == "__main__":
    VisionLKA()
    rospy.spin()

