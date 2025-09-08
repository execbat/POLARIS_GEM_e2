#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision LKA (follow-left-line, simple & robust)

Behavior:
- Detect ONLY the LEFT solid yellow line (HSV ∩ LAB-b), tuned for dark asphalt.
- Hard geometry gating: keep a left-side band, cut out the center (to ignore white dashed).
- Optional bootstrap: if yellow is missing, temporarily allow WHITE only inside the left band.
- Target track is placed between the left line and the (possibly shifted) camera center.
- Smooth target (innovation clamp + EMA), PD(+heading) steering with rate limit + EMA.
- Speed shaping with curve/steer slowdown, lateral accel and dv/dt limits.

Topics:
  sub: image (sensor_msgs/Image), /gem/safety/stop (std_msgs/Bool)
  pub: cmd (ackermann_msgs/AckermannDrive), debug (sensor_msgs/Image), lateral_error (std_msgs/Float32)
"""

import rospy, cv2, numpy as np, math
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge

def clamp(v, lo, hi): return lo if v < lo else hi if v > hi else v


class VisionLKA:
    def __init__(self):
        rospy.init_node("vision_lka")
        self.bridge = CvBridge()

        # ---------- CONTROL (PD + heading) ----------
        self.kp        = rospy.get_param("~kp",        0.035)
        self.kd        = rospy.get_param("~kd",        0.10)
        self.k_heading = rospy.get_param("~k_heading", 0.12)
        self.der_alpha = rospy.get_param("~der_alpha", 0.7)   # EMA for derivative (0..1)

        # ---------- STEERING SHAPING ----------
        self.steer_limit      = rospy.get_param("~steer_limit", 0.90)   # rad
        self.steer_rate_limit = rospy.get_param("~steer_rate_limit", 1.4)  # rad/s
        self.steer_alpha      = rospy.get_param("~steer_alpha", 0.22)   # EMA 0..1
        self.steer_sign       = rospy.get_param("~steer_sign",  -1.0)   # flip if your HW needs it
        self.steer_bias       = rospy.get_param("~steer_bias",   0.0)   # rad additive

        # ---------- SPEED PROFILE & LIMITS ----------
        self.target_speed   = rospy.get_param("~target_speed", 0.8)
        self.min_speed      = rospy.get_param("~min_speed",    0.30)
        self.recovery_speed = rospy.get_param("~recovery_speed", 0.45)
        self.k_curve_speed  = rospy.get_param("~k_curve_speed", 6.0)
        self.steer_slowdown = rospy.get_param("~steer_slowdown", 0.20)

        self.wheelbase   = rospy.get_param("~wheelbase", 1.2)
        self.a_accel_max = rospy.get_param("~a_accel_max", 0.35)
        self.a_brake_max = rospy.get_param("~a_brake_max", 1.20)
        self.a_lat_max   = rospy.get_param("~a_lat_max",   1.80)

        # ---------- ROI & SCANS ----------
        self.roi_top        = rospy.get_param("~roi_top", 0.58)     # tuned for your scene
        self.roi_top_boot   = rospy.get_param("~roi_top_boot", 0.58)
        self.scan_rows      = rospy.get_param("~scan_rows", [0.75, 0.85, 0.93, 0.98])
        self.min_valid_rows = rospy.get_param("~min_valid_rows", 2)
        self.edge_min_run_px = rospy.get_param("~edge_min_run_px", 14)
        self.left_search_limit_frac = rospy.get_param("~left_search_limit_frac", 0.70)
        self.hold_bad_ms    = rospy.get_param("~hold_bad_ms", 600)
        self.stop_if_lost   = rospy.get_param("~stop_if_lost", False)

        # ---------- TARGET PLACEMENT (between left line and camera center) ----------
        self.left_offset_px       = rospy.get_param("~left_offset_px", -1.0)       # >0 = absolute px from line
        self.left_to_center_frac  = rospy.get_param("~left_to_center_frac", 0.60)  # 0..1 toward center
        self.center_margin_px     = rospy.get_param("~center_margin_px", 20)
        self.min_offset_margin_px = rospy.get_param("~min_offset_margin_px", 20)
        self.max_center_jump_frac = rospy.get_param("~max_center_jump_frac", 0.06)
        self.center_alpha         = rospy.get_param("~lane_center_alpha", 0.25)
        self.cam_center_shift_px  = rospy.get_param("~cam_center_shift_px", 0.0)   # +right / -left

        # ---------- COLOR THRESHOLDS (yellow robust) ----------
        # Tuned on your screenshot: dark asphalt, dim yellow
        self.yellow_h_lo  = rospy.get_param("~yellow_h_lo", 18)
        self.yellow_h_hi  = rospy.get_param("~yellow_h_hi", 42)
        self.yellow_s_min = rospy.get_param("~yellow_s_min", 80)
        self.yellow_v_min = rospy.get_param("~yellow_v_min", 28)
        self.lab_b_min    = rospy.get_param("~lab_b_min",   145)

        # Geometry gating for masking (left-only, center cut)
        self.left_band_frac     = rospy.get_param("~left_band_frac", 0.62)
        self.center_ignore_frac = rospy.get_param("~center_ignore_frac", 0.18)

        # Morphology kernels
        self.close_kernel = rospy.get_param("~close_kernel", 7)
        self.open_kernel  = rospy.get_param("~open_kernel",  5)

        # ---------- BOOTSTRAP (fallback to white-in-left-band) ----------
        self.bootstrap_enable         = rospy.get_param("~bootstrap_enable", True)
        self.bootstrap_frames_confirm = rospy.get_param("~bootstrap_frames_confirm", 6)
        self.bootstrap_speed_scale    = rospy.get_param("~bootstrap_speed_scale", 0.75)
        self.yellow_good_px_threshold = rospy.get_param("~yellow_good_px_threshold", 250)
        # white thresholds (used only in bootstrap)
        self.s_thresh  = rospy.get_param("~s_thresh", 110)  # HSV S max for white
        self.v_thresh  = rospy.get_param("~v_thresh", 35)   # HSV V min
        self.hls_L_min = rospy.get_param("~hls_L_min", 190) # HLS L min

        # ---------- STATE ----------
        self.prev_t = rospy.get_time()
        self.prev_err = 0.0
        self.prev_de_f = 0.0
        self.prev_delta = 0.0
        self.steer_filt = 0.0
        self.center_px_ema = None
        self.v_cmd = 0.0
        self.prev_t_speed = rospy.get_time()
        self.last_ok_time = rospy.Time(0.0)
        self.last_cmd = AckermannDrive()
        self.estop = False
        self.bootstrap_active = bool(self.bootstrap_enable)
        self.yellow_ok_frames = 0

        # ---------- I/O ----------
        self.pub_cmd = rospy.Publisher("cmd", AckermannDrive, queue_size=10)
        self.pub_err = rospy.Publisher("lateral_error", Float32, queue_size=10)
        self.pub_dbg = rospy.Publisher("debug", Image, queue_size=1)
        rospy.Subscriber("image", Image, self.on_image, queue_size=1)
        rospy.Subscriber("/gem/safety/stop", Bool, self.on_stop, queue_size=1)

        rospy.loginfo("vision_lka: kp=%.3f kd=%.3f kh=%.3f sign=%.1f bias=%.2f center_shift=%.1fpx",
                      self.kp, self.kd, self.k_heading, self.steer_sign, self.steer_bias, self.cam_center_shift_px)

    # ========================== callbacks ==========================
    def on_stop(self, msg: Bool):
        self.estop = bool(msg.data)

    def on_image(self, msg: Image):
        # 0) BGR + ROI
        try:
            bgr_full = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "cv_bridge: %s", e)
            return

        H_full, W_full = bgr_full.shape[:2]
        recovering = (rospy.Time.now() - self.last_ok_time).to_sec()*1000.0 >= self.hold_bad_ms
        roi_top_eff = self.roi_top_boot if (self.bootstrap_active or recovering) else self.roi_top
        y0 = int(H_full * roi_top_eff); y0 = min(max(y0, 0), H_full - 2)
        roi = bgr_full[y0:H_full, :]
        H, W = roi.shape[:2]  # width unchanged

        # 1) Yellow detection (primary)
        mask_yel = self._yellow_mask(roi)
        yel_px = int(cv2.countNonZero(mask_yel))
        left_xs_yel, used_y_yel = self._scan_left(mask_yel, W, H)

        have_yellow = (len(left_xs_yel) >= self.min_valid_rows) and (yel_px > self.yellow_good_px_threshold)

        # 2) Bootstrap if yellow missing
        if have_yellow:
            self.yellow_ok_frames += 1
            if self.yellow_ok_frames >= int(self.bootstrap_frames_confirm):
                self.bootstrap_active = False
        else:
            self.yellow_ok_frames = 0

        if have_yellow or not self.bootstrap_active:
            left_xs, used_y = left_xs_yel, used_y_yel
            dbg_mask = mask_yel
            dbg_tag = "YEL"
        else:
            mask_white = self._white_mask(roi)
            left_band = self._left_white_band(mask_white, W)
            left_xs, used_y = self._scan_left(left_band, W, H)
            dbg_mask = left_band
            dbg_tag = "BOOT"

        # 3) Recovery if no left border
        if len(left_xs) < self.min_valid_rows:
            self._recovery(dbg_mask, W_full, H_full, y0, "NO LEFT LINE")
            return

        # 4) Target placement between left line and (shifted) camera center
        left_med = float(np.median(left_xs))
        img_center = (0.5 * W) + float(self.cam_center_shift_px)

        if self.left_offset_px is not None and self.left_offset_px > 0.0:
            desired_x = left_med + float(self.left_offset_px)
        else:
            toward_center = left_med + float(self.left_to_center_frac) * (img_center - left_med)
            desired_x = clamp(toward_center,
                              left_med + float(self.min_offset_margin_px),
                              img_center - float(self.center_margin_px))

        # Innovation clamp + EMA on target
        if self.center_px_ema is None:
            self.center_px_ema = desired_x
        else:
            max_jump = float(self.max_center_jump_frac) * W
            jump = clamp(desired_x - self.center_px_ema, -max_jump, max_jump)
            self.center_px_ema = (1.0 - self.center_alpha) * (self.center_px_ema + jump) + self.center_alpha * desired_x
        lane_target_px = float(self.center_px_ema)

        # 5) Errors
        err = (lane_target_px - img_center) / max(1.0, (0.5 * W))
        heading = self._estimate_heading_from_points(left_xs, used_y, W)

        # 6) PD + heading, rate limit, EMA, steering sign/bias
        t  = rospy.get_time()
        dt = max(1e-3, t - self.prev_t)
        de = (err - self.prev_err) / dt
        de_f = self.der_alpha * self.prev_de_f + (1.0 - self.der_alpha) * de

        delta_cmd = -(self.kp * err + self.kd * de_f + self.k_heading * heading) + float(self.steer_bias)
        max_step = float(self.steer_rate_limit) * dt
        delta_cmd = clamp(delta_cmd, self.prev_delta - max_step, self.prev_delta + max_step)
        delta_cmd = clamp(self.steer_sign * delta_cmd, -self.steer_limit, self.steer_limit)

        # EMA on steering
        a = float(self.steer_alpha)
        self.steer_filt = (1.0 - a) * self.steer_filt + a * delta_cmd

        self.prev_err = err
        self.prev_de_f = de_f
        self.prev_delta = delta_cmd
        self.prev_t = t

        # 7) Speed shaping + physical limits
        v_curve = 1.0 / (1.0 + self.k_curve_speed * abs(heading))
        v_steer = 1.0 - self.steer_slowdown * abs(self.steer_filt)
        v_scale = max(0.25, min(v_curve, v_steer))

        tgt = self.target_speed * (self.bootstrap_speed_scale if (dbg_tag == "BOOT") else 1.0)
        v_des = clamp(tgt * v_scale, self.min_speed, self.target_speed)

        # lateral accel limit: a_lat ~ v^2 * |tan(delta)| / L
        tan_d = abs(math.tan(self.steer_filt))
        if tan_d > 1e-4:
            v_lat_max = math.sqrt(max(0.0, self.a_lat_max * self.wheelbase / tan_d))
            v_des = min(v_des, v_lat_max)

        speed, dt_pub = self._ramped_speed(v_des, now=t)

        # 8) Publish
        self.pub_err.publish(float(err))
        self._publish_cmd(speed, self.steer_filt, dt_pub)
        self.last_ok_time = rospy.Time.now()

        # 9) Debug overlay
        dbg = cv2.cvtColor(dbg_mask, cv2.COLOR_GRAY2BGR)
        for y, x in zip(used_y, left_xs):
            cv2.circle(dbg, (int(x), int(y)), 3, (0, 255, 255), -1)
        cv2.line(dbg, (int(img_center), H-1), (int(img_center), H-40), (255, 0, 0), 2)
        cv2.line(dbg, (int(lane_target_px), H-1), (int(lane_target_px), H-40), (0, 0, 255), 2)
        txt = f"{dbg_tag} err={err:+.2f} hd={heading:+.2f} d={self.steer_filt:+.2f} v={speed:.2f}"
        cv2.putText(dbg, txt, (10, max(16, H-12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        canv = np.zeros((H_full, W_full, 3), dtype=np.uint8)
        canv[y0:H_full, :] = cv2.resize(dbg, (W_full, H_full - y0))
        try:
            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(canv, encoding='bgr8'))
        except Exception as e:
            rospy.logwarn_throttle(2.0, "debug publish failed: %s", e)

    # ========================== detectors ==========================
    def _yellow_mask(self, roi_bgr):
        H, W = roi_bgr.shape[:2]
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)

        # HSV ∩ LAB-b
        m_hsv = cv2.inRange(hsv,
                            (int(self.yellow_h_lo), int(self.yellow_s_min), int(self.yellow_v_min)),
                            (int(self.yellow_h_hi), 255, 255))
        m_lab = cv2.inRange(lab[:, :, 2], int(self.lab_b_min), 255)
        mask = cv2.bitwise_and(m_hsv, m_lab)

        # geometry gates
        left_band = int(float(self.left_band_frac) * W)
        center_cut = int(float(self.center_ignore_frac) * W)
        mask[:, left_band:] = 0
        cL, cR = (W//2 - center_cut), (W//2 + center_cut)
        cL = max(0, cL); cR = min(W, cR)
        if cR > cL: mask[:, cL:cR] = 0

        # morphology
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((int(self.close_kernel), int(self.close_kernel)), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((int(self.open_kernel),  int(self.open_kernel)),  np.uint8))
        return mask

    def _white_mask(self, roi_bgr):
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        hls = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HLS)
        white_hsv = cv2.inRange(hsv, (0, 0, int(self.v_thresh)), (179, int(self.s_thresh), 255))
        white_hls = cv2.inRange(hls[:, :, 1], int(self.hls_L_min), 255)
        mask = cv2.bitwise_or(white_hsv, white_hls)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        mask = cv2.medianBlur(mask, 5)
        return mask

    def _left_white_band(self, mask, W):
        out = mask.copy()
        # cut center completely
        cen = int(float(self.center_ignore_frac) * W)
        out[:, W//2 - cen : W//2 + cen] = 0
        # keep only left band
        band = int(float(self.left_band_frac) * W)
        out[:, band:] = 0
        return out

    def _scan_left(self, mask, W, H):
        """Find left border by scanning several rows; pick the rightmost run within left region."""
        left_xs, used_y = [], []
        x_search_max = int(float(self.left_search_limit_frac) * W)
        min_run = int(self.edge_min_run_px)

        for r in self.scan_rows:
            y = int(H * float(r)); y = min(max(y, 0), H-1)
            row = mask[y, :x_search_max]
            run = 0; idx = None
            for x in range(x_search_max-1, -1, -1):
                if row[x] > 0:
                    run += 1
                    if run >= min_run:
                        idx = x + run//2
                        break
                else:
                    run = 0
            if idx is not None:
                left_xs.append(float(idx)); used_y.append(y)

        return left_xs, used_y

    # ========================== helpers ==========================
    def _estimate_heading_from_points(self, xs, ys, W):
        """Fit x(y) in normalized coords; map slope to heading angle."""
        if len(xs) < 2: return 0.0
        xs = np.asarray(xs, dtype=np.float32)
        ys = np.asarray(ys, dtype=np.float32)
        y_norm = ys / max(1.0, float(ys.max() - ys.min() + 1e-3))
        x_norm = (xs - 0.5*W) / (0.5*W)
        if y_norm.ptp() < 1e-6: return 0.0
        k = float(np.polyfit(y_norm, x_norm, 1)[0])
        return float(math.atan(k))

    def _ramped_speed(self, v_des, now=None):
        if now is None:
            now = rospy.get_time()
        dt = max(1e-3, now - self.prev_t_speed)
        dv = float(v_des) - float(self.v_cmd)
        dv_max = (self.a_accel_max * dt) if dv >= 0.0 else (self.a_brake_max * dt)
        dv = clamp(dv, -abs(dv_max), abs(dv_max))
        self.v_cmd = clamp(self.v_cmd + dv, 0.0, self.target_speed)
        self.prev_t_speed = now
        return self.v_cmd, dt

    def _publish_cmd(self, speed, steer, dt=None):
        if self.estop:
            speed, steer = 0.0, 0.0
        now = rospy.get_time()
        if dt is None:
            dt = max(1e-3, now - getattr(self, "_last_pub_t", now))
        v_prev = float(getattr(self, "_last_pub_speed", 0.0))
        acc_prev = float(getattr(self, "_last_pub_acc", 0.0))
        acc = (speed - v_prev) / dt
        jerk = (acc - acc_prev) / dt

        cmd = AckermannDrive()
        cmd.speed = float(speed)
        cmd.steering_angle = float(clamp(steer, -self.steer_limit, self.steer_limit))
        cmd.acceleration = float(clamp(acc, -self.a_brake_max, self.a_accel_max))
        cmd.jerk = float(jerk)
        self.pub_cmd.publish(cmd)

        self._last_pub_t = now
        self._last_pub_speed = cmd.speed
        self._last_pub_acc = cmd.acceleration
        self.last_cmd = cmd

    def _recovery(self, mask, W_full, H_full, y0, label):
        v_des = 0.0 if self.stop_if_lost else self.recovery_speed
        speed, dt_pub = self._ramped_speed(v_des, now=rospy.get_time())
        steer = getattr(self.last_cmd, "steering_angle", 0.0)
        self._publish_cmd(speed, steer, dt_pub)

        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(dbg, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        canv = np.zeros((H_full, W_full, 3), dtype=np.uint8)
        canv[y0:H_full, :] = cv2.resize(dbg, (W_full, H_full - y0))
        try:
            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(canv, encoding='bgr8'))
        except Exception as e:
            rospy.logwarn_throttle(2.0, "debug publish failed: %s", e)


if __name__ == "__main__":
    VisionLKA()
    rospy.spin()

