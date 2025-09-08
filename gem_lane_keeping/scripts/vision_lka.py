#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Left-border LKA: follow slightly to the right of the LEFT solid yellow line.
- Only left yellow is used for tracking. All other markings are ignored.
- Startup bootstrap: if yellow is missing, temporarily allow white inside a left band only.
- Smooth target: EMA + innovation gating (max jump).
- Smooth steering: PD + heading, steering-rate limit, optional sign auto-calibration.
- Smooth speed: curve/steer slowdown, accel/brake limits, lateral accel limit.
"""

import rospy, cv2, numpy as np, math
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

        # --- steering controller (PD + heading) ---
        self.kp        = rospy.get_param("~kp",        0.035)
        self.kd        = rospy.get_param("~kd",        0.10)
        self.k_heading = rospy.get_param("~k_heading", 0.12)
        self.der_alpha = rospy.get_param("~der_alpha", 0.7)   # 0..1 EMA for derivative

        # --- steering shaping ---
        self.steer_limit      = rospy.get_param("~steer_limit", 0.90)    # [rad]
        self.steer_rate_limit = rospy.get_param("~steer_rate_limit", 1.4) # [rad/s]
        self.steer_alpha      = rospy.get_param("~steer_alpha", 0.22)     # EMA (0..1)
        self.steer_sign       = rospy.get_param("~steer_sign",  -1.0)     # may flip by auto-calib
        self.steer_bias       = rospy.get_param("~steer_bias",   0.0)     # manual trim [rad]

        # --- speed profile & limits ---
        self.target_speed   = rospy.get_param("~target_speed", 1.0)
        self.min_speed      = rospy.get_param("~min_speed",    0.35)
        self.recovery_speed = rospy.get_param("~recovery_speed", 0.5)
        self.k_curve_speed  = rospy.get_param("~k_curve_speed", 6.0)   # heading slowdown
        self.steer_slowdown = rospy.get_param("~steer_slowdown", 0.20) # steer slowdown

        self.wheelbase   = rospy.get_param("~wheelbase", 1.2)      # [m]
        self.a_accel_max = rospy.get_param("~a_accel_max", 0.35)   # [m/s^2]
        self.a_brake_max = rospy.get_param("~a_brake_max", 1.20)   # [m/s^2]
        self.a_lat_max   = rospy.get_param("~a_lat_max",   1.80)   # [m/s^2]

        # --- ROI & scanning ---
        self.roi_top        = rospy.get_param("~roi_top", 0.60)
        self.roi_top_boot   = rospy.get_param("~roi_top_boot", 0.55)
        self.scan_rows      = rospy.get_param("~scan_rows", [0.74, 0.84, 0.92, 0.98])
        self.min_valid_rows = rospy.get_param("~min_valid_rows", 2)
        self.hold_bad_ms    = rospy.get_param("~hold_bad_ms", 600)
        self.stop_if_lost   = rospy.get_param("~stop_if_lost", False)
        self.edge_min_run_px = rospy.get_param("~edge_min_run_px", 16)
        self.left_search_limit_frac = rospy.get_param("~left_search_limit_frac", 0.70)

        # --- target placement (between left border and camera center) ---
        self.left_offset_px        = rospy.get_param("~left_offset_px", -1.0)   # >0: absolute px offset from line
        self.left_to_center_frac   = rospy.get_param("~left_to_center_frac", 0.60)  # 0..1 toward image center
        self.center_margin_px      = rospy.get_param("~center_margin_px", 20)
        self.min_offset_margin_px  = rospy.get_param("~min_offset_margin_px", 20)
        self.max_center_jump_frac  = rospy.get_param("~max_center_jump_frac", 0.06) # per frame (of frame width)
        self.center_alpha          = rospy.get_param("~lane_center_alpha", 0.25)

        # --- camera center compensation ---
        self.cam_center_shift_px = rospy.get_param("~cam_center_shift_px", 0.0)  # +right, -left (pixels)

        # --- bootstrap if yellow missing ---
        self.bootstrap_enable         = rospy.get_param("~bootstrap_enable", True)
        self.bootstrap_frames_confirm = rospy.get_param("~bootstrap_frames_confirm", 8)
        self.bootstrap_speed_scale    = rospy.get_param("~bootstrap_speed_scale", 0.75)
        self.left_band_frac           = rospy.get_param("~left_band_frac", 0.30)   # allow white only here
        self.center_ignore_frac       = rospy.get_param("~center_ignore_frac", 0.25)
        self.yellow_good_px_threshold = rospy.get_param("~yellow_good_px_threshold", 300)

        # --- color thresholds ---
        # yellow robust (HSV + LAB b)
        self.yellow_h_lo  = rospy.get_param("~yellow_h_lo", 8)
        self.yellow_h_hi  = rospy.get_param("~yellow_h_hi", 70)
        self.yellow_s_min = rospy.get_param("~yellow_s_min", 50)
        self.yellow_v_min = rospy.get_param("~yellow_v_min", 50)
        self.lab_b_min    = rospy.get_param("~lab_b_min", 130)
        # white (bootstrap only)
        self.s_thresh  = rospy.get_param("~s_thresh", 110)
        self.v_thresh  = rospy.get_param("~v_thresh", 35)
        self.hls_L_min = rospy.get_param("~hls_L_min", 190)

        self.kernel_close = np.ones((5, 5), np.uint8)
        self.kernel_open  = np.ones((3, 3), np.uint8)

        # --- auto steering sign calibration ---
        self.auto_sign_calib = rospy.get_param("~auto_sign_calib", True)
        self.calib_window    = rospy.get_param("~calib_window", 20)
        self.calib_slope_thr = rospy.get_param("~calib_slope_thr", 0.02)
        self._hist_steer, self._hist_err = [], []
        self._sign_locked = False

        # --- state ---
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

        # --- ROS I/O ---
        self.pub_cmd = rospy.Publisher("cmd", AckermannDrive, queue_size=10)
        self.pub_err = rospy.Publisher("lateral_error", Float32, queue_size=10)
        self.pub_dbg = rospy.Publisher("debug", Image, queue_size=1)
        rospy.Subscriber("image", Image, self.on_image, queue_size=1)
        rospy.Subscriber("/gem/safety/stop", Bool, self.on_stop, queue_size=1)

        rospy.loginfo("vision_lka(left-only): kp=%.3f kd=%.3f kh=%.3f sign=%.1f bias=%.3f center_shift=%.1fpx",
                      self.kp, self.kd, self.k_heading, self.steer_sign, self.steer_bias, self.cam_center_shift_px)

    # ============================ callbacks ============================
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
        H, W = roi.shape[:2]

        # 1) Detect yellow (primary)
        mask_yel = self._yellow_mask(roi)
        yel_px = int(cv2.countNonZero(mask_yel))
        left_xs_yel, used_y_yel = self._scan_left(mask_yel, W, H)

        have_yellow = (len(left_xs_yel) >= self.min_valid_rows) and (yel_px > self.yellow_good_px_threshold)

        # 2) Bootstrap logic
        if have_yellow:
            self.yellow_ok_frames += 1
            if self.yellow_ok_frames >= int(self.bootstrap_frames_confirm):
                self.bootstrap_active = False
        else:
            self.yellow_ok_frames = 0

        if have_yellow or not self.bootstrap_active:
            left_xs, used_y = left_xs_yel, used_y_yel
            src = "Y"
            dbg_mask = mask_yel
        else:
            mask_white = self._white_mask(roi)
            left_band = self._left_white_band(mask_white, W)
            left_xs, used_y = self._scan_left(left_band, W, H)
            src = "W"
            dbg_mask = left_band

        # 3) Recovery if no left border
        if len(left_xs) < self.min_valid_rows:
            self._recovery(dbg_mask, W_full, H_full, y0, "NO LEFT LINE")
            return

        # 4) Target placement (between left border and possibly shifted camera center)
        left_med = float(np.median(left_xs))
        img_center = (0.5 * W) + float(self.cam_center_shift_px)

        if self.left_offset_px is not None and self.left_offset_px > 0.0:
            desired_x = left_med + float(self.left_offset_px)
        else:
            # pull toward center but stay away from center and from the line
            toward_center = left_med + self.left_to_center_frac * (img_center - left_med)
            desired_x = clamp(toward_center,
                              left_med + self.min_offset_margin_px,
                              img_center - self.center_margin_px)

        # innovation gating + EMA
        if self.center_px_ema is None:
            self.center_px_ema = desired_x
        else:
            max_jump = float(self.max_center_jump_frac) * W
            delta_c = clamp(desired_x - self.center_px_ema, -max_jump, max_jump)
            desired_x = self.center_px_ema + delta_c
            self.center_px_ema = (1.0 - self.center_alpha) * self.center_px_ema + self.center_alpha * desired_x

        lane_target_px = float(self.center_px_ema)

        # 5) Lateral error (normalize by half width)
        err = (lane_target_px - img_center) / max(1.0, (0.5 * W))

        # 6) Heading from left border slope (fit x(y))
        heading = self._estimate_heading_from_points(left_xs, used_y, W)

        # 7) PD + heading, rate limit, sign/bias, smoothing
        t = rospy.get_time()
        dt = max(1e-3, t - self.prev_t)

        de = (err - self.prev_err) / dt
        de_f = self.der_alpha * self.prev_de_f + (1.0 - self.der_alpha) * de

        delta_cmd = -(self.kp*err + self.kd*de_f + self.k_heading*heading) + self.steer_bias

        # steering-rate limit
        max_step = float(self.steer_rate_limit) * dt
        delta_cmd = clamp(delta_cmd, self.prev_delta - max_step, self.prev_delta + max_step)

        # sign + limit
        delta_cmd = clamp(self.steer_sign * delta_cmd, -self.steer_limit, self.steer_limit)

        # auto sign calibration (one-shot) if enabled and not locked
        if self.auto_sign_calib and not self._sign_locked:
            self._hist_steer.append(delta_cmd)
            self._hist_err.append(err)
            if len(self._hist_steer) > self.calib_window:
                self._hist_steer.pop(0); self._hist_err.pop(0)
                # if corr(steer, error) strongly positive => steering increases error -> flip sign once
                s = np.array(self._hist_steer, dtype=np.float32)
                e = np.array(self._hist_err, dtype=np.float32)
                if np.std(s) > 1e-3 and np.std(e) > 1e-3:
                    corr = float(np.corrcoef(s, e)[0,1])
                    if corr > self.calib_slope_thr:
                        self.steer_sign *= -1.0
                        rospy.logwarn("vision_lka: auto flipped steer_sign to %.1f", self.steer_sign)
                        self._sign_locked = True
                        # re-apply sign
                        delta_cmd = clamp(self.steer_sign * (delta_cmd / max(1e-6, self.steer_sign)),
                                          -self.steer_limit, self.steer_limit)

        # EMA on steering (optional)
        if self.steer_alpha > 0.0 and self.steer_alpha < 1.0:
            self.steer_filt = (1.0 - self.steer_alpha) * self.steer_filt + self.steer_alpha * delta_cmd
        else:
            self.steer_filt = delta_cmd

        self.prev_err = err
        self.prev_de_f = de_f
        self.prev_delta = delta_cmd
        self.prev_t = t

        # 8) Speed profile + physical limits (lat accel + accel/brake)
        v_scale_curve = 1.0 / (1.0 + self.k_curve_speed * abs(heading))
        v_scale_steer = 1.0 - self.steer_slowdown * abs(self.steer_filt)
        v_scale = max(0.25, min(v_scale_curve, v_scale_steer))
        tgt = self.target_speed * (self.bootstrap_speed_scale if self.bootstrap_active and src == "W" else 1.0)
        v_des = clamp(tgt * v_scale, self.min_speed, self.target_speed)

        # lateral accel limit: a_lat ~ v^2 / R ≈ v^2 * |tan(delta)| / L
        tan_d = abs(math.tan(self.steer_filt))
        if tan_d > 1e-4:
            v_lat_max = math.sqrt(max(0.0, self.a_lat_max * self.wheelbase / tan_d))
            v_des = min(v_des, v_lat_max)

        # accel/brake ramp
        speed, dt_pub = self._ramped_speed(v_des, now=t)

        # 9) Publish
        self.pub_err.publish(float(err))
        self._publish_cmd(speed, self.steer_filt, dt_pub)
        self.last_ok_time = rospy.Time.now()

        # 10) Debug overlay
        dbg = cv2.cvtColor(dbg_mask, cv2.COLOR_GRAY2BGR)
        for y, x in zip(used_y, left_xs):
            cv2.circle(dbg, (int(x), int(y)), 3, (0, 255, 255), -1)
        cv2.line(dbg, (int(img_center), H-1), (int(img_center), H-40), (255, 0, 0), 2)
        cv2.line(dbg, (int(lane_target_px), H-1), (int(lane_target_px), H-40), (0, 0, 255), 2)

        txt = f"{src} err={err:+.2f} hd={heading:+.2f} δ={self.steer_filt:+.2f} v={speed:.2f} sign={self.steer_sign:+.0f}"
        cv2.putText(dbg, txt, (10, max(14, H-12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

        canv = np.zeros((H_full, W_full, 3), dtype=np.uint8)
        canv[y0:H_full, :] = cv2.resize(dbg, (W_full, H_full - y0))
        try:
            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(canv, encoding='bgr8'))
        except Exception as e:
            rospy.logwarn_throttle(2.0, "debug publish failed: %s", e)

    # ============================ helpers ============================
    def _publish_cmd(self, speed, steer, dt=None):
        if self.estop:
            speed = 0.0; steer = 0.0
        now = rospy.get_time()
        if dt is None:
            dt = max(1e-3, now - getattr(self, "_last_pub_t", now))
        acc_prev = getattr(self, "_last_pub_acc", 0.0)
        v_prev = getattr(self, "_last_pub_speed", 0.0)

        # accel/brake clamp already handled in _ramped_speed
        cmd = AckermannDrive()
        cmd.speed = float(speed)
        cmd.steering_angle = float(clamp(steer, -self.steer_limit, self.steer_limit))
        cmd.acceleration = float((speed - v_prev) / dt)
        cmd.jerk = float((cmd.acceleration - acc_prev) / dt)

        self.pub_cmd.publish(cmd)
        self._last_pub_t = now
        self._last_pub_speed = cmd.speed
        self._last_pub_acc = cmd.acceleration
        self.last_cmd = cmd

    def _ramped_speed(self, v_des, now=None):
        if now is None:
            now = rospy.get_time()
        dt = max(1e-3, now - self.prev_t_speed)

        dv = float(v_des) - float(self.v_cmd)
        dv_max = self.a_accel_max * dt if dv >= 0.0 else self.a_brake_max * dt
        dv = clamp(dv, -abs(dv_max), abs(dv_max))

        self.v_cmd = clamp(self.v_cmd + dv, 0.0, self.target_speed)
        self.prev_t_speed = now
        return self.v_cmd, dt

    def _yellow_mask(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

        mask_hsv = cv2.inRange(hsv,
                               (self.yellow_h_lo, self.yellow_s_min, self.yellow_v_min),
                               (self.yellow_h_hi, 255, 255))
        mask_lab = cv2.inRange(lab[:,:,2], self.lab_b_min, 255)  # b channel
        mask = cv2.bitwise_and(mask_hsv, mask_lab)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
        mask = cv2.medianBlur(mask, 5)
        return mask

    def _white_mask(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
        white_hsv = cv2.inRange(hsv, (0, 0, self.v_thresh), (179, self.s_thresh, 255))
        white_hls = cv2.inRange(hls[:,:,1], self.hls_L_min, 255)
        mask = cv2.bitwise_or(white_hsv, white_hls)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
        mask = cv2.medianBlur(mask, 5)
        return mask

    def _left_white_band(self, mask, W):
        m = mask.copy()
        # cut out center completely
        c_ignore = int(W * float(self.center_ignore_frac))
        cx = W // 2
        xL_blk, xR_blk = max(0, cx - c_ignore), min(W, cx + c_ignore)
        if xR_blk > xL_blk:
            m[:, xL_blk:xR_blk] = 0
        # keep only left band near edge
        x_max = int(W * float(self.left_band_frac))
        m[:, x_max:] = 0
        return m

    def _scan_left(self, mask, W, H):
        """Find left border: for each scan-row find the *rightmost* pixel run within left search region."""
        left_xs, used_y = [], []
        x_search_max = int(W * float(self.left_search_limit_frac))
        for r in self.scan_rows:
            y = int(H * float(r)); y = min(max(y, 0), H-1)
            row = mask[y, :x_search_max]
            # find contiguous runs from right to left
            idx = None
            run = 0
            for x in range(x_search_max-1, -1, -1):
                if row[x] > 0:
                    run += 1
                    if run >= int(self.edge_min_run_px):
                        idx = x + run//2  # center of run
                        break
                else:
                    run = 0
            if idx is not None:
                left_xs.append(float(idx)); used_y.append(y)
        return left_xs, used_y

    def _estimate_heading_from_points(self, xs, ys, W):
        """Fit x = a*y + b in normalized coords and map slope to heading."""
        if len(xs) < 2: 
            return 0.0
        xs = np.array(xs, dtype=np.float32)
        ys = np.array(ys, dtype=np.float32)
        y_norm = ys / max(1.0, float(max(ys) - min(ys) + 1e-3))
        x_norm = (xs - 0.5*W) / (0.5*W)
        if y_norm.ptp() < 1e-6:
            return 0.0
        a, b = np.polyfit(y_norm, x_norm, 1)
        # small-angle mapping: slope ~ tan(heading)
        return float(math.atan(a))

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

