#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Follow-left-line LKA:
- Detect ONLY the solid YELLOW left border.
- Drive parallel to it with a fixed rightward offset.
- Ignore the dashed white center and the right border.

Control:
- Lateral error = (desired_x - image_center_x) normalized to [-1,1].
- Heading = slope (atan) of the left line.
- PD(+heading) steering with rate limit and EMA, smooth speed with lateral limits.
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

        # --- PD + heading ---
        self.kp         = rospy.get_param("~kp", 0.035)
        self.kd         = rospy.get_param("~kd", 0.10)
        self.k_heading  = rospy.get_param("~k_heading", 0.12)
        self.der_alpha  = rospy.get_param("~der_alpha", 0.7)  # EMA for derivative

        # --- steering shaping ---
        self.steer_limit      = rospy.get_param("~steer_limit", 0.90)
        self.steer_rate_limit = rospy.get_param("~steer_rate_limit", 1.4)   # rad/s
        self.steer_alpha      = rospy.get_param("~steer_alpha", 0.22)       # EMA 0..1
        self.steer_sign       = rospy.get_param("~steer_sign", -1.0)        # invert if needed

        # --- speed profile & limits ---
        self.target_speed   = rospy.get_param("~target_speed", 1.0)
        self.min_speed      = rospy.get_param("~min_speed", 0.35)
        self.recovery_speed = rospy.get_param("~recovery_speed", 0.5)
        self.k_curve_speed  = rospy.get_param("~k_curve_speed", 6.0)
        self.steer_slowdown = rospy.get_param("~steer_slowdown", 0.20)

        self.wheelbase   = rospy.get_param("~wheelbase", 1.2)
        self.a_accel_max = rospy.get_param("~a_accel_max", 0.35)
        self.a_brake_max = rospy.get_param("~a_brake_max", 1.20)
        self.a_lat_max   = rospy.get_param("~a_lat_max", 1.80)

        # --- ROI & scans ---
        self.roi_top        = rospy.get_param("~roi_top", 0.60)
        self.scan_rows      = rospy.get_param("~scan_rows", [0.72, 0.82, 0.90, 0.96])
        self.min_valid_rows = rospy.get_param("~min_valid_rows", 2)
        self.hold_bad_ms    = rospy.get_param("~hold_bad_ms", 600)
        self.stop_if_lost   = rospy.get_param("~stop_if_lost", False)

        # --- geometry / smoothing ---
        self.edge_min_run_px      = rospy.get_param("~edge_min_run_px", 16)   # contiguous run to accept as line
        self.center_alpha         = rospy.get_param("~lane_center_alpha", 0.25)
        self.max_center_jump_frac = rospy.get_param("~max_center_jump_frac", 0.06)  # of full width per frame

        # --- desired offset from the left line ---
        # If left_offset_px > 0 -> use it; otherwise use left_offset_frac * ROI width.
        self.left_offset_px   = rospy.get_param("~left_offset_px", -1.0)
        self.left_offset_frac = rospy.get_param("~left_offset_frac", 0.38)  # ~38% of ROI width to the right

        # --- speed scaling when visibility is weak (few points) ---
        self.one_side_speed_scale = rospy.get_param("~one_side_speed_scale", 0.85)

        # --- color thresholds for robust YELLOW (HSV OR LAB-b) ---
        self.yellow_h_lo  = rospy.get_param("~yellow_h_lo", 10)
        self.yellow_h_hi  = rospy.get_param("~yellow_h_hi", 60)
        self.yellow_s_min = rospy.get_param("~yellow_s_min", 60)
        self.yellow_v_min = rospy.get_param("~yellow_v_min", 60)
        self.lab_b_min    = rospy.get_param("~lab_b_min", 140)

        self.kernel_close = np.ones((5,5), np.uint8)
        self.kernel_open  = np.ones((3,3), np.uint8)

        # --- state ---
        self.prev_t = rospy.get_time()
        self.prev_err = 0.0
        self.prev_de_f = 0.0
        self.prev_delta = 0.0
        self.steer_filt = 0.0
        self.center_px_ema = None
        self.v_cmd = 0.0
        self.prev_t_speed = rospy.get_time()
        self.last_cmd = AckermannDrive()
        self.last_ok_time = rospy.Time(0.0)
        self.estop = False

        # --- I/O ---
        self.pub_cmd = rospy.Publisher("cmd", AckermannDrive, queue_size=10)
        self.pub_err = rospy.Publisher("lateral_error", Float32, queue_size=10)
        self.pub_dbg = rospy.Publisher("debug", Image, queue_size=1)
        rospy.Subscriber("image", Image, self.on_image, queue_size=1)
        rospy.Subscriber("/gem/safety/stop", Bool, self.on_stop, queue_size=1)

    # -------------------- callbacks --------------------
    def on_stop(self, msg: Bool):
        self.estop = bool(msg.data)

    def on_image(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "cv_bridge: %s", e); return

        H_full, W_full = bgr.shape[:2]
        y0 = int(H_full * self.roi_top); y0 = min(max(y0, 0), H_full - 2)
        roi = bgr[y0:H_full, :]
        H, W = roi.shape[:2]

        mask_yellow = self._yellow_mask(roi)

        # find left border positions at scan rows
        left_xs, used_y = self._scan_left(mask_yellow, W, H)

        if len(left_xs) < self.min_valid_rows:
            self._recovery(mask_yellow, W_full, H_full, y0, "NO LEFT LINE")
            return

        # desired x: offset right from the left line
        offset_px = self.left_offset_px if self.left_offset_px > 0 else float(self.left_offset_frac) * W
        desired_x = float(np.median(left_xs)) + offset_px
        desired_x = clamp(desired_x, 10.0, W_full - 10.0)

        # innovation gating + EMA for desired center
        max_jump = float(self.max_center_jump_frac) * W_full
        if self.center_px_ema is None:
            self.center_px_ema = desired_x
        else:
            jump = clamp(desired_x - self.center_px_ema, -max_jump, +max_jump)
            self.center_px_ema = self.center_px_ema + jump
            a = float(self.center_alpha)
            self.center_px_ema = (1.0 - a) * self.center_px_ema + a * desired_x
        desired_x = self.center_px_ema

        # lateral error (normalized to [-1,1])
        img_center_px = 0.5 * W_full
        err = (desired_x - img_center_px) / (W_full * 0.5)

        # heading from left border slope
        heading = self._heading_from_points(used_y, left_xs, W_full, H)

        # PD + heading with rate limit and EMA
        t = rospy.get_time()
        dt = max(1e-3, t - self.prev_t)
        de = (err - self.prev_err) / dt
        de_f = self.der_alpha * self.prev_de_f + (1.0 - self.der_alpha) * de

        delta = -(self.kp * err + self.kd * de_f + self.k_heading * heading)
        max_step = self.steer_rate_limit * dt
        delta = clamp(delta, self.prev_delta - max_step, self.prev_delta + max_step)
        delta = clamp(delta, -self.steer_limit, self.steer_limit)

        self.prev_err = err
        self.prev_de_f = de_f
        self.prev_delta = delta
        self.prev_t = t

        new_delta = delta * self.steer_sign
        a = float(self.steer_alpha)
        self.steer_filt = (1.0 - a) * self.steer_filt + a * new_delta

        # speed profile (slightly slower if we had few valid rows)
        vis_scale = self.one_side_speed_scale if len(left_xs) <= self.min_valid_rows else 1.0
        v_curve = 1.0 / (1.0 + self.k_curve_speed * abs(heading))
        v_steer = 1.0 - self.steer_slowdown * abs(self.steer_filt)
        v_scale = max(0.25, min(v_curve, v_steer)) * vis_scale
        v_des = clamp(self.target_speed * v_scale, self.min_speed, self.target_speed)
        v_des = self._limit_by_lateral(v_des, self.steer_filt)
        speed, dt_pub = self._ramped_speed(v_des)

        # publish
        self.pub_err.publish(float(err))
        self.publish_cmd(speed, self.steer_filt, dt_pub)
        self.last_ok_time = rospy.Time.now()
        self.last_cmd.speed = float(speed)
        self.last_cmd.steering_angle = float(self.steer_filt)

        # debug image
        dbg = self._draw_debug(roi, mask_yellow, left_xs, used_y, desired_x, W_full)
        self._publish_debug(dbg, W_full, H_full, y0)

    # -------------------- detectors --------------------
    def _yellow_mask(self, roi_bgr):
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)

        y_hsv = cv2.inRange(
            hsv,
            (int(self.yellow_h_lo), int(self.yellow_s_min), int(self.yellow_v_min)),
            (int(self.yellow_h_hi), 255, 255),
        )
        y_lab = cv2.inRange(lab[:, :, 2], int(self.lab_b_min), 255)
        mask = cv2.bitwise_or(y_hsv, y_lab)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel_open)
        mask = cv2.medianBlur(mask, 5)
        return mask

    def _scan_left(self, mask_yellow, W, H):
        left_xs, used_y = [], []
        min_run = int(self.edge_min_run_px)
        mid = W // 2

        def pick_run_from_left(row):
            run = 0; start = None; pick = None
            for x in range(0, mid):
                if row[x] > 0:
                    if run == 0: start = x
                    run += 1
                else:
                    if run >= min_run: pick = start + run // 2
                    run = 0
            if run >= min_run: pick = start + run // 2
            return pick

        for r in self.scan_rows:
            y = int(H * float(r))
            y = 0 if y < 0 else H - 1 if y >= H else y
            row = mask_yellow[y, :]
            lx = pick_run_from_left(row)
            if lx is not None:
                left_xs.append(lx)
                used_y.append(y)
        return left_xs, used_y

    # -------------------- helpers --------------------
    def _heading_from_points(self, y_list, x_list, W_full, H_roi):
        if len(x_list) < 2:
            return 0.0
        y_arr = np.asarray(y_list, dtype=np.float32)
        x_arr = np.asarray(x_list, dtype=np.float32)
        x_norm = (x_arr - 0.5 * W_full) / (W_full * 0.5)
        y_norm = y_arr / max(1.0, H_roi)
        if y_norm.max() - y_norm.min() < 1e-3:
            return 0.0
        k = float(np.polyfit(y_norm, x_norm, 1)[0])
        return math.atan(k)

    def _limit_by_lateral(self, v_des, delta):
        tan_d = abs(math.tan(delta))
        if tan_d > 1e-4:
            v_lat_max = math.sqrt(max(0.0, self.a_lat_max * self.wheelbase / tan_d))
            v_des = min(v_des, v_lat_max)
        return v_des

    def _ramped_speed(self, v_des, now=None):
        if now is None: now = rospy.get_time()
        dt = max(1e-3, now - self.prev_t_speed)
        dv = float(v_des) - float(self.v_cmd)
        if dv >= 0.0: dv = min(dv, self.a_accel_max * dt)
        else:         dv = max(dv, -self.a_brake_max * dt)
        self.v_cmd = clamp(self.v_cmd + dv, 0.0, self.target_speed)
        self.prev_t_speed = now
        return self.v_cmd, dt

    def publish_cmd(self, speed, steer, dt=None):
        if self.estop: speed, steer = 0.0, 0.0
        now = rospy.get_time()
        if dt is None: dt = max(1e-3, now - getattr(self, "_last_pub_t", now))
        acc  = (speed - getattr(self, "_last_pub_speed", 0.0)) / dt
        jerk = (acc - getattr(self, "_last_pub_acc", 0.0)) / dt
        cmd = AckermannDrive()
        cmd.speed = float(speed)
        cmd.steering_angle = float(clamp(steer, -self.steer_limit, self.steer_limit))
        cmd.acceleration = float(clamp(acc, -self.a_brake_max, self.a_accel_max))
        cmd.jerk = float(jerk)
        self.pub_cmd.publish(cmd)
        self._last_pub_t = now; self._last_pub_speed = float(speed); self._last_pub_acc = float(acc)

    def _recovery(self, mask, W_full, H_full, y0, label):
        now = rospy.get_time()
        v_des = 0.0 if self.stop_if_lost else self.recovery_speed
        speed, dt_pub = self._ramped_speed(v_des, now=now)
        steer = getattr(self.last_cmd, "steering_angle", 0.0)
        self.publish_cmd(speed, steer, dt_pub)
        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(dbg, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        self._publish_debug(dbg, W_full, H_full, y0)

    def _draw_debug(self, roi_bgr, mask_yellow, left_xs, used_y, desired_x, W_full):
        dbg = roi_bgr.copy()
        y_bgr = cv2.cvtColor(mask_yellow, cv2.COLOR_GRAY2BGR)
        mix = cv2.addWeighted(dbg, 0.7, y_bgr, 0.5, 0.0)
        for y_scan, lx in zip(used_y, left_xs):
            cv2.circle(mix, (int(lx), int(y_scan)), 5, (0,0,255), -1)
        cv2.line(mix, (int(0.5*W_full), mix.shape[0]-5), (int(0.5*W_full), mix.shape[0]-55), (255,0,0), 2)   # image center
        cv2.line(mix, (int(desired_x), mix.shape[0]-5), (int(desired_x), mix.shape[0]-55), (0,255,255), 2)   # desired track
        txt = f"delta={self.steer_filt:+.2f} v={getattr(self.last_cmd,'speed',0.0):.2f}"
        cv2.putText(mix, txt, (10, mix.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        return mix

    def _publish_debug(self, roi_bgr, W_full, H_full, y0):
        canv = np.zeros((H_full, W_full, 3), dtype=np.uint8)
        try:
            canv[y0:H_full, :] = cv2.resize(roi_bgr, (W_full, H_full - y0))
        except Exception:
            canv[y0:H_full, :] = roi_bgr
        try:
            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(canv, encoding='bgr8'))
        except Exception as e:
            rospy.logwarn_throttle(2.0, "debug publish failed: %s", e)


if __name__ == "__main__":
    VisionLKA()
    rospy.spin()

