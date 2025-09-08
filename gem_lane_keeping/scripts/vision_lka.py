#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LKA for roads with solid YELLOW borders (preferred) and a dashed WHITE center line.
If yellow is weak/missing, fallback uses WHITE only in left/right bands, ignoring the middle.
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

        # --- PD + heading ---
        self.kp         = rospy.get_param("~kp", 0.035)
        self.kd         = rospy.get_param("~kd", 0.10)
        self.k_heading  = rospy.get_param("~k_heading", 0.12)
        self.der_alpha  = rospy.get_param("~der_alpha", 0.7)  # EMA for derivative

        # --- steering shaping ---
        self.steer_limit      = rospy.get_param("~steer_limit", 0.90)
        self.steer_rate_limit = rospy.get_param("~steer_rate_limit", 2.0)   # rad/s
        self.steer_alpha      = rospy.get_param("~steer_alpha", 0.20)       # EMA (0..1)
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

        # --- ROI & scan rows ---
        self.roi_top        = rospy.get_param("~roi_top", 0.60)
        self.scan_rows      = rospy.get_param("~scan_rows", [0.70, 0.80, 0.90, 0.96])
        self.min_valid_rows = rospy.get_param("~min_valid_rows", 2)
        self.hold_bad_ms    = rospy.get_param("~hold_bad_ms", 600)
        self.stop_if_lost   = rospy.get_param("~stop_if_lost", False)

        # --- lane geometry smoothing ---
        self.lane_w_min_px            = rospy.get_param("~lane_w_min_px", 100)
        self.lane_w_max_px            = rospy.get_param("~lane_w_max_px", 340)
        self.lane_w_ema               = rospy.get_param("~lane_w_ema", 0.20)
        self.lane_w_px_default_frac   = rospy.get_param("~lane_w_px_default_frac", 0.28)
        self.center_alpha             = rospy.get_param("~lane_center_alpha", 0.25)
        self.edge_min_run_px          = rospy.get_param("~edge_min_run_px", 16)

        # --- fallback control (white edges) ---
        self.use_white_fallback   = rospy.get_param("~use_white_fallback", True)
        self.min_yellow_px        = rospy.get_param("~min_yellow_px", 400)   # below -> fallback
        self.white_edge_band_frac = rospy.get_param("~white_edge_band_frac", 0.25)  # 25% bands on both sides
        self.center_ignore_frac   = rospy.get_param("~center_ignore_frac", 0.18)    # extra middle cut for white

        # --- color thresholds ---
        # yellow (HSV)
        self.yellow_h_lo  = rospy.get_param("~yellow_h_lo", 10)
        self.yellow_h_hi  = rospy.get_param("~yellow_h_hi", 60)
        self.yellow_s_min = rospy.get_param("~yellow_s_min", 60)
        self.yellow_v_min = rospy.get_param("~yellow_v_min", 60)
        # yellow (LAB b-channel)
        self.lab_b_min    = rospy.get_param("~lab_b_min", 140)
        # white (for debug and fallback)
        self.s_thresh     = rospy.get_param("~s_thresh", 110)
        self.v_thresh     = rospy.get_param("~v_thresh", 35)
        self.hls_L_min    = rospy.get_param("~hls_L_min", 190)

        self.kernel_close = np.ones((5,5), np.uint8)
        self.kernel_open  = np.ones((3,3), np.uint8)

        # state
        self.prev_t = rospy.get_time()
        self.prev_err = 0.0
        self.prev_de_f = 0.0
        self.prev_delta = 0.0
        self.steer_filt = 0.0
        self.center_px_ema = None
        self.last_lane_w_px = None
        self.v_cmd = 0.0
        self.prev_t_speed = rospy.get_time()
        self.last_cmd = AckermannDrive()
        self.last_ok_time = rospy.Time(0.0)
        self.estop = False

        # IO
        self.pub_cmd = rospy.Publisher("cmd", AckermannDrive, queue_size=10)
        self.pub_err = rospy.Publisher("lateral_error", Float32, queue_size=10)
        self.pub_dbg = rospy.Publisher("debug", Image, queue_size=1)
        rospy.Subscriber("image", Image, self.on_image, queue_size=1)
        rospy.Subscriber("/gem/safety/stop", Bool, self.on_stop, queue_size=1)

    # --------------- callbacks ---------------
    def on_stop(self, msg: Bool):
        self.estop = bool(msg.data)

    def on_image(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "cv_bridge: %s", e); return

        H_full, W_full = bgr.shape[:2]
        y0 = int(H_full * self.roi_top); y0 = min(max(y0, 0), H_full-2)
        roi = bgr[y0:H_full, :]
        H, W = roi.shape[:2]

        if self.last_lane_w_px is None:
            self.last_lane_w_px = int(max(self.lane_w_min_px,
                                   min(self.lane_w_max_px, W * self.lane_w_px_default_frac)))

        mask_yel, mask_wht = self._color_masks(roi)

        # build edges: prefer yellow; if weak -> white in side bands only
        edges = mask_yel.copy()
        if (cv2.countNonZero(mask_yel) < int(self.min_yellow_px)) and self.use_white_fallback:
            edges = cv2.bitwise_or(edges, self._white_edges_band(mask_wht))

        centers, used_y, lane_w = self._scan_centers(edges, W, H)

        if len(centers) < self.min_valid_rows:
            self._recovery(edges, W_full, H_full, y0, "NO LANE")
            return

        if lane_w is not None:
            a = float(self.lane_w_ema)
            self.last_lane_w_px = (1.0 - a) * self.last_lane_w_px + a * float(lane_w)

        lane_center_px = float(np.median(centers[-max(2, len(centers)):]))

        if self.center_px_ema is None:
            self.center_px_ema = lane_center_px
        else:
            a = float(self.center_alpha)
            self.center_px_ema = (1.0 - a)*self.center_px_ema + a*lane_center_px
        lane_center_px = self.center_px_ema

        img_center_px = 0.5 * W_full
        err = (lane_center_px - img_center_px) / (W_full * 0.5)

        y_arr = np.asarray(used_y, dtype=np.float32)
        c_arr = np.asarray(centers, dtype=np.float32)
        if y_arr.size >= 2 and y_arr.max() - y_arr.min() > 1e-3:
            c_norm = (c_arr - (0.5*W_full)) / (W_full*0.5)
            y_norm = y_arr / max(1.0, H)
            k = np.polyfit(y_norm, c_norm, 1)[0]
        else:
            k = 0.0
        heading = math.atan(float(k))

        # PD + heading with rate limit and EMA
        t  = rospy.get_time(); dt = max(1e-3, t - self.prev_t)
        de = (err - self.prev_err) / dt
        de_f = self.der_alpha * self.prev_de_f + (1.0 - self.der_alpha) * de

        delta = -(self.kp*err + self.kd*de_f + self.k_heading*heading)
        max_step = self.steer_rate_limit * dt
        delta = clamp(delta, self.prev_delta - max_step, self.prev_delta + max_step)
        delta = clamp(delta, -self.steer_limit, self.steer_limit)

        self.prev_err, self.prev_de_f, self.prev_delta, self.prev_t = err, de_f, delta, t

        a = float(self.steer_alpha)
        self.steer_filt = (1.0 - a)*self.steer_filt + a*(delta * self.steer_sign)

        # speed profile + hard limits
        v_curve = 1.0 / (1.0 + self.k_curve_speed * abs(heading))
        v_steer = 1.0 - self.steer_slowdown * abs(self.steer_filt)
        v_scale = max(0.25, min(v_curve, v_steer))
        v_des   = clamp(self.target_speed * v_scale, self.min_speed, self.target_speed)
        v_des   = self._limit_by_lateral(v_des, self.steer_filt)
        speed, dt_pub = self._ramped_speed(v_des)

        self.pub_err.publish(float(err))
        self.publish_cmd(speed, self.steer_filt, dt_pub)
        self.last_ok_time = rospy.Time.now()
        self.last_cmd.speed = float(speed)
        self.last_cmd.steering_angle = float(self.steer_filt)

        dbg = self._draw_debug(roi, edges, mask_yel, mask_wht, centers, used_y, lane_center_px, W_full)
        self._publish_debug(dbg, W_full, H_full, y0)

    # --------------- detectors ---------------
    def _color_masks(self, roi_bgr):
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        hls = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HLS)
        lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)

        y_hsv = cv2.inRange(hsv,
                            (int(self.yellow_h_lo), int(self.yellow_s_min), int(self.yellow_v_min)),
                            (int(self.yellow_h_hi), 255, 255))
        y_lab = cv2.inRange(lab[:, :, 2], int(self.lab_b_min), 255)
        mask_yellow = cv2.bitwise_or(y_hsv, y_lab)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, self.kernel_close)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN,  self.kernel_open)
        mask_yellow = cv2.medianBlur(mask_yellow, 5)

        white_hsv = cv2.inRange(hsv, (0, 0, int(self.v_thresh)), (179, int(self.s_thresh), 255))
        white_hls = cv2.inRange(hls[:, :, 1], int(self.hls_L_min), 255)
        mask_white = cv2.bitwise_or(white_hsv, white_hls)
        mask_white = cv2.medianBlur(mask_white, 5)
        return mask_yellow, mask_white

    def _white_edges_band(self, mask_white):
        h, w = mask_white.shape[:2]
        band = int(w * float(self.white_edge_band_frac))
        cen  = int(w * float(self.center_ignore_frac))
        out = np.zeros_like(mask_white)
        # left band
        out[:, :band] = mask_white[:, :band]
        # right band
        out[:, w-band:] = cv2.bitwise_or(out[:, w-band:], mask_white[:, w-band:])
        # remove center band entirely
        cL, cR = (w//2 - cen), (w//2 + cen)
        cL = max(0, cL); cR = min(w, cR)
        if cR > cL:
            out[:, cL:cR] = 0
        return out

    def _scan_centers(self, mask_edges, W, H):
        centers, used_y = [], []
        lane_w = None
        min_run = int(self.edge_min_run_px)

        def pick_run_from_left(row):
            run = 0; start = None; pick = None; mid = W // 2
            for x in range(0, mid):
                if row[x] > 0:
                    if run == 0: start = x
                    run += 1
                else:
                    if run >= min_run: pick = start + run // 2
                    run = 0
            if run >= min_run: pick = start + run // 2
            return pick

        def pick_run_from_right(row):
            run = 0; start = None; pick = None; mid = W // 2
            for x in range(W-1, mid-1, -1):
                if row[x] > 0:
                    if run == 0: start = x
                    run += 1
                else:
                    if run >= min_run: pick = start - run // 2
                    run = 0
            if run >= min_run: pick = start - run // 2
            return pick

        for r in self.scan_rows:
            y = int(H * float(r))
            y = 0 if y < 0 else H-1 if y >= H else y
            row = mask_edges[y, :]

            left_idx  = pick_run_from_left(row)
            right_idx = pick_run_from_right(row)

            if left_idx is not None and right_idx is not None:
                wpx = right_idx - left_idx
                if self.lane_w_min_px <= wpx <= self.lane_w_max_px:
                    centers.append(0.5 * (left_idx + right_idx))
                    used_y.append(y)
                    lane_w = wpx
            else:
                lw = self.last_lane_w_px if self.last_lane_w_px is not None else int(W * self.lane_w_px_default_frac)
                lw = max(self.lane_w_min_px, min(self.lane_w_max_px, lw))
                if left_idx is not None:
                    centers.append(left_idx + 0.5 * lw); used_y.append(y)
                elif right_idx is not None:
                    centers.append(right_idx - 0.5 * lw); used_y.append(y)

        return centers, used_y, lane_w

    # --------------- speed & debug helpers ---------------
    def _limit_by_lateral(self, v_des, delta):
        tan_d = abs(math.tan(delta))
        if tan_d > 1e-4:
            v_des = min(v_des, math.sqrt(max(0.0, self.a_lat_max * self.wheelbase / tan_d)))
        return v_des

    def _ramped_speed(self, v_des, now=None):
        if now is None: now = rospy.get_time()
        dt = max(1e-3, now - self.prev_t_speed)
        dv = float(v_des) - float(self.v_cmd)
        dv = min(dv,  self.a_accel_max * dt) if dv >= 0.0 else max(dv, -self.a_brake_max * dt)
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

    def _draw_debug(self, roi_bgr, edges, mask_yel, mask_wht, centers, used_y, lane_center_px, W_full):
        dbg = roi_bgr.copy()
        e_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        y_bgr = cv2.cvtColor(mask_yel, cv2.COLOR_GRAY2BGR)
        w_bgr = cv2.cvtColor(mask_wht, cv2.COLOR_GRAY2BGR)
        mix = cv2.addWeighted(dbg, 0.7, e_bgr, 0.6, 0.0)
        mix = cv2.addWeighted(mix, 1.0, y_bgr, 0.3, 0.0)
        mix = cv2.addWeighted(mix, 1.0, w_bgr, 0.2, 0.0)
        for y_scan, cx in zip(used_y, centers):
            cv2.circle(mix, (int(cx), int(y_scan)), 4, (0, 0, 255), -1)
        cv2.line(mix, (int(0.5*W_full), mix.shape[0]-5), (int(0.5*W_full), mix.shape[0]-55), (255,0,0), 2)
        cv2.line(mix, (int(lane_center_px), mix.shape[0]-5), (int(lane_center_px), mix.shape[0]-55), (0,255,255), 2)
        txt = f"delta={getattr(self, 'steer_filt', 0.0):+.2f} v={getattr(self.last_cmd, 'speed', 0.0):.2f}"
        cv2.putText(mix, txt, (10, mix.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1, cv2.LINE_AA)
        return mix

    def _publish_debug(self, roi_bgr, W_full, H_full, y0):
        canv = np.zeros((H_full, W_full, 3), dtype=np.uint8)
        try: canv[y0:H_full, :] = cv2.resize(roi_bgr, (W_full, H_full - y0))
        except Exception: canv[y0:H_full, :] = roi_bgr
        try: self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(canv, encoding='bgr8'))
        except Exception as e: rospy.logwarn_throttle(2.0, "debug publish failed: %s", e)


if __name__ == "__main__":
    VisionLKA()
    rospy.spin()

