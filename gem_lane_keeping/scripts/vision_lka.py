#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision LKA (robust lane keeping for GEM)
- Detects left and right solid YELLOW borders (HSV ∩ LAB-b), robust on dark asphalt.
- Ignores center white dashed by geometry gates.
- Lane center = between-left-right; if only left is visible -> extrapolate using last lane width.
- Target position = fixed fraction between side borders (lane_pos_frac); smoothed + innovation clamp.
- Control = PD on lateral error + heading, with steering rate limit and EMA.
- Speed = curve/steer slowdown + lateral-accel and dv/dt limits.
- Lost-mode = keep last target, slow down, keep publishing.

Publishes: ackermann_msgs/AckermannDrive on topic "cmd" (remap to /gem/ackermann_cmd).
Debug overlay on "debug".
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

        # --- control (PD + heading)
        self.kp        = rospy.get_param("~kp",        0.070)
        self.kd        = rospy.get_param("~kd",        0.14)
        self.k_heading = rospy.get_param("~k_heading", 0.22)
        self.der_alpha = rospy.get_param("~der_alpha", 0.5)   # EMA for derivative (0..1)

        # --- steering shaping
        self.steer_limit      = rospy.get_param("~steer_limit", 0.90)  # rad
        self.steer_rate_limit = rospy.get_param("~steer_rate_limit", 4.0) # rad/s
        self.steer_alpha      = rospy.get_param("~steer_alpha", 0.10)  # EMA of output
        self.steer_sign       = rospy.get_param("~steer_sign",  -1.0)  # flip if your HW is inverted
        self.steer_bias       = rospy.get_param("~steer_bias",   0.0)  # rad

        # --- speed profile & limits
        self.target_speed   = rospy.get_param("~target_speed", 0.55)
        self.min_speed      = rospy.get_param("~min_speed",    0.30)
        self.recovery_speed = rospy.get_param("~recovery_speed", 0.40)
        self.k_curve_speed  = rospy.get_param("~k_curve_speed", 8.0)
        self.steer_slowdown = rospy.get_param("~steer_slowdown", 0.22)

        self.wheelbase   = rospy.get_param("~wheelbase", 1.2)
        self.a_accel_max = rospy.get_param("~a_accel_max", 0.35)
        self.a_brake_max = rospy.get_param("~a_brake_max", 1.20)
        self.a_lat_max   = rospy.get_param("~a_lat_max",   1.80)

        # --- ROI & scans
        self.roi_top      = rospy.get_param("~roi_top", 0.58)
        self.scan_rows    = rospy.get_param("~scan_rows", [0.75, 0.85, 0.93, 0.98])
        self.min_valid_rows = rospy.get_param("~min_valid_rows", 2)
        self.edge_min_run_px = rospy.get_param("~edge_min_run_px", 12)
        self.hold_bad_ms  = rospy.get_param("~hold_bad_ms", 400)
        self.stop_if_lost = rospy.get_param("~stop_if_lost", False)

        # --- desired lane position
        # fraction from left to right (0=center is 0.5, closer to left is <0.5, closer to right is >0.5)
        self.lane_pos_frac       = rospy.get_param("~lane_pos_frac", 0.60)  # default: a bit right of left line
        self.center_alpha        = rospy.get_param("~center_alpha", 0.25)   # EMA of target x
        self.max_center_jump_frac= rospy.get_param("~max_center_jump_frac", 0.08)
        self.cam_center_shift_px = rospy.get_param("~cam_center_shift_px", 0.0)  # +right/-left pixel shift

        # --- color thresholds (robust yellow)
        self.yellow_h_lo  = rospy.get_param("~yellow_h_lo", 18)
        self.yellow_h_hi  = rospy.get_param("~yellow_h_hi", 42)
        self.yellow_s_min = rospy.get_param("~yellow_s_min", 80)
        self.yellow_v_min = rospy.get_param("~yellow_v_min", 28)
        self.lab_b_min    = rospy.get_param("~lab_b_min",   145)

        # geometry gates for masks
        self.left_band_frac      = rospy.get_param("~left_band_frac", 0.62)  # keep x <= frac*W
        self.right_band_frac     = rospy.get_param("~right_band_frac", 0.62) # keep x >= (1-frac)*W for right
        self.center_ignore_frac  = rospy.get_param("~center_ignore_frac", 0.18)

        # morphology
        self.close_kernel = rospy.get_param("~close_kernel", 7)
        self.open_kernel  = rospy.get_param("~open_kernel",  5)

        # --- state
        self.prev_t = rospy.get_time()
        self.prev_err = 0.0
        self.prev_de_f = 0.0
        self.prev_delta = 0.0
        self.steer_filt = 0.0

        self.v_cmd = 0.0
        self.prev_t_speed = rospy.get_time()

        self.last_ok_time = rospy.Time(0.0)
        self.last_lane_w_px = None
        self.center_px_ema = None
        self.last_cmd = AckermannDrive()
        self.estop = False

        # --- IO
        self.pub_cmd = rospy.Publisher("cmd", AckermannDrive, queue_size=10)
        self.pub_err = rospy.Publisher("lateral_error", Float32, queue_size=10)
        self.pub_dbg = rospy.Publisher("debug", Image, queue_size=1)
        rospy.Subscriber("image", Image, self.on_image, queue_size=1)
        rospy.Subscriber("/gem/safety/stop", Bool, self.on_stop, queue_size=1)

        rospy.loginfo("vision_lka ready: kp=%.3f kd=%.3f kh=%.3f steer_sign=%.1f", self.kp, self.kd, self.k_heading, self.steer_sign)

    # ======================= callbacks =======================
    def on_stop(self, msg: Bool):
        self.estop = bool(msg.data)

    def on_image(self, msg: Image):
        try:
            bgr_full = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "cv_bridge: %s", e)
            return

        Hf, Wf = bgr_full.shape[:2]
        y0 = int(clamp(self.roi_top, 0.0, 0.95) * Hf)
        roi = bgr_full[y0:Hf, :]
        H, W = roi.shape[:2]
        img_center = 0.5*W + float(self.cam_center_shift_px)

        # 1) color masks
        mask_yel = self._yellow_mask(roi)
        mask_left  = self._gate_left(mask_yel, W)
        mask_right = self._gate_right(mask_yel, W)

        # 2) scan left/right borders
        Lxs, Lys = self._scan_border(mask_left, side="left", W=W, H=H)
        Rxs, Rys = self._scan_border(mask_right, side="right", W=W, H=H)

        have_L = len(Lxs) >= self.min_valid_rows
        have_R = len(Rxs) >= self.min_valid_rows

        # 3) lane width and target x
        lane_w = None
        if have_L and have_R:
            # pair rows by nearest y to compute width robustly
            y_pairs = []
            for y in Lys:
                diffs = np.abs(np.array(Rys) - y)
                j = int(np.argmin(diffs))
                y_pairs.append((y, Rys[j], Lxs[Lys.index(y)], Rxs[j]))
            widths = [abs(rx - lx) for (_, _, lx, rx) in y_pairs if rx > lx]
            if widths:
                lane_w = float(np.median(widths))
        # smooth lane width
        if lane_w is not None and lane_w > 10:
            self.last_lane_w_px = float(0.8*(self.last_lane_w_px if self.last_lane_w_px else lane_w) + 0.2*lane_w)

        # desired target x:
        if have_L and have_R:
            left_med  = float(np.median(Lxs))
            right_med = float(np.median(Rxs))
            center_geom = 0.5*(left_med + right_med)
            desired_x = left_med + float(self.lane_pos_frac) * (right_med - left_med)
        elif have_L and (self.last_lane_w_px is not None):
            left_med = float(np.median(Lxs))
            desired_x = left_med + float(self.lane_pos_frac) * float(self.last_lane_w_px)
            center_geom = left_med + 0.5*float(self.last_lane_w_px)
        else:
            # no borders visible enough: recovery
            self._recovery(mask_yel, Wf, Hf, y0, "NO LANE")
            return

        # clamp target innovation + EMA
        if self.center_px_ema is None:
            self.center_px_ema = desired_x
        else:
            max_jump = float(self.max_center_jump_frac) * W
            innovation = clamp(desired_x - self.center_px_ema, -max_jump, max_jump)
            self.center_px_ema = (1.0 - self.center_alpha) * (self.center_px_ema + innovation) + self.center_alpha * desired_x
        target_x = float(self.center_px_ema)

        # 4) lateral error (+ heading from local fit of centerline)
        err = (target_x - img_center) / max(1.0, 0.5*W)

        # estimate heading: fit x(y) of lane center using whichever points we have
        if have_L and have_R:
            Ys = np.array(Lys + Rys, dtype=np.float32)
            Xs = np.array([left_med]*len(Lys) + [right_med]*len(Rys), dtype=np.float32)
            # centerline x = (left+right)/2; approximate slope using border trend
            # safer: compute slope from whichever border is longer
            heading = self._heading_from_points(Lxs if len(Lxs)>len(Rxs) else Rxs,
                                                Lys if len(Lys)>len(Rys) else Rys, W)
        elif have_L:
            heading = self._heading_from_points(Lxs, Lys, W)
        else:
            heading = 0.0

        # 5) PD + heading, rate limit, EMA and sign/bias
        t  = rospy.get_time()
        dt = max(1e-3, t - self.prev_t)
        de = (err - self.prev_err) / dt
        de_f = self.der_alpha*self.prev_de_f + (1.0 - self.der_alpha)*de

        delta_cmd = -(self.kp*err + self.kd*de_f + self.k_heading*heading) + float(self.steer_bias)
        max_step = float(self.steer_rate_limit) * dt
        delta_cmd = clamp(delta_cmd, self.prev_delta - max_step, self.prev_delta + max_step)
        delta_cmd = clamp(self.steer_sign * delta_cmd, -self.steer_limit, self.steer_limit)

        a = float(self.steer_alpha)
        self.steer_filt = (1.0 - a)*self.steer_filt + a*delta_cmd

        self.prev_err, self.prev_de_f, self.prev_delta, self.prev_t = err, de_f, delta_cmd, t

        # 6) speed shaping and physical limits
        v_curve = 1.0 / (1.0 + self.k_curve_speed * abs(heading))
        v_steer = 1.0 - self.steer_slowdown * abs(self.steer_filt)
        v_scale = max(0.25, min(v_curve, v_steer))
        v_des   = clamp(self.target_speed * v_scale, self.min_speed, self.target_speed)

        tan_d = abs(math.tan(self.steer_filt))
        if tan_d > 1e-4:
            v_lat_max = math.sqrt(max(0.0, self.a_lat_max * self.wheelbase / tan_d))
            v_des = min(v_des, v_lat_max)

        speed, dt_pub = self._ramped_speed(v_des, now=t)

        # 7) publish
        self.pub_err.publish(float(err))
        self._publish_cmd(speed, self.steer_filt, dt_pub)
        self.last_ok_time = rospy.Time.now()

        # 8) debug
        dbg = cv2.cvtColor(mask_yel, cv2.COLOR_GRAY2BGR)
        for y,x in zip(Lys, Lxs): cv2.circle(dbg, (int(x), int(y)), 3, (0,255,255), -1)
        for y,x in zip(Rys, Rxs): cv2.circle(dbg, (int(x), int(y)), 3, (255,255,0), -1)
        cv2.line(dbg, (int(img_center), H-1), (int(img_center), H-40), (255,0,0), 2)
        cv2.line(dbg, (int(target_x),  H-1), (int(target_x),  H-40), (0,0,255), 2)
        txt = f"e={err:+.2f} hd={heading:+.2f} d={self.steer_filt:+.2f} v={speed:.2f}"
        cv2.putText(dbg, txt, (10, max(16, H-12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1, cv2.LINE_AA)
        self._publish_debug(dbg, bgr_full, y0)

    # ======================= detectors =======================
    def _yellow_mask(self, roi_bgr):
        H, W = roi_bgr.shape[:2]
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)

        m_hsv = cv2.inRange(hsv,
                            (int(self.yellow_h_lo), int(self.yellow_s_min), int(self.yellow_v_min)),
                            (int(self.yellow_h_hi), 255, 255))
        m_lab = cv2.inRange(lab[:,:,2], int(self.lab_b_min), 255)
        mask = cv2.bitwise_and(m_hsv, m_lab)

        # morphology
        if self.close_kernel > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((int(self.close_kernel), int(self.close_kernel)), np.uint8))
        if self.open_kernel > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((int(self.open_kernel),  int(self.open_kernel)),  np.uint8))
        return mask

    def _gate_left(self, mask, W):
        out = mask.copy()
        cen = int(float(self.center_ignore_frac) * W)
        out[:, W//2 - cen : W//2 + cen] = 0
        band = int(float(self.left_band_frac) * W)
        out[:, band:] = 0
        return out

    def _gate_right(self, mask, W):
        out = mask.copy()
        cen = int(float(self.center_ignore_frac) * W)
        out[:, W//2 - cen : W//2 + cen] = 0
        band = int((1.0 - float(self.right_band_frac)) * W)
        out[:, :band] = 0
        return out

    def _scan_border(self, mask, side, W, H):
        xs, ys = [], []
        min_run = int(self.edge_min_run_px)
        if side == "left":
            xlim = int(float(self.left_band_frac) * W)
            for r in self.scan_rows:
                y = int(H * float(r)); y = min(max(y, 0), H-1)
                row = mask[y, :xlim]
                run = 0; idx = None
                for x in range(xlim-1, -1, -1):
                    if row[x] > 0:
                        run += 1
                        if run >= min_run: idx = x + run//2; break
                    else:
                        run = 0
                if idx is not None:
                    xs.append(float(idx)); ys.append(y)
        else:
            xlim = int((1.0 - float(self.right_band_frac)) * W)
            for r in self.scan_rows:
                y = int(H * float(r)); y = min(max(y, 0), H-1)
                row = mask[y, xlim:]
                run = 0; idx = None
                for i, pix in enumerate(row):
                    if pix > 0:
                        run += 1
                        if run >= min_run:
                            idx = xlim + i - run//2
                            break
                    else:
                        run = 0
                if idx is not None:
                    xs.append(float(idx)); ys.append(y)
        return xs, ys

    # ======================= helpers =======================
    def _heading_from_points(self, xs, ys, W):
        if len(xs) < 2: return 0.0
        xs = np.asarray(xs, dtype=np.float32)
        ys = np.asarray(ys, dtype=np.float32)
        y_norm = (ys - ys.min()) / max(1.0, (ys.max() - ys.min()))
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

    def _publish_debug(self, roi_bgr_mask_or, bgr_full, y0):
        Hf, Wf = bgr_full.shape[:2]
        canv = np.zeros((Hf, Wf, 3), dtype=np.uint8)
        try:
            roi_resized = cv2.resize(roi_bgr_mask_or, (Wf, Hf - y0))
            canv[y0:Hf, :] = roi_resized
            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(canv, encoding='bgr8'))
        except Exception as e:
            rospy.logwarn_throttle(2.0, "debug publish failed: %s", e)

    def _recovery(self, mask, Wf, Hf, y0, label):
        v_des = 0.0 if self.stop_if_lost else self.recovery_speed
        speed, dt_pub = self._ramped_speed(v_des, now=rospy.get_time())
        steer = getattr(self.last_cmd, "steering_angle", 0.0)
        self._publish_cmd(speed, steer, dt_pub)

        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(dbg, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        self._publish_debug(dbg, np.zeros((Hf, Wf, 3), np.uint8), y0)


if __name__ == "__main__":
    VisionLKA()
    rospy.spin()

