#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Vision LKA (left-yellow tracking + curvature feed-forward)

- Detect left solid yellow via HSV ∩ LAB with morphology; ignore center band.
- Dense vertical scans -> robust left border points -> quadratic fit x(y) in normalized coords.
- From poly: heading and curvature; target point = left + offset (lane_w_px_est * lane_pos_frac).
- Control: PD on lateral error + K_heading*heading + K_curv*curvature (feed-forward).
- Steering-rate limit + EMA smoothing. Speed limited by curvature and lateral acceleration.
- Publishes AckermannDrive on topic "cmd".

Subs:  image (sensor_msgs/Image), /gem/safety/stop (Bool)
Pubs:  cmd (ackermann_msgs/AckermannDrive), debug (sensor_msgs/Image), lateral_error (std_msgs/Float32)
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
        self.br = CvBridge()

        # ---------- control ----------
        self.kp        = rospy.get_param("~kp",        0.085)
        self.kd        = rospy.get_param("~kd",        0.16)
        self.k_heading = rospy.get_param("~k_heading", 0.22)
        self.k_curv_ff = rospy.get_param("~k_curv_ff", 0.55)   # curvature feed-forward gain
        self.der_alpha = rospy.get_param("~der_alpha", 0.5)    # EMA for derivative

        # steering shaping
        self.steer_limit      = rospy.get_param("~steer_limit", 0.95)
        self.steer_rate_limit = rospy.get_param("~steer_rate_limit", 5.0)
        self.steer_alpha      = rospy.get_param("~steer_alpha", 0.12)
        self.steer_sign       = rospy.get_param("~steer_sign",  -1.0)  # set +1.0 if inverted
        self.steer_bias       = rospy.get_param("~steer_bias",   0.0)

        # speed & limits
        self.target_speed   = rospy.get_param("~target_speed", 0.55)
        self.min_speed      = rospy.get_param("~min_speed",    0.30)
        self.recovery_speed = rospy.get_param("~recovery_speed", 0.35)
        self.k_v_curv       = rospy.get_param("~k_v_curv",     0.65)  # v ≈ k_v_curv / |kappa_img|
        self.k_curve_speed  = rospy.get_param("~k_curve_speed", 8.0)  # extra slowdown by heading
        self.steer_slowdown = rospy.get_param("~steer_slowdown", 0.22)
        self.wheelbase      = rospy.get_param("~wheelbase",     1.2)
        self.a_accel_max    = rospy.get_param("~a_accel_max",   0.35)
        self.a_brake_max    = rospy.get_param("~a_brake_max",   1.20)
        self.a_lat_max      = rospy.get_param("~a_lat_max",     1.80)

        # ROI & geometry
        self.roi_top   = rospy.get_param("~roi_top", 0.56)
        self.center_cut_frac = rospy.get_param("~center_ignore_frac", 0.18)
        self.left_band_frac  = rospy.get_param("~left_band_frac",     0.80)  # keep only x <= 0.80W
        self.lookahead_row   = rospy.get_param("~lookahead_row",       0.85)  # y* inside ROI for target
        self.cam_center_shift_px = rospy.get_param("~cam_center_shift_px", 0.0)

        # yellow thresholds (robust for dark asphalt)
        self.yellow_h_lo  = rospy.get_param("~yellow_h_lo", 18)
        self.yellow_h_hi  = rospy.get_param("~yellow_h_hi", 42)
        self.yellow_s_min = rospy.get_param("~yellow_s_min", 65)
        self.yellow_v_min = rospy.get_param("~yellow_v_min", 22)
        self.lab_b_min    = rospy.get_param("~lab_b_min",  138)
        self.close_kernel = rospy.get_param("~close_kernel", 7)
        self.open_kernel  = rospy.get_param("~open_kernel",  5)

        # scans & fitting
        self.scan_y0_frac = rospy.get_param("~scan_y0_frac", 0.35)  # start of scans inside ROI
        self.scan_step_px = rospy.get_param("~scan_step_px",  8)
        self.min_run_px   = rospy.get_param("~edge_min_run_px", 10)
        self.min_pts_fit  = rospy.get_param("~min_pts_fit",  10)
        self.ransac_thr   = rospy.get_param("~ransac_thr",  0.06)   # in normalized x (0..1)
        self.fit_max_iter = rospy.get_param("~fit_max_iter", 3)

        # lane target offset
        self.lane_w_px_est   = rospy.get_param("~lane_w_px_est", 140.0)
        self.lane_pos_frac   = rospy.get_param("~lane_pos_frac", 0.60)  # 0.5 center; >0.5 closer to right
        self.center_alpha    = rospy.get_param("~center_alpha", 0.25)
        self.max_jump_frac   = rospy.get_param("~max_center_jump_frac", 0.08)

        # state
        self.prev_t = rospy.get_time()
        self.prev_err = 0.0
        self.prev_de_f = 0.0
        self.prev_delta = 0.0
        self.steer_filt = 0.0

        self.v_cmd = 0.0
        self.prev_t_speed = rospy.get_time()
        self.center_px_ema = None
        self.last_cmd = AckermannDrive()
        self.estop = False

        # I/O
        self.pub_cmd = rospy.Publisher("cmd", AckermannDrive, queue_size=10)
        self.pub_err = rospy.Publisher("lateral_error", Float32, queue_size=10)
        self.pub_dbg = rospy.Publisher("debug", Image, queue_size=1)
        rospy.Subscriber("image", Image, self.on_image, queue_size=1)
        rospy.Subscriber("/gem/safety/stop", Bool, self.on_stop, queue_size=1)

    # ---------- callbacks ----------
    def on_stop(self, msg: Bool):
        self.estop = bool(msg.data)

    def on_image(self, msg: Image):
        try:
            bgr_full = self.br.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "cv_bridge: %s", e); return

        Hf, Wf = bgr_full.shape[:2]
        y0 = int(clamp(self.roi_top, 0.0, 0.95) * Hf)
        roi = bgr_full[y0:Hf, :]
        H, W = roi.shape[:2]
        img_center = 0.5*W + float(self.cam_center_shift_px)

        # 1) yellow mask (HSV ∩ LAB)
        mask = self._yellow_mask(roi)
        # gate: cut center band and keep left 80%
        cut = int(self.center_cut_frac * W)
        mask[:, W//2-cut:W//2+cut] = 0
        mask[:, int(self.left_band_frac*W):] = 0

        # 2) collect left border points by vertical scans
        ys, xs = self._scan_left(mask)

        if len(xs) < self.min_pts_fit:
            self._recovery(mask, Wf, Hf, y0, "NO LEFT LINE")
            return

        # 3) fit quadratic in normalized coords: x_n = a*y_n^2 + b*y_n + c
        y_n = np.asarray(ys, np.float32) / float(H)
        x_n = (np.asarray(xs, np.float32) - 0.5*W) / (0.5*W)
        a,b,c = self._robust_quadfit(y_n, x_n)

        # 4) heading and curvature at lookahead
        y_star = clamp(self.lookahead_row, 0.0, 0.99)
        x_star_n = (a*(y_star**2) + b*y_star + c)                    # normalized x at y*
        dx_dy_n  = (2*a*y_star + b)
        d2x_dy2_n= (2*a)
        heading  = math.atan(dx_dy_n)
        curv_img = d2x_dy2_n / max(1e-6, (1.0 + dx_dy_n*dx_dy_n)**1.5)  # signed curvature in image-normalized coords

        # 5) target position = left + offset (pixels)
        x_left_px = (x_star_n * (0.5*W)) + 0.5*W
        target_x = x_left_px + float(self.lane_pos_frac) * float(self.lane_w_px_est)

        # innovation clamp + EMA
        if self.center_px_ema is None:
            self.center_px_ema = target_x
        else:
            max_jump = float(self.max_jump_frac) * W
            innov = clamp(target_x - self.center_px_ema, -max_jump, max_jump)
            self.center_px_ema = (1.0 - self.center_alpha)*(self.center_px_ema + innov) + self.center_alpha*target_x

        target_x = float(self.center_px_ema)
        err = (target_x - img_center) / max(1.0, 0.5*W)  # normalized lateral error

        # 6) control (PD + heading + curvature feed-forward)
        t  = rospy.get_time()
        dt = max(1e-3, t - self.prev_t)
        de = (err - self.prev_err)/dt
        de_f = self.der_alpha*self.prev_de_f + (1.0 - self.der_alpha)*de

        delta_cmd = -(self.kp*err + self.kd*de_f + self.k_heading*heading) \
                    - self.k_curv_ff*curv_img + float(self.steer_bias)

        max_step = float(self.steer_rate_limit) * dt
        delta_cmd = clamp(delta_cmd, self.prev_delta - max_step, self.prev_delta + max_step)
        delta_cmd = clamp(self.steer_sign * delta_cmd, -self.steer_limit, self.steer_limit)

        a_out = float(self.steer_alpha)
        self.steer_filt = (1.0 - a_out)*self.steer_filt + a_out*delta_cmd

        self.prev_err, self.prev_de_f, self.prev_delta, self.prev_t = err, de_f, delta_cmd, t

        # 7) speed profile: v ≈ k_v_curv/|κ| + additional slowdowns + a_lat clamp
        v_from_curv = self.k_v_curv / max(0.05, abs(curv_img))
        v_curve = 1.0 / (1.0 + self.k_curve_speed * abs(heading))
        v_steer = 1.0 - self.steer_slowdown * abs(self.steer_filt)
        v_des   = self.target_speed * min(1.0, v_curve, v_steer, v_from_curv)
        v_des   = clamp(v_des, self.min_speed, self.target_speed)

        # lateral acceleration constraint
        tan_d = abs(math.tan(self.steer_filt))
        if tan_d > 1e-4:
            v_lat_max = math.sqrt(max(0.0, self.a_lat_max * self.wheelbase / tan_d))
            v_des = min(v_des, v_lat_max)

        speed, dt_pub = self._ramped_speed(v_des, now=t)

        # 8) publish
        self.pub_err.publish(float(err))
        self._publish_cmd(speed, self.steer_filt, dt_pub)

        # 9) debug overlay
        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # draw fitted curve
        yy = np.linspace(0, H-1, 30, dtype=np.int32)
        yn = yy.astype(np.float32)/float(H)
        xn = (a*(yn**2) + b*yn + c)
        xx = (xn*(0.5*W) + 0.5*W).astype(np.int32)
        for i in range(len(yy)-1):
            cv2.line(dbg, (int(xx[i]), int(yy[i])), (int(xx[i+1]), int(yy[i+1])), (0,255,255), 2)
        # draw center and target
        cv2.line(dbg, (int(img_center), H-1), (int(img_center), H-40), (255,0,0), 2)
        cv2.line(dbg, (int(target_x),  H-1), (int(target_x),  H-40), (0,0,255), 2)
        txt = f"e={err:+.2f} hd={heading:+.2f} k={curv_img:+.2f} d={self.steer_filt:+.2f} v={speed:.2f}"
        cv2.putText(dbg, txt, (10, max(18, H-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1, cv2.LINE_AA)
        self._publish_debug(dbg, bgr_full, y0)

    # ---------- vision helpers ----------
    def _yellow_mask(self, roi_bgr):
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
        m_hsv = cv2.inRange(hsv,
                            (int(self.yellow_h_lo), int(self.yellow_s_min), int(self.yellow_v_min)),
                            (int(self.yellow_h_hi), 255, 255))
        m_lab = cv2.inRange(lab[:,:,2], int(self.lab_b_min), 255)
        mask = cv2.bitwise_and(m_hsv, m_lab)
        if self.close_kernel > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                    np.ones((int(self.close_kernel), int(self.close_kernel)), np.uint8))
        if self.open_kernel > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                    np.ones((int(self.open_kernel), int(self.open_kernel)), np.uint8))
        return mask

    def _scan_left(self, mask):
        H, W = mask.shape[:2]
        y_start = int(clamp(self.scan_y0_frac, 0.0, 0.95)*H)
        ys, xs = [], []
        min_run = int(self.min_run_px)
        xlim = int(self.left_band_frac * W)
        for y in range(y_start, H-2, max(1, int(self.scan_step_px))):
            row = mask[y, :xlim]
            run = 0; idx = None
            for x in range(xlim-1, -1, -1):
                if row[x] > 0:
                    run += 1
                    if run >= min_run:
                        idx = x + run//2
                        break
                else:
                    run = 0
            if idx is not None:
                ys.append(y); xs.append(float(idx))
        return ys, xs

    def _robust_quadfit(self, y_n, x_n):
        # simple trimmed LS: iterate, drop big residuals
        a, b, c = np.polyfit(y_n, x_n, 2)
        for _ in range(int(self.fit_max_iter)):
            pred = a*(y_n**2) + b*y_n + c
            res = np.abs(pred - x_n)
            thr = max(float(self.ransac_thr), float(np.median(res)*1.5))
            keep = res <= thr
            if keep.sum() < max(6, int(0.5*len(y_n))):
                break
            a, b, c = np.polyfit(y_n[keep], x_n[keep], 2)
        return float(a), float(b), float(c)

    # ---------- command & debug ----------
    def _ramped_speed(self, v_des, now=None):
        if now is None: now = rospy.get_time()
        dt = max(1e-3, now - self.prev_t_speed)
        dv = float(v_des) - float(self.v_cmd)
        dv_max = (self.a_accel_max*dt) if dv>=0.0 else (self.a_brake_max*dt)
        dv = clamp(dv, -abs(dv_max), abs(dv_max))
        self.v_cmd = clamp(self.v_cmd + dv, 0.0, self.target_speed)
        self.prev_t_speed = now
        return self.v_cmd, dt

    def _publish_cmd(self, speed, steer, dt=None):
        if self.estop: speed, steer = 0.0, 0.0
        now = rospy.get_time()
        if dt is None: dt = max(1e-3, now - getattr(self, "_last_pub_t", now))
        v_prev = float(getattr(self, "_last_pub_speed", 0.0))
        acc_prev = float(getattr(self, "_last_pub_acc", 0.0))
        acc = (speed - v_prev)/dt
        jerk = (acc - acc_prev)/dt

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

    def _publish_debug(self, roi_bgr, bgr_full, y0):
        Hf, Wf = bgr_full.shape[:2]
        canv = np.zeros((Hf, Wf, 3), dtype=np.uint8)
        try:
            canv[y0:Hf, :] = cv2.resize(roi_bgr, (Wf, Hf - y0))
            self.pub_dbg.publish(self.br.cv2_to_imgmsg(canv, encoding='bgr8'))
        except Exception as e:
            rospy.logwarn_throttle(2.0, "debug publish failed: %s", e)

    def _recovery(self, mask, Wf, Hf, y0, label):
        v_des = 0.0 if False else self.recovery_speed
        speed, dt_pub = self._ramped_speed(v_des, now=rospy.get_time())
        steer = getattr(self.last_cmd, "steering_angle", 0.0)
        self._publish_cmd(speed, steer, dt_pub)
        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(dbg, label, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        self._publish_debug(dbg, np.zeros((Hf, Wf, 3), np.uint8), y0)

if __name__ == "__main__":
    VisionLKA()
    rospy.spin()

