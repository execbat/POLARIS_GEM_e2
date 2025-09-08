#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision LKA (final): robust left/right border detection and lane holding.

- Color: HSV∩LAB mask for yellow (tuned for dark asphalt).
- Geometry: left/right gating; ignore center white dashed.
- Edges: Canny + Hough in left/right zones; pick borders by slope sign.
- Border estimate = median of bottom intersections of candidate lines.
- Lane width filtered; if only left exists -> extrapolate from last width.
- Target = fixed fraction between borders (lane_pos_frac), with EMA & innovation clamp.
- Control = PD on lateral error + heading, steering-rate limit + EMA.
- Speed = curve/steer slowdown + lateral-accel and dv/dt limits.
- Publishes AckermannDrive on "cmd" (remap to /gem/ackermann_cmd).

Topics:
  sub: image (sensor_msgs/Image), /gem/safety/stop (Bool)
  pub: cmd (AckermannDrive), debug (Image), lateral_error (Float32)
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

        # --- control (PD + heading)
        self.kp        = rospy.get_param("~kp",        0.08)
        self.kd        = rospy.get_param("~kd",        0.14)
        self.k_heading = rospy.get_param("~k_heading", 0.22)
        self.der_alpha = rospy.get_param("~der_alpha", 0.5)

        # --- steering shaping
        self.steer_limit      = rospy.get_param("~steer_limit", 0.90)
        self.steer_rate_limit = rospy.get_param("~steer_rate_limit", 4.0)
        self.steer_alpha      = rospy.get_param("~steer_alpha", 0.12)
        self.steer_sign       = rospy.get_param("~steer_sign",  -1.0)  # set to +1.0 if inverted
        self.steer_bias       = rospy.get_param("~steer_bias",   0.0)

        # --- speed & limits
        self.target_speed   = rospy.get_param("~target_speed", 0.55)
        self.min_speed      = rospy.get_param("~min_speed",    0.30)
        self.recovery_speed = rospy.get_param("~recovery_speed", 0.40)
        self.k_curve_speed  = rospy.get_param("~k_curve_speed", 8.0)
        self.steer_slowdown = rospy.get_param("~steer_slowdown", 0.22)
        self.wheelbase      = rospy.get_param("~wheelbase", 1.2)
        self.a_accel_max    = rospy.get_param("~a_accel_max", 0.35)
        self.a_brake_max    = rospy.get_param("~a_brake_max", 1.20)
        self.a_lat_max      = rospy.get_param("~a_lat_max",   1.80)

        # --- ROI & gating
        self.roi_top       = rospy.get_param("~roi_top", 0.58)
        self.center_cut    = rospy.get_param("~center_ignore_frac", 0.12)
        self.left_band     = rospy.get_param("~left_band_frac", 0.80)   # allow left up to 80% of width
        self.right_band    = rospy.get_param("~right_band_frac", 0.80)  # allow right from 20% rightwards

        # --- Hough/Canny
        self.canny1 = rospy.get_param("~canny1", 60)
        self.canny2 = rospy.get_param("~canny2", 150)
        self.hough_threshold  = rospy.get_param("~hough_threshold", 25)
        self.hough_min_len    = rospy.get_param("~hough_min_length", 60)
        self.hough_max_gap    = rospy.get_param("~hough_max_gap",  20)
        self.hough_min_ang_deg= rospy.get_param("~hough_min_angle_deg", 15)

        # --- yellow thresholds
        self.yellow_h_lo  = rospy.get_param("~yellow_h_lo", 18)
        self.yellow_h_hi  = rospy.get_param("~yellow_h_hi", 42)
        self.yellow_s_min = rospy.get_param("~yellow_s_min", 70)
        self.yellow_v_min = rospy.get_param("~yellow_v_min", 25)
        self.lab_b_min    = rospy.get_param("~lab_b_min",  140)
        self.close_kernel = rospy.get_param("~close_kernel", 7)
        self.open_kernel  = rospy.get_param("~open_kernel",  5)

        # --- target placement
        self.lane_pos_frac        = rospy.get_param("~lane_pos_frac", 0.60)  # 0.5 center; 0.6 = a bit right of left
        self.center_alpha         = rospy.get_param("~center_alpha", 0.25)
        self.max_center_jump_frac = rospy.get_param("~max_center_jump_frac", 0.08)
        self.cam_center_shift_px  = rospy.get_param("~cam_center_shift_px", 0.0)

        # --- scan assist
        self.scan_rows       = rospy.get_param("~scan_rows", [0.75,0.85,0.93,0.98])
        self.min_valid_rows  = rospy.get_param("~min_valid_rows", 1)
        self.edge_min_run_px = rospy.get_param("~edge_min_run_px", 12)
        self.hold_bad_ms     = rospy.get_param("~hold_bad_ms", 500)
        self.stop_if_lost    = rospy.get_param("~stop_if_lost", False)

        # --- state
        self.prev_t = rospy.get_time()
        self.prev_err = 0.0
        self.prev_de_f = 0.0
        self.prev_delta = 0.0
        self.steer_filt = 0.0
        self.v_cmd = 0.0
        self.prev_t_speed = rospy.get_time()
        self.last_ok_time = rospy.Time(0.0)
        self.last_lane_w_px = 140.0  # sensible default
        self.center_px_ema = None
        self.last_cmd = AckermannDrive()
        self.estop = False

        # --- IO
        self.pub_cmd = rospy.Publisher("cmd", AckermannDrive, queue_size=10)
        self.pub_err = rospy.Publisher("lateral_error", Float32, queue_size=10)
        self.pub_dbg = rospy.Publisher("debug", Image, queue_size=1)
        rospy.Subscriber("image", Image, self.on_image, queue_size=1)
        rospy.Subscriber("/gem/safety/stop", Bool, self.on_stop, queue_size=1)

    # ====== callbacks ======
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

        # 1) color mask for yellow (HSV ∩ LAB-b), robust to dark asphalt
        mask_yel = self._yellow_mask(roi)

        # 2) edges & Hough
        center_cut = int(self.center_cut*W)
        left_lim   = int(self.left_band*W)
        right_lim  = int((1.0 - self.right_band)*W)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), self.canny1, self.canny2)

        # gate center to suppress white dashed
        edges[:, W//2-center_cut:W//2+center_cut] = 0

        # run Hough on whole ROI, then filter by geometry + slope sign
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, self.hough_threshold,
                                minLineLength=self.hough_min_len, maxLineGap=self.hough_max_gap)

        min_ang = np.deg2rad(self.hough_min_ang_deg)
        left_bottom_x, right_bottom_x = [], []
        overlay = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        if lines is not None:
            for x1,y1,x2,y2 in lines[:,0,:]:
                dx, dy = (x2-x1), (y2-y1)
                if dy == 0: continue
                ang = abs(math.atan2(dy, dx))
                if ang < min_ang:  # near-horizontal → ignore
                    continue
                # bottom intersection
                xb = x1 + (H-1 - y1) * (dx / float(dy))

                if dx*dy < 0 and xb <= left_lim:         # negative slope → left border
                    left_bottom_x.append(float(xb))
                    cv2.line(overlay, (x1,y1), (x2,y2), (0,255,0), 2)
                elif dx*dy > 0 and xb >= right_lim:      # positive slope → right border
                    right_bottom_x.append(float(xb))
                    cv2.line(overlay, (x1,y1), (x2,y2), (255,0,0), 2)

        # 3) robust border estimates (median of bottom intersections)
        left_xb  = float(np.median(left_bottom_x))  if left_bottom_x  else None
        right_xb = float(np.median(right_bottom_x)) if right_bottom_x else None

        # fallback: use color-scan if Hough gave nothing
        if left_xb is None:
            mleft = self._gate_left(mask_yel.copy(), W)
            Lxs, Lys = self._scan_rows(mleft, side="left", W=W, H=H)
            if Lxs: left_xb = float(np.median(Lxs))
        if right_xb is None:
            mright = self._gate_right(mask_yel.copy(), W)
            Rxs, Rys = self._scan_rows(mright, side="right", W=W, H=H)
            if Rxs: right_xb = float(np.median(Rxs))

        have_L = left_xb is not None
        have_R = right_xb is not None

        # lane width update
        lane_w = None
        if have_L and have_R and right_xb > left_xb:
            lane_w = right_xb - left_xb
        if lane_w is not None and lane_w > 10:
            self.last_lane_w_px = 0.8*self.last_lane_w_px + 0.2*float(lane_w)

        # desired target x at bottom
        if have_L and have_R:
            desired_x = left_xb + float(self.lane_pos_frac) * (right_xb - left_xb)
        elif have_L:
            desired_x = left_xb + float(self.lane_pos_frac) * float(self.last_lane_w_px)
        else:
            self._recovery(mask_yel, Wf, Hf, y0, "NO LANE")
            return

        # innovation clamp + EMA
        if self.center_px_ema is None:
            self.center_px_ema = desired_x
        else:
            max_jump = float(self.max_center_jump_frac) * W
            innov = clamp(desired_x - self.center_px_ema, -max_jump, max_jump)
            self.center_px_ema = (1.0 - self.center_alpha)*(self.center_px_ema + innov) + self.center_alpha*desired_x
        target_x = float(self.center_px_ema)

        # heading: use leftmost good line orientation if present
        heading = 0.0
        if lines is not None:
            # choose longest left (neg slope) line for heading
            best_len = 0.0
            for x1,y1,x2,y2 in lines[:,0,:]:
                dx, dy = (x2-x1), (y2-y1)
                if dy == 0: continue
                xb = x1 + (H-1 - y1) * (dx / float(dy))
                if dx*dy < 0 and xb <= left_lim:  # left candidate
                    L = math.hypot(dx, dy)
                    if L > best_len:
                        best_len = L
                        k = (dx / float(dy))  # x as function of y
                        heading = math.atan(k)  # small-angle approx w.r.t. image axes

        # lateral error (normalize by half width)
        err = (target_x - img_center) / max(1.0, 0.5*W)

        # PD + heading; rate limit; EMA; sign/bias
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

        # speed profile + physical limits
        v_curve = 1.0 / (1.0 + self.k_curve_speed * abs(heading))
        v_steer = 1.0 - self.steer_slowdown * abs(self.steer_filt)
        v_scale = max(0.25, min(v_curve, v_steer))
        v_des   = clamp(self.target_speed * v_scale, self.min_speed, self.target_speed)

        tan_d = abs(math.tan(self.steer_filt))
        if tan_d > 1e-4:
            v_lat_max = math.sqrt(max(0.0, self.a_lat_max * self.wheelbase / tan_d))
            v_des = min(v_des, v_lat_max)

        speed, dt_pub = self._ramped_speed(v_des, now=t)

        # publish
        self.pub_err.publish(float(err))
        self._publish_cmd(speed, self.steer_filt, dt_pub)
        self.last_ok_time = rospy.Time.now()

        # debug overlay
        dbg = cv2.cvtColor(mask_yel, cv2.COLOR_GRAY2BGR)
        dbg = cv2.addWeighted(dbg, 1.0, overlay, 0.8, 0.0)
        cv2.line(dbg, (int(img_center), H-1), (int(img_center), H-40), (255,0,0), 2)
        cv2.line(dbg, (int(target_x),  H-1), (int(target_x),  H-40), (0,0,255), 2)
        txt = f"e={err:+.2f} hd={heading:+.2f} d={self.steer_filt:+.2f} v={speed:.2f}"
        cv2.putText(dbg, txt, (10, max(16, H-12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1, cv2.LINE_AA)
        self._publish_debug(dbg, bgr_full, y0)

    # ====== detectors/helpers ======
    def _yellow_mask(self, roi_bgr):
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)

        m_hsv = cv2.inRange(
            hsv,
            (int(self.yellow_h_lo), int(self.yellow_s_min), int(self.yellow_v_min)),
            (int(self.yellow_h_hi), 255, 255)
        )
        m_lab = cv2.inRange(lab[:,:,2], int(self.lab_b_min), 255)
        mask = cv2.bitwise_and(m_hsv, m_lab)

        if self.close_kernel > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((int(self.close_kernel), int(self.close_kernel)), np.uint8))
        if self.open_kernel > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((int(self.open_kernel),  int(self.open_kernel)),  np.uint8))
        return mask

    def _gate_left(self, mask, W):
        out = mask.copy()
        c = int(self.center_cut*W)
        out[:, W//2-c:W//2+c] = 0
        out[:, int(self.left_band*W):] = 0
        return out

    def _gate_right(self, mask, W):
        out = mask.copy()
        c = int(self.center_cut*W)
        out[:, W//2-c:W//2+c] = 0
        out[:, :int((1.0 - self.right_band)*W)] = 0
        return out

    def _scan_rows(self, mask, side, W, H):
        xs, ys = [], []
        min_run = int(self.edge_min_run_px)
        if side == "left":
            xlim = int(self.left_band*W)
            for r in self.scan_rows:
                y = int(H*float(r)); y = min(max(y,0), H-1)
                row = mask[y,:xlim]
                run=0; idx=None
                for x in range(xlim-1,-1,-1):
                    if row[x]>0:
                        run+=1
                        if run>=min_run: idx=x+run//2; break
                    else: run=0
                if idx is not None: xs.append(float(idx)); ys.append(y)
        else:
            x0 = int((1.0 - self.right_band)*W)
            for r in self.scan_rows:
                y = int(H*float(r)); y = min(max(y,0), H-1)
                row = mask[y,x0:]
                run=0; idx=None
                for i,p in enumerate(row):
                    if p>0:
                        run+=1
                        if run>=min_run: idx=x0+i-run//2; break
                    else: run=0
                if idx is not None: xs.append(float(idx)); ys.append(y)
        return xs, ys

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

