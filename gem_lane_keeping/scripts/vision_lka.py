#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GEM vision LKA with soft speed planner:
- curvature-based speed cap
- accel/decel rate limiting
- slow re-acquire ramp after lane loss
- graceful slowdown when lane is lost
"""

import rospy, cv2, numpy as np
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge

def clamp(v, lo, hi): return lo if v < lo else hi if v > hi else v

class VisionLKA:
    def __init__(self):
        rospy.init_node("vision_lka")
        self.bridge = CvBridge()

        # ===== Control (steering) =====
        self.kp           = float(rospy.get_param("~kp", 0.015))
        self.kd           = float(rospy.get_param("~kd", 0.050))
        self.k_heading    = float(rospy.get_param("~k_heading", 0.005))
        self.steer_limit  = float(rospy.get_param("~steer_limit", 0.40))
        self.alpha_steer  = float(rospy.get_param("~alpha_steer", 0.35))   # 0..1, larger = smoother
        self.steer_rate_lim = float(rospy.get_param("~steer_rate_limit", 0.04))  # rad per frame
        self.invert_steer = bool(rospy.get_param("~invert_steer", True))

        # ===== Speed planner =====
        self.target_speed = float(rospy.get_param("~target_speed", 1.5))  # m/s
        self.k_slow       = float(rospy.get_param("~k_slow", 0.65))       # 0..1, slow on curves
        self.max_accel    = float(rospy.get_param("~max_accel", 0.3))     # m/s^2
        self.max_decel    = float(rospy.get_param("~max_decel", 0.9))     # m/s^2 (positive number)
        # when lane is re-acquired: start slow and ramp
        self.reacq_v      = float(rospy.get_param("~reacquire_v", 0.6))   # m/s
        self.reacq_time   = float(rospy.get_param("~reacquire_ramp_s", 2.5)) # s to ramp from reacq_v
        self.stop_brake   = float(rospy.get_param("~stop_decel", 1.2))    # m/s^2 when we must stop

        # ===== ROI & scanning =====
        self.roi_top      = float(rospy.get_param("~roi_y_top",
                             rospy.get_param("~roi_top", 0.55)))
        self.scan_rows    = list(rospy.get_param("~scan_rows", [0.86,0.90,0.94,0.97]))
        self.min_valid_rows = int(rospy.get_param("~min_valid_rows", 1))
        self.min_mask_px  = int(rospy.get_param("~min_mask_px", 120))
        self.min_lane_w   = int(rospy.get_param("~min_lane_width_px", 12))
        self.scan_margin  = int(rospy.get_param("~scan_margin", 14))
        self.hold_bad_ms  = int(rospy.get_param("~hold_bad_ms", 600))

        # ===== Color/edges =====
        self.white_s_max  = int(rospy.get_param("~white_s_max", 90))
        self.white_v_min  = int(rospy.get_param("~white_v_min", 150))
        self.hls_L_min    = int(rospy.get_param("~hls_L_min", 190))
        self.lab_b_min    = int(rospy.get_param("~lab_b_min", 140))
        self.canny_low    = int(rospy.get_param("~canny_low", 0))
        self.canny_high   = int(rospy.get_param("~canny_high", 0))
        self.morph_kernel = int(rospy.get_param("~morph_kernel", 7))

        # ===== State =====
        self.estop        = False
        self.prev_err     = 0.0
        self.prev_t       = rospy.get_time()
        self.prev_steer   = 0.0
        self.last_ok_time = rospy.Time(0)
        self.had_lane     = False

        # speed state
        self.v_out        = 0.0           # actually commanded (rate-limited) speed
        self.v_ref        = 0.0           # desired/planned speed before rate-limit
        self.reacq_until  = rospy.Time(0) # end of re-acquire phase

        # ===== I/O =====
        self.pub_cmd = rospy.Publisher("cmd", AckermannDrive, queue_size=10)
        self.pub_err = rospy.Publisher("lateral_error", Float32, queue_size=10)
        self.pub_dbg = rospy.Publisher("debug", Image, queue_size=1)
        rospy.Subscriber("image", Image, self.on_image, queue_size=1, buff_size=2**24)
        rospy.Subscriber("/gem/safety/stop", Bool, self.on_stop, queue_size=1)

        rospy.loginfo("LKA: v=%.2f kp=%.3f kd=%.3f kh=%.3f steer_lim=%.2f "
                      "alpha=%.2f dSteer<=%.3f k_slow=%.2f acc<=%.2f dec<=%.2f reacq_v=%.2f t=%.1fs",
                      self.target_speed, self.kp, self.kd, self.k_heading, self.steer_limit,
                      self.alpha_steer, self.steer_rate_lim, self.k_slow,
                      self.max_accel, self.max_decel, self.reacq_v, self.reacq_time)

    # ---------- utils ----------
    def on_stop(self, msg: Bool): self.estop = bool(msg.data)

    def _rate_limit(self, target, current, rate, dt):
        """Limit |target-current| by rate*dt."""
        max_step = rate * dt
        return current + clamp(target - current, -max_step, +max_step)

    def _publish(self, speed, steer):
        """Steering smoothing + rate limit + E-stop gating."""
        if self.estop:
            speed, steer = 0.0, 0.0
        if self.invert_steer:
            steer = -steer

        # steer low-pass
        raw = clamp(steer, -self.steer_limit, self.steer_limit)
        filt = (1.0 - self.alpha_steer) * self.prev_steer + self.alpha_steer * raw
        d = clamp(filt - self.prev_steer, -self.steer_rate_lim, self.steer_rate_lim)
        self.prev_steer += d

        cmd = AckermannDrive()
        cmd.speed = float(max(0.0, speed))
        cmd.steering_angle = float(clamp(self.prev_steer, -self.steer_limit, self.steer_limit))
        # (Some controllers ignore acceleration, but we set it consistently)
        cmd.acceleration = 0.0
        self.pub_cmd.publish(cmd)

    # ---------- main image callback ----------
    def on_image(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "cv_bridge: %s", e); return

        h, w = bgr.shape[:2]
        y0 = int(h * clamp(self.roi_top, 0.0, 0.98))
        roi = bgr[y0:h, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        white_hsv = cv2.inRange(hsv, (0, 0, self.white_v_min), (179, self.white_s_max, 255))
        _, white_hls = cv2.threshold(hls[:,:,1], self.hls_L_min, 255, cv2.THRESH_BINARY)
        _, yellow_lab = cv2.threshold(lab[:,:,2], self.lab_b_min, 255, cv2.THRESH_BINARY)

        mask = cv2.bitwise_or(white_hsv, white_hls)
        mask = cv2.bitwise_or(mask, yellow_lab)

        if self.canny_high > 0:
            edges = cv2.Canny(gray, self.canny_low, self.canny_high)
            bright = cv2.inRange(gray, 140, 255)
            edges = cv2.bitwise_and(edges, bright)
            edges[:, :10] = 0; edges[:, -10:] = 0; edges[:10, :] = 0
            mask = cv2.bitwise_or(mask, edges)

        k = max(3, self.morph_kernel | 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k,k), np.uint8))
        mask = cv2.medianBlur(mask, 5)

        nz = int(np.count_nonzero(mask))
        now = rospy.Time.now()
        t  = rospy.get_time()
        dt = max(1e-3, t - self.prev_t)
        self.prev_t = t

        had_lane_prev = self.had_lane

        if nz < self.min_mask_px:
            # lane lost -> decelerate smoothly and hold last steer for a short time
            self.had_lane = False
            # ramp speed down
            self.v_ref = 0.0
            self.v_out = self._rate_limit(self.v_ref, self.v_out, self.stop_brake, dt)
            self._publish(self.v_out, self.prev_steer)
            self._debug(mask, w, h, y0, txt="NO LANE (area)")
            return

        # scan rows
        ys = mask.shape[0]
        centers, used_y = [], []
        margin = max(0, int(self.scan_margin))
        for r in self.scan_rows:
            y_scan = int(clamp(r, 0.0, 0.999) * ys)
            row = mask[y_scan, :]
            row = row[margin: row.shape[0] - margin]
            xs = np.flatnonzero(row)
            if xs.size >= self.min_lane_w:
                L = int(xs[0]) + margin
                R = int(xs[-1]) + margin
                if (R - L) >= self.min_lane_w:
                    centers.append(0.5*(L+R))
                    used_y.append(y_scan)

        if len(centers) < self.min_valid_rows:
            self.had_lane = False
            self.v_ref = 0.0
            self.v_out = self._rate_limit(self.v_ref, self.v_out, self.stop_brake, dt)
            self._publish(self.v_out, self.prev_steer)
            self._debug(mask, w, h, y0, txt="NO LANE (rows)")
            return

        # lane valid
        self.had_lane = True
        lane_center_px = float(np.mean(centers[-max(1, len(centers)//2):]))
        img_center_px  = 0.5 * w
        err = (img_center_px - lane_center_px) / (w*0.5)

        # heading from centers trend
        heading = 0.0
        if len(centers) >= 2:
            z = np.polyfit(np.asarray(used_y, np.float32), np.asarray(centers, np.float32), 1)
            slope = z[0]
            heading = -float(slope) / (w*0.5)

        # PD steering
        d_err = (err - self.prev_err) / dt
        self.prev_err = err
        steer_raw = self.kp*err + self.kd*d_err + self.k_heading*heading
        steer_raw = clamp(steer_raw, -self.steer_limit, self.steer_limit)

        # ----- SPEED PLANNER -----
        # curvature proxy from desired steer
        curvature = min(1.0, abs(steer_raw) / max(1e-6, self.steer_limit))
        plan = self.target_speed * (1.0 - self.k_slow * curvature)   # 0..target_speed

        # if re-acquiring lane -> start at low speed and ramp
        if not had_lane_prev and self.had_lane:
            self.reacq_until = now + rospy.Duration.from_sec(self.reacq_time)
            # immediately clamp reference down to reacq_v
            plan = min(plan, self.reacq_v)

        if now < self.reacq_until:
            plan = min(plan, self.reacq_v)  # keep capped while in reacquire window

        # rate-limit speed (accel/decel)
        # first move v_ref towards plan at accel limits…
        self.v_ref = plan
        # then move v_out towards v_ref with separate accel/decel limits
        if self.v_ref >= self.v_out:
            self.v_out = self._rate_limit(self.v_ref, self.v_out, self.max_accel, dt)
        else:
            self.v_out = self._rate_limit(self.v_ref, self.v_out, self.max_decel, dt)

        # Publish
        self._publish(self.v_out, steer_raw)
        self.last_ok_time = now

        # Debug
        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for y_scan, cx in zip(used_y, centers):
            cv2.line(dbg, (int(cx), y_scan), (int(cx), max(0, y_scan-28)), (0,0,255), 2)
        cv2.line(dbg, (int(img_center_px), ys-6), (int(img_center_px), ys-42), (255,0,0), 1)
        text = f"nz={nz} err={err:+.2f} dE={d_err:+.2f} hd={heading:+.3f} steer={steer_raw:+.2f} v={self.v_out:.2f}"
        cv2.putText(dbg, text, (10, dbg.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1, cv2.LINE_AA)
        self._publish_debug(dbg, w, h, y0)

    # ---------- debug helpers ----------
    def _debug(self, mask, w, h, y0, txt):
        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(dbg, txt, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
        self._publish_debug(dbg, w, h, y0)

    def _publish_debug(self, roi_bgr, w, h, y0):
        canv = np.zeros((h, w, 3), dtype=np.uint8)
        if roi_bgr.ndim == 2: roi_bgr = cv2.cvtColor(roi_bgr, cv2.COLOR_GRAY2BGR)
        canv[y0:h, :] = cv2.resize(roi_bgr, (w, h-y0))
        try:
            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(canv, encoding='bgr8'))
        except Exception as e:
            rospy.logwarn_throttle(2.0, "debug publish failed: %s", e)


if __name__ == "__main__":
    VisionLKA()
    rospy.spin()

