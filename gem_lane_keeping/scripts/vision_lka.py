#!/usr/bin/env python3
import rospy, cv2, numpy as np, math
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge
from math import exp

def clamp(v, lo, hi): 
    return lo if v < lo else (hi if v > hi else v)

class VisionLKA:
    """
    Vision-based lane keeping with robust smoothing:
    - Multi-row scan to estimate lane center.
    - LPF on error and steering.
    - Steering slew-rate limit.
    - Speed scheduling vs steering magnitude.
    - Tiny optional integral with anti-windup.
    - Safe fallback when lane is not visible.
    """

    def __init__(self):
        rospy.init_node("vision_lka")
        self.bridge = CvBridge()

        # ---------- Control gains (conservative baseline) ----------
        self.kp           = rospy.get_param("~kp", 0.35)     # main proportional gain (on normalized lateral error)
        self.kd           = rospy.get_param("~kd", 0.10)     # derivative gain (on filtered error)
        self.ki           = rospy.get_param("~ki", 0.00)     # <= keep near 0 to avoid windup for now
        self.k_heading    = rospy.get_param("~k_heading", 0.10)  # heading (lane tilt) weight, modest

        # ---------- Steering limits & smoothing ----------
        self.steer_limit      = rospy.get_param("~steer_limit", 0.60)   # [rad] abs steering cap
        self.steer_rate_limit = rospy.get_param("~steer_rate_limit", 0.8) # [rad/s] max change per second
        self.err_lpf_tau      = rospy.get_param("~err_lpf_tau", 0.25)   # [s] error LPF time constant
        self.steer_lpf_tau    = rospy.get_param("~steer_lpf_tau", 0.15) # [s] steering LPF time constant
        self.err_deadband     = rospy.get_param("~err_deadband", 0.03)  # normalized error deadband

        # ---------- Speed scheduling vs steering ----------
        self.target_speed     = rospy.get_param("~target_speed", 1.30)  # [m/s] nominal cruise
        self.min_speed        = rospy.get_param("~min_speed", 0.60)     # [m/s] min when turning hard
        self.slowdown_gain    = rospy.get_param("~steer_slowdown", 0.80) # 0..1 how strong to reduce by |steer|
        self.slowdown_pow     = rospy.get_param("~slowdown_pow", 1.2)

        # ---------- Lane detection (ROI + color masks) ----------
        # ROI trapezoid (bottom = full width, top = fraction), starts at roi_y_top*H
        self.roi_y_top        = rospy.get_param("~roi_y_top", 0.58)     # 0..1 from top
        self.roi_top_width    = rospy.get_param("~roi_top_width_frac", 0.75) # fraction of width at ROI top

        # Multi-scan rows (relative to ROI height)
        self.scan_rows        = rospy.get_param("~scan_rows", [0.60, 0.72, 0.84, 0.92])
        self.min_mask_px      = rospy.get_param("~min_mask_px", 150)
        self.min_lane_w       = rospy.get_param("~min_lane_width_px", 22)
        self.min_valid_rows   = rospy.get_param("~min_valid_rows", 2)

        # HSV/HLS/LAB thresholds (whites/yellows)
        self.white_s_max      = rospy.get_param("~white_s_max", 70)     # HSV S <=
        self.white_v_min      = rospy.get_param("~white_v_min", 165)    # HSV V >=
        self.hls_L_min        = rospy.get_param("~hls_L_min", 190)      # HLS L >=
        self.lab_b_min        = rospy.get_param("~lab_b_min", 140)      # LAB b >= (yellow)
        self.morph_k          = rospy.get_param("~morph_kernel", 5)     # morphology kernel
        self.canny_low        = rospy.get_param("~canny_low", 30)       # optional edges assist
        self.canny_high       = rospy.get_param("~canny_high", 100)

        # Behavior when lane lost
        self.hold_bad_ms      = rospy.get_param("~hold_bad_ms", 400)    # ms to hold last command before stopping
        self.invert_steer     = rospy.get_param("~invert_steer", True) # flip sign if steering polarity is opposite

        # ---------- State ----------
        self.estop        = False
        self.e_filt       = 0.0    # filtered error
        self.e_int        = 0.0    # integral term
        self.last_t       = rospy.get_time()
        self.steer_filt   = 0.0    # filtered steering
        self.last_cmd     = AckermannDrive()
        self.last_ok_time = rospy.Time(0)

        # ---------- IO ----------
        self.pub_cmd = rospy.Publisher("cmd", AckermannDrive, queue_size=10)
        self.pub_err = rospy.Publisher("lateral_error", Float32, queue_size=10)
        self.pub_dbg = rospy.Publisher("debug", Image, queue_size=1)
        rospy.Subscriber("image", Image, self.on_image, queue_size=1)
        rospy.Subscriber("/gem/safety/stop", Bool, self.on_stop, queue_size=1)

        rospy.loginfo("LKA init: KP=%.2f KD=%.2f KI=%.3f KH=%.2f | steer_lim=%.2f rate=%.2f | "
                      "err_tau=%.2f steer_tau=%.2f deadband=%.2f | v=%.2f..%.2f slow(%.2f,pow=%.2f) | "
                      "ROI(y=%.2f, top=%.2f) rows=%s",
                      self.kp, self.kd, self.ki, self.k_heading,
                      self.steer_limit, self.steer_rate_limit,
                      self.err_lpf_tau, self.steer_lpf_tau, self.err_deadband,
                      self.min_speed, self.target_speed, self.slowdown_gain, self.slowdown_pow,
                      self.roi_y_top, self.roi_top_width, str(self.scan_rows))

    # -------------------- Callbacks --------------------

    def on_stop(self, msg: Bool):
        self.estop = bool(msg.data)

    def on_image(self, msg: Image):
        # Time & dt
        t = rospy.get_time()
        dt = max(1e-3, t - self.last_t)
        self.last_t = t

        # 1) Convert image and build trapezoid ROI
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "cv_bridge error: %s", e)
            return

        H, W = bgr.shape[:2]
        y0 = int(self.roi_y_top * H)
        if y0 >= H-2: y0 = max(0, H-2)
        roi_h = H - y0

        # trapezoid mask: top width fraction, bottom full width
        top_w = int(self.roi_top_width * W)
        l_top = (W - top_w) // 2
        r_top = l_top + top_w
        trapezoid = np.array([[0, roi_h-1],
                              [W-1, roi_h-1],
                              [r_top, 0],
                              [l_top, 0]], dtype=np.int32)

        roi_full = bgr[y0:H, :]
        roi = np.zeros_like(roi_full)
        cv2.fillPoly(roi, [trapezoid], (255,255,255))
        roi = cv2.bitwise_and(roi_full, roi)

        # 2) Build lane mask (white OR yellow) with gentle morphology
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

        white_hsv = cv2.inRange(hsv, (0, 0, self.white_v_min), (179, self.white_s_max, 255))
        white_hls = cv2.inRange(hls[:,:,1], self.hls_L_min, 255)
        yellow_lab = cv2.inRange(lab[:,:,2], self.lab_b_min, 255)

        mask = cv2.bitwise_or(white_hsv, white_hls)
        mask = cv2.bitwise_or(mask, yellow_lab)

        k = self.morph_k
        if k > 1:
            kernel = np.ones((k,k), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.medianBlur(mask, 5)

        # Light edge assist to keep borders during turns
        edges = cv2.Canny(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), self.canny_low, self.canny_high)
        mask = cv2.bitwise_or(mask, edges)

        nz = int(np.count_nonzero(mask))
        if nz < self.min_mask_px:
            self._fallback(mask, W, H, y0, "NO LANE (area)", dt)
            return

        # 3) Multi-row scan for lane center
        centers, used_y = [], []
        ys = mask.shape[0]
        for r in self.scan_rows:
            y_scan = int(ys * float(r))
            y_scan = np.clip(y_scan, 0, ys-1)
            scan = mask[y_scan, :]
            xs = np.where(scan > 0)[0]
            if xs.size >= self.min_lane_w:
                left_idx, right_idx = int(xs[0]), int(xs[-1])
                if (right_idx - left_idx) >= self.min_lane_w:
                    centers.append(0.5 * (left_idx + right_idx))
                    used_y.append(y_scan)

        if len(centers) < self.min_valid_rows:
            self._fallback(mask, W, H, y0, "NO LANE (rows)", dt)
            return

        # Prefer lower half rows (closer to vehicle) to reduce jitter
        take = max(2, len(centers)//2)
        lane_center_px = float(np.mean(centers[-take:]))

        img_center_px = 0.5 * W
        err_norm = (img_center_px - lane_center_px) / (W * 0.5)  # normalized lateral error, right = negative

        # Optional heading from linear fit of (y, cx)
        heading = 0.0
        if len(centers) >= 2:
            z = np.polyfit(np.array(used_y, dtype=np.float32),
                           np.array(centers, dtype=np.float32), 1)
            slope = float(z[0])             # px per row
            heading = clamp(slope / (W * 0.5), -0.6, 0.6)  # normalized tilt

        # 4) Error LPF + deadband
        if abs(err_norm) < self.err_deadband:
            err_n_db = 0.0
        else:
            err_n_db = err_norm

        alpha_e = exp(-dt / max(1e-3, self.err_lpf_tau))
        self.e_filt = alpha_e * self.e_filt + (1.0 - alpha_e) * err_n_db
        d_err = (self.e_filt - getattr(self, "_e_filt_prev", self.e_filt)) / dt
        self._e_filt_prev = self.e_filt

        # 5) PID (mostly PD) + heading
        self.e_int += self.e_filt * dt
        # anti-windup clamp
        i_max = 0.5 * self.steer_limit / max(1e-6, self.ki) if self.ki > 0 else 0.0
        if self.ki > 0:
            self.e_int = clamp(self.e_int, -i_max, i_max)
        else:
            self.e_int = 0.0

        steer_cmd = self.kp * self.e_filt + self.kd * d_err + self.k_heading * heading + self.ki * self.e_int
        if self.invert_steer:
            steer_cmd = -steer_cmd

        # 6) Steering LPF + slew-rate limit
        alpha_s = exp(-dt / max(1e-3, self.steer_lpf_tau))
        self.steer_filt = alpha_s * self.steer_filt + (1.0 - alpha_s) * steer_cmd

        # slew rate
        max_delta = self.steer_rate_limit * dt
        desired = clamp(self.steer_filt, -self.steer_limit, self.steer_limit)
        current = getattr(self, "_steer_out_prev", 0.0)
        delta = clamp(desired - current, -max_delta, max_delta)
        steer_out = clamp(current + delta, -self.steer_limit, self.steer_limit)
        self._steer_out_prev = steer_out

        # 7) Speed scheduling vs steering
        steer_ratio = abs(steer_out) / max(1e-6, self.steer_limit)
        speed = self.target_speed * (1.0 - self.slowdown_gain * (steer_ratio ** self.slowdown_pow))
        speed = clamp(speed, self.min_speed, self.target_speed)

        # 8) Safety stop
        if self.estop:
            speed = 0.0
            steer_out = 0.0
            self.e_int = 0.0

        # 9) Publish
        self.pub_err.publish(float(self.e_filt))
        self._publish_cmd(speed, steer_out)
        self.last_ok_time = rospy.Time.now()

        # 10) Debug image
        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for y_scan, cx in zip(used_y, centers):
            cv2.line(dbg, (int(cx), y_scan), (int(cx), max(0, y_scan-30)), (0,0,255), 2)
        cv2.line(dbg, (int(img_center_px), ys-5), (int(img_center_px), ys-50), (255,0,0), 1)
        cv2.line(dbg, (int(lane_center_px), ys-5), (int(lane_center_px), ys-50), (0,255,0), 2)
        txt = f"e={self.e_filt:+.2f} de={d_err:+.2f} hd={heading:+.2f} st={steer_out:+.2f} v={speed:.2f} nz={nz}"
        cv2.putText(dbg, txt, (10, dbg.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
        self._publish_debug(dbg, W, H, y0)

    # -------------------- helpers --------------------

    def _fallback(self, mask, W, H, y0, label, dt):
        """When lane lost: hold last command briefly, then gentle stop."""
        hold = (rospy.Time.now() - self.last_ok_time).to_sec() * 1000.0 < self.hold_bad_ms
        if hold and not self.estop:
            # keep last command but also respect slew-rate next frames
            self._publish_cmd(self.last_cmd.speed, self.last_cmd.steering_angle)
        else:
            self.e_int = 0.0
            self._publish_cmd(0.0, 0.0)

        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(dbg, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)
        self._publish_debug(dbg, W, H, y0)

    def _publish_cmd(self, speed, steer):
        cmd = AckermannDrive()
        cmd.speed = float(speed)
        cmd.steering_angle = float(steer)
        self.pub_cmd.publish(cmd)
        self.last_cmd = cmd

    def _publish_debug(self, roi_bgr, W, H, y0):
        canv = np.zeros((H, W, 3), dtype=np.uint8)
        canv[y0:H, :] = cv2.resize(roi_bgr, (W, H - y0))
        try:
            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(canv, encoding='bgr8'))
        except Exception as e:
            rospy.logwarn_throttle(2.0, "vision_lka: debug publish failed: %s", e)

if __name__ == "__main__":
    VisionLKA()
    rospy.spin()

