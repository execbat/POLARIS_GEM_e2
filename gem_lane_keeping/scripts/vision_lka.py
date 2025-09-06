#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust lane keeping for GEM:
- Combines color + edges to a binary mask.
- Uses histogram + sliding windows to track left/right lanes.
- Mirrors the missing side if only one lane is visible.
- Smooths steering and speed; slows down on tight turns.
ROS IO:
  sub  image: sensor_msgs/Image         (remap to /gem/front_single_camera/image_raw)
  sub  /gem/safety/stop: std_msgs/Bool
  pub  cmd:   ackermann_msgs/AckermannDrive
  pub  debug: sensor_msgs/Image         (overlay with windows/fit)
  pub  lateral_error: std_msgs/Float32  (normalized [-1..1])
"""

import rospy, cv2, numpy as np, time
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge

def clamp(x, lo, hi): return lo if x < lo else (hi if x > hi else x)

class VisionLKA:
    def __init__(self):
        rospy.init_node("vision_lka")
        self.bridge = CvBridge()

        # --- Geometry / ROI ---
        self.roi_y_top     = rospy.get_param("~roi_y_top", 0.58)   # top of ROI as fraction of height
        self.scan_margin   = rospy.get_param("~scan_margin", 2)    # don't crop borders hard
        self.min_mask_px   = rospy.get_param("~min_mask_px", 120)  # minimal nonzero mask area in ROI
        self.min_pts_fit   = rospy.get_param("~min_pts_fit", 200)  # minimal points to accept a polynomial fit

        # --- Binarization (color + edges) ---
        self.white_s_max   = rospy.get_param("~white_s_max", 120)
        self.white_v_min   = rospy.get_param("~white_v_min", 120)
        self.hls_L_min     = rospy.get_param("~hls_L_min", 170)
        self.lab_b_min     = rospy.get_param("~lab_b_min", 140)    # yellows
        self.canny_low     = rospy.get_param("~canny_low", 40)
        self.canny_high    = rospy.get_param("~canny_high", 120)
        self.morph_kernel  = rospy.get_param("~morph_kernel", 7)   # odd

        # --- Sliding windows ---
        self.n_windows     = rospy.get_param("~n_windows", 9)
        self.win_margin    = rospy.get_param("~win_margin", 60)
        self.win_minpix    = rospy.get_param("~win_minpix", 40)
        self.nom_lane_w_px = rospy.get_param("~nom_lane_w_px", 0.45)  # fraction of width used as nominal lane width
        self.min_lane_w_px = rospy.get_param("~min_lane_width_px", 20)

        # --- Control ---
        self.base_speed    = rospy.get_param("~target_speed", 1.2)
        self.kp            = rospy.get_param("~kp", 0.018)
        self.kd            = rospy.get_param("~kd", 0.06)
        self.k_heading     = rospy.get_param("~k_heading", 0.007)
        self.steer_limit   = rospy.get_param("~steer_limit", 0.40)
        self.max_accel     = rospy.get_param("~max_accel", 0.3)     # m/s^2
        self.max_decel     = rospy.get_param("~max_decel", 1.0)     # m/s^2
        self.speed_turn_k  = rospy.get_param("~speed_turn_k", 0.85) # slowdown factor per |steer|
        self.steer_alpha   = rospy.get_param("~steer_alpha", 0.25)  # low-pass for steering (0..1)
        self.hold_bad_ms   = rospy.get_param("~hold_bad_ms", 400)

        # --- State ---
        self.estop         = False
        self.prev_err      = 0.0
        self.prev_t        = rospy.get_time()
        self.last_ok_time  = rospy.Time(0.0)
        self.last_speed    = 0.0
        self.steer_filt    = 0.0
        self.prev_lane_w   = None  # px

        # --- ROS IO ---
        self.pub_cmd = rospy.Publisher("cmd", AckermannDrive, queue_size=10)
        self.pub_err = rospy.Publisher("lateral_error", Float32, queue_size=10)
        self.pub_dbg = rospy.Publisher("debug", Image, queue_size=1)
        rospy.Subscriber("image", Image, self.on_image, queue_size=1, buff_size=2**24)
        rospy.Subscriber("/gem/safety/stop", Bool, self.on_stop, queue_size=1)

        rospy.loginfo("vision_lka ready: base_speed=%.2f ROI_top=%.2f", self.base_speed, self.roi_y_top)

    # ----------------------- ROS callbacks -----------------------

    def on_stop(self, msg: Bool):
        self.estop = bool(msg.data)

    def on_image(self, msg: Image):
        # Convert to BGR and cut ROI
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "cv_bridge: %s", e); return
        H, W = img.shape[:2]
        y0 = int(H * self.roi_y_top); y0 = min(max(0, y0), H-2)
        roi = img[y0:H, :].copy()
        h = roi.shape[0]

        # ---- 1) Build robust binary mask (color + edges) ----
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        white_hsv = cv2.inRange(hsv, (0, 0, self.white_v_min), (179, self.white_s_max, 255))
        white_hls = cv2.inRange(hls[:, :, 1], self.hls_L_min, 255)
        yellow    = cv2.inRange(lab[:, :, 2], self.lab_b_min, 255)

        # Edges but only on bright pixels (suppresses grass)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), self.canny_low, self.canny_high)
        edges = cv2.bitwise_and(edges, cv2.inRange(gray, 180, 255))

        mask = white_hsv | white_hls | yellow | edges

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morph_kernel, self.morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
        mask = cv2.dilate(mask, k, iterations=1)

        # keep borders (do NOT trim right edge hard)
        mask[:self.scan_margin, :] = 0

        nz = int(np.count_nonzero(mask))
        if nz < self.min_mask_px:
            self._stop_with_debug(roi, mask, W, H, y0, "NO LANE (area)")
            return

        # ---- 2) Sliding windows on mask ----
        # Histogram over lower half of ROI
        hist = np.sum(mask[h//2:, :], axis=0)
        midpoint = W // 2
        # we look for strong peaks left/right of the midpoint to avoid dashed centerline
        left_base  = np.argmax(hist[:max(10, midpoint-50)])
        right_base = np.argmax(hist[min(W-1, midpoint+50):]) + midpoint + 50
        right_base = min(right_base, W-1)

        # parameters
        win_h = h // self.n_windows
        nonzero_y, nonzero_x = np.nonzero(mask)
        left_x, left_y, right_x, right_y = [], [], [], []
        lx_current, rx_current = left_base, right_base

        for w_i in range(self.n_windows):
            win_y_low  = h - (w_i + 1) * win_h
            win_y_high = h - w_i * win_h
            win_xleft_low  = max(0, lx_current - self.win_margin)
            win_xleft_high = min(W, lx_current + self.win_margin)
            win_xright_low  = max(0, rx_current - self.win_margin)
            win_xright_high = min(W, rx_current + self.win_margin)

            # indices of nonzero points inside windows
            good_left_inds = np.where(
                (nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)
            )[0]
            good_right_inds = np.where(
                (nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)
            )[0]

            left_x.extend(nonzero_x[good_left_inds]); left_y.extend(nonzero_y[good_left_inds])
            right_x.extend(nonzero_x[good_right_inds]); right_y.extend(nonzero_y[good_right_inds])

            if len(good_left_inds) > self.win_minpix:
                lx_current = int(np.mean(nonzero_x[good_left_inds]))
            if len(good_right_inds) > self.win_minpix:
                rx_current = int(np.mean(nonzero_x[good_right_inds]))

        left_x = np.array(left_x); left_y = np.array(left_y)
        right_x = np.array(right_x); right_y = np.array(right_y)

        have_left  = left_x.size  >= self.min_pts_fit
        have_right = right_x.size >= self.min_pts_fit

        # If both sides found: fit 2nd order polynomials in ROI coords
        px_y = np.arange(h, dtype=np.float32)

        left_fit, right_fit = None, None
        if have_left:
            left_fit = np.polyfit(left_y.astype(np.float32), left_x.astype(np.float32), 2)
        if have_right:
            right_fit = np.polyfit(right_y.astype(np.float32), right_x.astype(np.float32), 2)

        # Estimate lane center at bottom of ROI
        y_eval = float(h - 1)
        centers = []
        lane_w_px = None
        if left_fit is not None:
            lx_bot = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
            centers.append(lx_bot)
        if right_fit is not None:
            rx_bot = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
            centers.append(rx_bot)

        if have_left and have_right:
            lane_w_px = float(abs(rx_bot - lx_bot))
            self.prev_lane_w = lane_w_px
            lane_center_px = 0.5*(lx_bot + rx_bot)
        elif have_left or have_right:
            # Mirror the visible side using nominal/previous lane width
            lane_w_guess = self.prev_lane_w if self.prev_lane_w is not None else (self.nom_lane_w_px * W)
            lane_w_guess = max(lane_w_guess, self.min_lane_w_px)

            if have_left:
                rx_bot = lx_bot + lane_w_guess
                lane_center_px = 0.5*(lx_bot + rx_bot)
            else:
                lx_bot = rx_bot - lane_w_guess
                lane_center_px = 0.5*(lx_bot + rx_bot)
        else:
            # No lanes -> hold or stop
            if (rospy.Time.now() - self.last_ok_time).to_sec()*1000.0 < self.hold_bad_ms:
                self._publish_cmd(self.last_speed, self.steer_filt)
                self._publish_debug(img, roi, mask, None, None, W, H, y0, text="HOLD (no lanes)")
            else:
                self._stop_with_debug(roi, mask, W, H, y0, "NO LANE (rows)")
            return

        # Normalize lateral error relative to image center
        img_center_px = 0.5*W
        err_px = (img_center_px - lane_center_px)
        err = float(err_px) / (0.5*W)  # ~[-1..1]

        # Heading proxy: center slope at bottom from polynomial(s)
        heading = 0.0
        slopes = []
        if left_fit is not None:
            slopes.append(2*left_fit[0]*y_eval + left_fit[1])
        if right_fit is not None:
            slopes.append(2*right_fit[0]*y_eval + right_fit[1])
        if slopes:
            heading = -float(np.mean(slopes)) / (0.5*W)  # normalize and sign so that >0 => steer right

        # PD + heading, then smoothing
        t = rospy.get_time(); dt = max(1e-3, t - self.prev_t)
        d_err = (err - self.prev_err) / dt
        self.prev_err, self.prev_t = err, t

        steer_cmd = self.kp*err + self.kd*d_err + self.k_heading*heading
        steer_cmd = clamp(steer_cmd, -self.steer_limit, self.steer_limit)
        # low-pass the steering to prevent oscillations
        self.steer_filt = (1.0 - self.steer_alpha)*self.steer_filt + self.steer_alpha*steer_cmd

        # Speed: slow down with steering magnitude; ramp with accel limits
        target_v = self.base_speed * (1.0 - self.speed_turn_k*min(1.0, abs(self.steer_filt)/self.steer_limit))
        dv = target_v - self.last_speed
        if dv > 0:
            dv = min(dv, self.max_accel * dt)
        else:
            dv = max(dv, -self.max_decel * dt)
        v_cmd = max(0.0, self.last_speed + dv)

        # Safety
        if self.estop:
            v_cmd = 0.0
            self.steer_filt = 0.0

        self._publish_cmd(v_cmd, self.steer_filt)
        self.last_speed = v_cmd
        self.last_ok_time = rospy.Time.now()

        # ---- Debug image ----
        self._publish_debug(img, roi, mask, left_fit, right_fit, W, H, y0,
                            text=f"err={err:+.2f} de={d_err:+.2f} hd={heading:+.2f} st={self.steer_filt:+.2f} v={v_cmd:.2f}")

    # ----------------------- helpers -----------------------

    def _publish_cmd(self, speed, steer):
        cmd = AckermannDrive()
        cmd.speed = float(speed)
        cmd.steering_angle = float(steer)
        self.pub_cmd.publish(cmd)
        self.pub_err.publish(float(steer))  # publish something useful if you want; or err instead

    def _stop_with_debug(self, roi, mask, W, H, y0, label):
        self._publish_cmd(0.0, 0.0)
        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(dbg, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2, cv2.LINE_AA)
        canv = np.zeros((H, W, 3), dtype=np.uint8); canv[y0:H, :] = cv2.resize(dbg, (W, H-y0))
        self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(canv, encoding='bgr8'))

    def _publish_debug(self, img, roi, mask, left_fit, right_fit, W, H, y0, text=""):
        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        h = roi.shape[0]
        yy = np.arange(h, dtype=np.float32)
        if left_fit is not None:
            lx = (left_fit[0]*yy**2 + left_fit[1]*yy + left_fit[2]).astype(np.int32)
            for i in range(0, h, max(1, h//20)):
                cv2.circle(dbg, (int(clamp(lx[i], 0, W-1)), i), 2, (0,0,255), -1)
        if right_fit is not None:
            rx = (right_fit[0]*yy**2 + right_fit[1]*yy + right_fit[2]).astype(np.int32)
            for i in range(0, h, max(1, h//20)):
                cv2.circle(dbg, (int(clamp(rx[i], 0, W-1)), i), 2, (255,0,0), -1)

        cv2.putText(dbg, text, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        canv = np.zeros((H, W, 3), dtype=np.uint8); canv[y0:H, :] = cv2.resize(dbg, (W, H-y0))
        self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(canv, encoding='bgr8'))


if __name__ == "__main__":
    VisionLKA()
    rospy.spin()

