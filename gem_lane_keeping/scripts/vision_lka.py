#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lane keeping using Canny + HoughLinesP.

Inputs:
  - sensor_msgs/Image on topic "image" (remap to /gem/front_single_camera/image_raw)
Outputs:
  - ackermann_msgs/AckermannDrive on topic "cmd"
  - sensor_msgs/Image on topic "debug" (visualization)
  - std_msgs/Float32 on "lateral_error" (normalized lateral error)

Tuning knobs are exposed as ROS params (see below).
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


class VisionLKA_Hough:
    def __init__(self):
        rospy.init_node("vision_lka")

        # ---------------- I/O ----------------
        self.bridge = CvBridge()
        self.pub_cmd = rospy.Publisher("cmd", AckermannDrive, queue_size=10)
        self.pub_err = rospy.Publisher("lateral_error", Float32, queue_size=10)
        self.pub_dbg = rospy.Publisher("debug", Image, queue_size=1)
        rospy.Subscriber("image", Image, self.on_image, queue_size=1)
        rospy.Subscriber("/gem/safety/stop", Bool, self.on_stop, queue_size=1)

        # ---------------- Control params ----------------
        self.target_speed   = rospy.get_param("~target_speed", 1.5)
        self.min_speed      = rospy.get_param("~min_speed",   0.6)
        self.steer_limit    = rospy.get_param("~steer_limit", 0.55)
        self.kp             = rospy.get_param("~kp", 0.90)       # lateral P (center error)
        self.kd             = rospy.get_param("~kd", 0.15)       # lateral D
        self.k_heading      = rospy.get_param("~k_heading", 0.35)# heading gain (from line slope/angle)
        self.deadband       = rospy.get_param("~deadband", 0.00)
        self.steer_slowdown = rospy.get_param("~steer_slowdown", 0.75)  # how much to slow vs |steer|
        self.invert_steer   = rospy.get_param("~invert_steer", False)

        # ---------------- ROI (trapezoid) ----------------
        # Fraction of image height where ROI starts and ends (0..1)
        self.roi_y_top  = rospy.get_param("~roi_y_top", 0.60)    # start of ROI (from top)
        self.roi_y_bot  = rospy.get_param("~roi_y_bot", 1.00)    # end of ROI (usually 1.0 = bottom)
        # Trapezoid top width as fraction of image width (0..1), bottom width = 1.0
        self.roi_top_width_frac = rospy.get_param("~roi_top_width_frac", 0.30)

        # ---------------- Canny/Hough ----------------
        self.blur_ksize = rospy.get_param("~blur_ksize", 5)  # must be odd
        self.canny_low  = rospy.get_param("~canny_low", 60)
        self.canny_high = rospy.get_param("~canny_high", 150)

        self.hough_rho         = rospy.get_param("~hough_rho", 1.0)
        self.hough_theta_deg   = rospy.get_param("~hough_theta_deg", 1.0)
        self.hough_thresh      = rospy.get_param("~hough_thresh", 30)
        self.hough_min_len     = rospy.get_param("~hough_min_len", 25)
        self.hough_max_gap     = rospy.get_param("~hough_max_gap", 20)

        # Accept only sufficiently slanted lines (|slope| >= slope_min)
        self.slope_min         = rospy.get_param("~slope_min", 0.35)

        # If only one side is visible, assume lane width in pixels (scaled by image width)
        self.lane_width_frac   = rospy.get_param("~lane_width_frac", 0.45)  # proportion of image width
        self.y_eval_frac       = rospy.get_param("~y_eval_frac", 0.92)      # y (in ROI) where we compute x_left/x_right

        # Confidence / gating
        self.min_segments      = rospy.get_param("~min_segments", 2)  # min total Hough segments to consider valid
        self.hold_bad_ms       = rospy.get_param("~hold_bad_ms", 500) # hold last command before stopping
        self.min_valid_sides   = rospy.get_param("~min_valid_sides", 1)  # allow 1 side only

        # ---------------- State ----------------
        self.estop      = False
        self.prev_err   = 0.0
        self.prev_t     = rospy.get_time()
        self.last_ok_t  = rospy.Time(0.0)
        self.last_cmd   = AckermannDrive()
        self.last_cmd.speed = self.min_speed
        self.last_cmd.steering_angle = 0.0

        rospy.loginfo(
            "VisionLKA Hough: v=%.2f..%.2f steer_lim=%.2f kp=%.2f kd=%.2f kh=%.2f "
            "roi_y=[%.2f..%.2f], top_w=%.2f canny(%d,%d) blur=%d "
            "hough(rho=%.1f, theta=%ddeg, thr=%d, minLen=%d, maxGap=%d) slope_min=%.2f "
            "lane_w=%.2f y_eval=%.2f min_seg=%d hold=%dms",
            self.min_speed, self.target_speed, self.steer_limit, self.kp, self.kd, self.k_heading,
            self.roi_y_top, self.roi_y_bot, self.roi_top_width_frac,
            self.canny_low, self.canny_high, self.blur_ksize,
            self.hough_rho, int(self.hough_theta_deg), self.hough_thresh, self.hough_min_len, self.hough_max_gap,
            self.slope_min, self.lane_width_frac, self.y_eval_frac, self.min_segments, self.hold_bad_ms
        )

    # --------------- Callbacks ---------------
    def on_stop(self, msg: Bool):
        self.estop = bool(msg.data)

    def on_image(self, img_msg: Image):
        # Convert to BGR
        try:
            bgr = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "cv_bridge error: %s", e)
            return

        H, W = bgr.shape[:2]

        # ----- Build trapezoid ROI mask -----
        y_top = int(clamp(self.roi_y_top * H, 0, H - 2))
        y_bot = int(clamp(self.roi_y_bot * H, y_top + 2, H))
        roi_h = y_bot - y_top
        if roi_h < 4:
            return  # degenerate

        # Polygon: trapezoid vertices (clockwise)
        top_w = int(self.roi_top_width_frac * W)
        x_top_l = (W - top_w) // 2
        x_top_r = x_top_l + top_w

        roi_poly = np.array([
            [0,     y_bot],
            [W,     y_bot],
            [x_top_r, y_top],
            [x_top_l, y_top]
        ], dtype=np.int32)

        mask_full = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(mask_full, roi_poly, 255)

        # ----- Preprocess edges (Canny) -----
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self.blur_ksize >= 3 and self.blur_ksize % 2 == 1:
            gray = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)

        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        edges_roi = cv2.bitwise_and(edges, mask_full)
        roi = edges_roi[y_top:y_bot, :]

        # ----- HoughLinesP -----
        lines = cv2.HoughLinesP(
            roi,
            rho=self.hough_rho,
            theta=np.deg2rad(self.hough_theta_deg),
            threshold=self.hough_thresh,
            minLineLength=self.hough_min_len,
            maxLineGap=self.hough_max_gap
        )

        dbg = bgr.copy()
        # Draw ROI polygon
        cv2.polylines(dbg, [roi_poly], isClosed=True, color=(255, 0, 0), thickness=1)

        left_points  = []  # list of (m, b, weight)
        right_points = []  # list of (m, b, weight)
        total_segments = 0

        if lines is not None:
            total_segments = len(lines)
            for ln in lines:
                x1, y1, x2, y2 = ln[0]
                # Convert from ROI coords to full image coords (for drawing)
                y1f, y2f = y1 + y_top, y2 + y_top

                dx = float(x2 - x1)
                dy = float(y2 - y1)
                if abs(dx) < 1e-3:
                    continue  # vertical degenerate
                m = dy / dx  # NOTE: y grows downward
                if abs(m) < self.slope_min:
                    continue  # too horizontal

                b = (y1 - m * x1)  # intercept in ROI coordinates
                length = np.hypot(dx, dy)
                weight = max(5.0, length)  # longer lines weigh more

                if m < 0:
                    left_points.append((m, b, weight))
                    color = (0, 255, 0)
                else:
                    right_points.append((m, b, weight))
                    color = (0, 200, 200)

                # draw the segment on debug
                cv2.line(dbg, (x1, y1f), (x2, y2f), color, 2)

        # Evaluate lane borders at a fixed y inside ROI
        y_eval = int(clamp(self.y_eval_frac * roi_h, 0, roi_h - 1))
        x_left, x_right, have_left, have_right = None, None, False, False

        def weighted_x_at_y(eval_y, items):
            """items: list of (m,b,w) in ROI coords; return weighted average x at y=eval_y"""
            xs = []
            ws = []
            for m, b, w in items:
                if abs(m) < 1e-6:
                    continue
                x = (eval_y - b) / m
                xs.append(x)
                ws.append(w)
            if not xs:
                return None
            xs = np.array(xs)
            ws = np.array(ws)
            return float(np.average(xs, weights=ws))

        if left_points:
            x_left = weighted_x_at_y(y_eval, left_points)
            if x_left is not None:
                have_left = True
        if right_points:
            x_right = weighted_x_at_y(y_eval, right_points)
            if x_right is not None:
                have_right = True

        # If not enough segments overall OR neither side found -> hold or stop
        if total_segments < self.min_segments or (not have_left and not have_right):
            self._hold_or_stop()
            self._publish_debug(dbg)
            rospy.loginfo_throttle(1.0, "[hough] segments=%d, left=%s right=%s -> hold/stop",
                                   total_segments, str(have_left), str(have_right))
            return

        # Infer missing side using lane_width
        lane_w_px = self.lane_width_frac * float(W)
        if have_left and not have_right:
            x_right = x_left + lane_w_px
        elif have_right and not have_left:
            x_left = x_right - lane_w_px

        # Bound to image
        x_left  = clamp(float(x_left),  0.0, float(W - 1))
        x_right = clamp(float(x_right), 0.0, float(W - 1))

        # Lane center at eval_y
        lane_center = 0.5 * (x_left + x_right)
        img_center  = 0.5 * W
        # Lateral error: >0 => lane center is to the left of image center -> steer right (depending on sign convention)
        err = (img_center - lane_center) / (0.5 * W)

        # Heading error: use average slope sign (convert to small angle in radians)
        # If both sides present, estimate centerline slope from left/right slopes; else use the available side.
        def avg_slope(items):
            if not items:
                return None
            ms = np.array([m for (m, _, _) in items], dtype=np.float32)
            ws = np.array([w for (_, _, w) in items], dtype=np.float32)
            return float(np.average(ms, weights=ws))

        mL = avg_slope(left_points)
        mR = avg_slope(right_points)
        if mL is not None and mR is not None:
            m_center = 0.5 * (mL + mR)
        elif mL is not None:
            m_center = mL
        elif mR is not None:
            m_center = mR
        else:
            m_center = 0.0

        # Convert slope to heading error. y increases downward; small-angle approx:
        heading = -np.arctan(m_center)  # negative to make "right-leaning" => positive correction
        # PD + heading
        t = rospy.get_time()
        dt = max(1e-3, t - self.prev_t)
        d_err = (err - self.prev_err) / dt
        self.prev_err, self.prev_t = err, t

        steer = self.kp * err + self.kd * d_err + self.k_heading * float(heading)
        if self.invert_steer:
            steer = -steer
        steer = clamp(steer, -self.steer_limit, self.steer_limit)

        # Speed governor
        steer_ratio = abs(steer) / max(1e-6, self.steer_limit)
        speed = self.target_speed * (1.0 - self.steer_slowdown * steer_ratio)
        speed = max(self.min_speed, speed)

        if self.estop:
            speed, steer = 0.0, 0.0

        # Publish
        cmd = AckermannDrive()
        cmd.speed = float(speed)
        cmd.steering_angle = float(steer)
        self.pub_cmd.publish(cmd)
        self.pub_err.publish(float(err))
        self.last_cmd = cmd
        self.last_ok_t = rospy.Time.now()

        # Debug overlays
        # draw y_eval line within ROI
        y_eval_full = y_top + y_eval
        cv2.line(dbg, (0, y_eval_full), (W - 1, y_eval_full), (200, 200, 200), 1)
        # draw left/right/center points
        cv2.circle(dbg, (int(x_left),  y_eval_full), 4, (0, 255, 0), -1)
        cv2.circle(dbg, (int(x_right), y_eval_full), 4, (0, 255, 0), -1)
        cv2.circle(dbg, (int(lane_center), y_eval_full), 4, (0, 0, 255), -1)
        cv2.line(dbg, (int(W/2), y_eval_full-20), (int(W/2), y_eval_full+20), (255, 0, 0), 1)

        txt = f"seg={total_segments} L={have_left} R={have_right} err={err:+.2f} de={d_err:+.2f} head={float(heading):+.2f} st={steer:+.2f} v={speed:.2f}"
        cv2.putText(dbg, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        self._publish_debug(dbg)

        rospy.loginfo_throttle(1.0, "[hough] " + txt)

    # --------------- Helpers ---------------
    def _hold_or_stop(self):
        # Hold last non-zero command for a short period, then stop
        if (rospy.Time.now() - self.last_ok_t).to_sec() * 1000.0 < self.hold_bad_ms and not self.estop:
            cmd = self.last_cmd
            if cmd.speed <= 1e-3:
                cmd = AckermannDrive(speed=self.min_speed, steering_angle=0.0)
            self.pub_cmd.publish(cmd)
        else:
            self.pub_cmd.publish(AckermannDrive(speed=0.0, steering_angle=0.0))

    def _publish_debug(self, bgr):
        try:
            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8'))
        except Exception as e:
            rospy.logwarn_throttle(2.0, "debug publish failed: %s", e)


if __name__ == "__main__":
    VisionLKA_Hough()
    rospy.spin()

