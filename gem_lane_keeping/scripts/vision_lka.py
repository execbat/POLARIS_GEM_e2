#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision-based lane keeping for Gazebo GEM car (ROS1, Noetic).

Key ideas:
- Trapezoid ROI anchors detection to the road area in front of the car.
- Color pre-mask (white/yellow) + Canny edges -> clean lane mask for Gazebo textures.
- Multi-row scanning finds lane centers; weighted average prefers lower rows (closer).
- Heading (slope) from a linear fit through those centers.
- PD on lateral error + heading term => steering.
- Green-balance fallback: if no clear lines, balance green areas left/right to keep center.
- Soft hold of last command for brief dropouts. Obeys /gem/safety/stop.

Topics (relative):
  Sub:  "image"  (sensor_msgs/Image)  -> remap to /gem/front_single_camera/image_raw
        "/gem/safety/stop" (std_msgs/Bool)
  Pub:  "cmd"    (ackermann_msgs/AckermannDrive)
        "lateral_error" (std_msgs/Float32)
        "debug"  (sensor_msgs/Image)
"""
import math
import numpy as np
import cv2
import rospy
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

        # ======= CONTROL/GATING PARAMS =======
        self.target_speed   = rospy.get_param("~target_speed", 1.5)
        self.min_speed      = rospy.get_param("~min_speed",    0.7)
        self.kp             = rospy.get_param("~kp",           0.70)
        self.kd             = rospy.get_param("~kd",           0.10)
        self.k_heading      = rospy.get_param("~k_heading",    0.45)
        self.steer_limit    = rospy.get_param("~steer_limit",  0.60)
        self.steer_slowdown = rospy.get_param("~steer_slowdown", 0.65)
        self.invert_steer   = rospy.get_param("~invert_steer", False)

        # ======= ROI (TRAPEZOID) =======
        # Top/bottom in fraction of image height (0..1), top width fraction (0..1)
        self.roi_y_top          = rospy.get_param("~roi_y_top", 0.52)
        self.roi_y_bot          = rospy.get_param("~roi_y_bot", 1.00)
        self.roi_top_width_frac = rospy.get_param("~roi_top_width_frac", 0.65)

        # ======= COLOR MASK (WHITE/YELLOW) + CANNY =======
        self.blur_ksize  = rospy.get_param("~blur_ksize", 5)     # odd
        self.canny_low   = rospy.get_param("~canny_low",  40)
        self.canny_high  = rospy.get_param("~canny_high", 120)
        # HSV white/yellow thresholds (tuned for Gazebo road/lines)
        self.white_v_min = rospy.get_param("~white_v_min", 180)  # V >=
        self.white_s_max = rospy.get_param("~white_s_max", 60)   # S <=
        self.yellow_h_lo = rospy.get_param("~yellow_h_lo", 15)
        self.yellow_h_hi = rospy.get_param("~yellow_h_hi", 40)
        self.yellow_s_min= rospy.get_param("~yellow_s_min", 80)
        self.yellow_v_min= rospy.get_param("~yellow_v_min", 80)

        # ======= SCAN GEOMETRY =======
        # Row positions are relative to ROI height
        self.scan_rows       = rospy.get_param("~scan_rows", [0.60, 0.72, 0.84, 0.92, 0.97])
        self.min_valid_rows  = rospy.get_param("~min_valid_rows", 2)
        self.min_lane_w_px   = rospy.get_param("~min_lane_width_px", 22)
        self.min_mask_area   = rospy.get_param("~min_mask_area", 150)
        self.y_eval_frac     = rospy.get_param("~y_eval_frac", 0.95)   # bottom fraction for center eval
        # Estimated lane width as fraction of image width (used when only one edge visible)
        self.lane_width_frac = rospy.get_param("~lane_width_frac", 0.52)

        # ======= GREEN-BALANCE FALLBACK =======
        self.use_green_balance = rospy.get_param("~use_green_balance", True)
        self.green_h_lo   = rospy.get_param("~green_h_lo", 30)
        self.green_h_hi   = rospy.get_param("~green_h_hi", 95)
        self.green_s_min  = rospy.get_param("~green_s_min", 60)
        self.green_v_min  = rospy.get_param("~green_v_min", 20)
        self.min_green_cols= rospy.get_param("~min_green_cols", 200)  # min green pixels in ROI to trust fallback

        # ======= DROP-OUT HANDLING =======
        self.hold_bad_ms  = rospy.get_param("~hold_bad_ms", 350)  # hold last cmd for brief blank frames

        # ======= DEBUG =======
        self.debug_overlay = rospy.get_param("~debug_overlay", True)

        # ======= STATE =======
        self.prev_err     = 0.0
        self.prev_t       = rospy.get_time()
        self.last_ok_time = rospy.Time(0.0)
        self.last_center  = None  # px
        self.last_cmd     = AckermannDrive()
        self.estop        = False

        # ======= IO =======
        self.pub_cmd = rospy.Publisher("cmd", AckermannDrive, queue_size=10)
        self.pub_err = rospy.Publisher("lateral_error", Float32, queue_size=10)
        self.pub_dbg = rospy.Publisher("debug", Image, queue_size=1)
        rospy.Subscriber("image", Image, self.on_image, queue_size=1)
        rospy.Subscriber("/gem/safety/stop", Bool, self.on_stop, queue_size=1)

        rospy.loginfo("vision_lka: ROI[y=%.2f..%.2f, top_w=%.2f] Canny[%d,%d] "
                      "white(V>=%d,S<=%d) yellow(H=%d..%d,S>=%d,V>=%d) scans=%s "
                      "lane_w_frac=%.2f kp=%.2f kd=%.2f kh=%.2f steer_lim=%.2f inv=%s",
                      self.roi_y_top, self.roi_y_bot, self.roi_top_width_frac,
                      self.canny_low, self.canny_high,
                      self.white_v_min, self.white_s_max,
                      self.yellow_h_lo, self.yellow_h_hi, self.yellow_s_min, self.yellow_v_min,
                      str(self.scan_rows), self.lane_width_frac,
                      self.kp, self.kd, self.k_heading, self.steer_limit, self.invert_steer)

    # ---------- callbacks ----------
    def on_stop(self, msg: Bool):
        self.estop = bool(msg.data)

    def publish_cmd(self, speed, steer):
        if self.estop:
            speed = 0.0
            steer = 0.0
        if self.invert_steer:
            steer = -steer
        cmd = AckermannDrive()
        cmd.speed = float(max(0.0, speed))
        cmd.steering_angle = float(clamp(steer, -self.steer_limit, self.steer_limit))
        self.pub_cmd.publish(cmd)
        self.last_cmd = cmd

    def on_image(self, msg: Image):
        # 0) Convert
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "cv_bridge: %s", e)
            return

        H, W = bgr.shape[:2]
        # 1) ROI trapezoid
        y_top = int(H * self.roi_y_top)
        y_bot = int(H * self.roi_y_bot)
        top_w = int(W * self.roi_top_width_frac)
        x_left_top  = (W - top_w) // 2
        x_right_top = x_left_top + top_w
        roi_poly = np.array([[0, y_bot],
                             [W, y_bot],
                             [x_right_top, y_top],
                             [x_left_top,  y_top]], dtype=np.int32)
        roi_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(roi_mask, roi_poly, 255)

        # 2) Color pre-mask (white+yellow)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv,
                                 (0, 0, self.white_v_min),
                                 (179, self.white_s_max, 255))
        yellow_mask = cv2.inRange(hsv,
                                  (self.yellow_h_lo, self.yellow_s_min, self.yellow_v_min),
                                  (self.yellow_h_hi, 255, 255))
        color_mask = cv2.bitwise_or(white_mask, yellow_mask)
        color_mask = cv2.bitwise_and(color_mask, roi_mask)

        # 3) Canny on gray, gated by color mask (more robust in Gazebo)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self.blur_ksize >= 3 and self.blur_ksize % 2 == 1:
            gray = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        edges = cv2.bitwise_and(edges, color_mask)

        # 4) Lane mask (prefer edges; fall back to color area if edges too sparse)
        lane_mask = edges.copy()
        nz_edges = int(np.count_nonzero(lane_mask))
        if nz_edges < self.min_mask_area:
            # Use closed color mask (no edges) as fallback
            lane_mask = color_mask.copy()
            lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            lane_mask = cv2.medianBlur(lane_mask, 5)

        # 5) Multi-row center extraction
        centers_px = []
        rows_y = []
        roi_h = max(1, y_bot - y_top)
        lm_w = max(2, self.min_lane_w_px)
        # weights prefer bottom (closer to car)
        weights = []
        for r in self.scan_rows:
            yy = y_top + int(roi_h * float(r))
            yy = max(y_top, min(y_bot - 1, yy))
            scan = lane_mask[yy, :]
            xs = np.where(scan > 0)[0]
            if xs.size >= lm_w:
                left, right = int(xs[0]), int(xs[-1])
                if (right - left) >= lm_w:
                    cx = 0.5 * (left + right)
                else:
                    cx = float(xs.mean())
                centers_px.append(cx)
                rows_y.append(yy)
                # weight by closeness to bottom
                w = 1.0 + 2.0 * ((yy - y_top) / float(roi_h + 1e-6))
                weights.append(w)

        # 6) Decide center & heading
        have_rows = len(centers_px) >= self.min_valid_rows
        img_center_px = 0.5 * W
        err_norm, heading_norm = 0.0, 0.0
        overlay = bgr.copy() if self.debug_overlay else None

        if have_rows:
            centers_px = np.asarray(centers_px, dtype=np.float32)
            rows_y = np.asarray(rows_y, dtype=np.float32)
            weights = np.asarray(weights, dtype=np.float32)
            # weighted center (prefer lower rows)
            center_est_px = float(np.average(centers_px, weights=weights))
            # heading: slope of center vs row
            try:
                z = np.polyfit(rows_y, centers_px, 1)
                slope = float(z[0])   # px per pixel-height
            except Exception:
                slope = 0.0

            # lateral error normalized to [-1..1]
            err_px = img_center_px - center_est_px
            err_norm = float(err_px / (0.5 * W))
            # heading normalization (scale slope by width)
            heading_norm = float(slope / (0.5 * W))

            self.last_center = center_est_px
            self.last_ok_time = rospy.Time.now()

            # Debug draw
            if overlay is not None:
                cv2.polylines(overlay, [roi_poly], True, (0, 255, 0), 2)
                for yy, cx in zip(rows_y.astype(int), centers_px.astype(int)):
                    cv2.line(overlay, (cx, yy), (cx, max(0, yy - 30)), (0, 0, 255), 2)
                cv2.line(overlay, (int(img_center_px), y_bot - 5),
                         (int(img_center_px), y_bot - 60), (255, 0, 0), 2)
        else:
            # 7) Green-balance fallback (if enabled)
            used_fallback = False
            if self.use_green_balance:
                gmask = self._green_mask(hsv, roi_mask)
                nz_g = int(np.count_nonzero(gmask))
                if nz_g >= self.min_green_cols:
                    # Find column where cumulative green ~ half
                    col_sum = np.sum(gmask[y_top:y_bot, :], axis=0).astype(np.float32)
                    cs = np.cumsum(col_sum)
                    total = cs[-1] if cs.size else 0.0
                    if total > 0.0:
                        half = 0.5 * total
                        k = int(np.searchsorted(cs, half))
                        k = max(0, min(W - 1, k))
                        center_est_px = float(k)
                        err_px = img_center_px - center_est_px
                        err_norm = float(err_px / (0.5 * W))
                        heading_norm = 0.0  # unknown; be conservative
                        self.last_center = center_est_px
                        self.last_ok_time = rospy.Time.now()
                        used_fallback = True
                        if overlay is not None:
                            cv2.polylines(overlay, [roi_poly], True, (0, 255, 255), 2)
                            cv2.line(overlay, (k, y_bot - 5), (k, max(y_top, y_bot - 60)), (0, 255, 255), 2)
                            cv2.putText(overlay, "GREEN BALANCE", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            if not used_fallback:
                # Hold last command briefly, then stop
                age_ms = (rospy.Time.now() - self.last_ok_time).to_sec() * 1000.0
                if age_ms < self.hold_bad_ms:
                    self.publish_cmd(self.last_cmd.speed, self.last_cmd.steering_angle)
                else:
                    self.pub_err.publish(0.0)
                    self.publish_cmd(0.0, 0.0)
                self._publish_debug(overlay, lane_mask, roi_poly, y_top, y_bot,
                                    text="NO LANE", color=(0, 0, 255))
                return

        # 8) PD + heading -> steering
        t = rospy.get_time()
        dt = max(1e-3, t - self.prev_t)
        d_err = (err_norm - self.prev_err) / dt
        self.prev_err, self.prev_t = err_norm, t

        steer = self.kp * err_norm + self.kd * d_err + self.k_heading * heading_norm
        steer = clamp(steer, -self.steer_limit, self.steer_limit)

        # 9) Speed scheduling: slow down for larger steer
        speed = self.target_speed * (1.0 - self.steer_slowdown * abs(steer) / self.steer_limit)
        speed = max(self.min_speed, speed)

        # 10) Publish
        self.pub_err.publish(float(err_norm))
        self.publish_cmd(speed, steer)

        # 11) Debug frame
        if overlay is not None:
            # show lane mask inside ROI as semi-transparent overlay
            mask_rgb = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)
            alpha = 0.45
            overlay = cv2.addWeighted(overlay, 1.0, mask_rgb, alpha, 0.0)
            txt = f"err={err_norm:+.2f} d={d_err:+.2f} hd={heading_norm:+.2f} st={steer:+.2f} v={speed:.2f}"
            cv2.putText(overlay, txt, (10, y_top - 10 if y_top > 20 else 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        self._publish_debug(overlay, lane_mask, roi_poly, y_top, y_bot)

    # ---------- helpers ----------
    def _green_mask(self, hsv, roi_mask):
        """Binary mask of 'green-ish grass' inside ROI."""
        gm = cv2.inRange(hsv,
                         (self.green_h_lo, self.green_s_min, self.green_v_min),
                         (self.green_h_hi, 255, 255))
        gm = cv2.bitwise_and(gm, roi_mask)
        gm = cv2.medianBlur(gm, 5)
        return gm

    def _publish_debug(self, overlay_bgr, lane_mask, roi_poly, y_top, y_bot, text=None, color=(0, 255, 0)):
        if overlay_bgr is None:
            # build from scratch for safety
            H, W = lane_mask.shape[:2]
            overlay_bgr = np.zeros((H, W, 3), dtype=np.uint8)
            mask_rgb = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)
            overlay_bgr[y_top:y_bot, :] = mask_rgb[y_top:y_bot, :]
            cv2.polylines(overlay_bgr, [roi_poly], True, color, 2)
        else:
            cv2.polylines(overlay_bgr, [roi_poly], True, color, 2)
        if text:
            cv2.putText(overlay_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, color, 2, cv2.LINE_AA)
        try:
            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(overlay_bgr, encoding="bgr8"))
        except Exception as e:
            rospy.logwarn_throttle(2.0, "debug publish: %s", e)


if __name__ == "__main__":
    VisionLKA()
    rospy.spin()

