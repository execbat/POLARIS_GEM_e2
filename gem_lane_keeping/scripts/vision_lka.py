#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision-based Lane Keeping for GEM Polaris (ROS1/Noetic).

Features:
- Robust lane mask: HSV white, HLS white (L channel) and LAB yellow, with morphology.
- Multi-row edge scanning (left/right borders) to estimate lane center and heading.
- Fallback from the left yellow edge with a FIXED pixel offset to lane center.
- Optional mid-lane dashed white cue around image center.
- Confidence metric + "guard rails" to avoid drifting when vision is weak.
- PD(+I) steering; I-term auto-resets on low confidence.
- Speed governor: slow down for high steering angles and low confidence.
- Debug image published as a full-frame canvas with ROI overlay and annotations.

Remap topics in launch:
  image  -> /gem/front_single_camera/image_raw
  cmd    -> /vision_lka/cmd
  debug  -> /vision_lka/debug
  lateral_error -> /vision_lka/lateral_error
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class VisionLKA:
    def __init__(self):
        rospy.init_node("vision_lka")
        self.bridge = CvBridge()

        # ---------- Control gains / limits ----------
        self.target_speed = rospy.get_param("~target_speed", 1.5)
        self.min_speed    = rospy.get_param("~min_speed",   0.6)
        self.kp           = rospy.get_param("~kp", 0.020)
        self.kd           = rospy.get_param("~kd", 0.060)
        self.ki           = rospy.get_param("~ki", 0.0015)     # tiny; reset on low confidence
        self.k_heading    = rospy.get_param("~k_heading", 0.010)  # heading sign handled below
        self.steer_limit  = rospy.get_param("~steer_limit", 0.50)
        # Shift desired geometric center horizontally (+right, -left), px in image coords
        self.center_bias_px = rospy.get_param("~center_bias_px", 80.0)

        # ---------- Mask thresholds ----------
        # HSV white: S <= s_thresh, V >= v_thresh
        self.s_thresh     = rospy.get_param("~s_thresh", 120)
        self.v_thresh     = rospy.get_param("~v_thresh", 30)
        # HLS white: L >= hls_L_min
        self.hls_L_min    = rospy.get_param("~hls_L_min", 180)
        # LAB yellow: b >= lab_b_min
        self.lab_b_min    = rospy.get_param("~lab_b_min", 130)

        # ---------- Geometry / scanning ----------
        self.roi_top        = rospy.get_param("~roi_top", 0.58)  # fraction of image height
        self.scan_rows      = rospy.get_param("~scan_rows", [0.55, 0.70, 0.85])  # row fractions in ROI
        self.min_valid_rows = rospy.get_param("~min_valid_rows", 1)
        self.min_mask_px    = rospy.get_param("~min_mask_px", 150)
        self.min_lane_w     = rospy.get_param("~min_lane_width_px", 16)

        # Lane width adaptation (used ONLY when both edges are visible)
        self.lane_px_width   = rospy.get_param("~lane_px_width_init", 150.0)
        self.lane_w_min      = rospy.get_param("~lane_w_min_px",  90.0)
        self.lane_w_max      = rospy.get_param("~lane_w_max_px", 220.0)
        self.lane_w_alpha    = rospy.get_param("~lane_w_alpha", 0.35)

        # Behavior: guard rails & hold
        self.hold_bad_ms     = rospy.get_param("~hold_bad_ms", 500)   # ms to hold last command when vision drops
        self.guard_lo_frac   = rospy.get_param("~guard_lo_frac", 0.20)
        self.guard_hi_frac   = rospy.get_param("~guard_hi_frac", 0.80)

        # Fusion weights and yellow fallback offset
        # Fixed pixel offset from the left yellow edge to our lane center
        self.offset_from_yellow_px = rospy.get_param("~offset_from_yellow_px", 200.0)
        self.edges_weight = rospy.get_param("~edges_weight", 0.55)   # both borders (primary)
        self.mid_weight   = rospy.get_param("~mid_weight",   0.60)   # dashed white near image center
        self.yellow_weight= rospy.get_param("~yellow_weight",0.01)   # yellow fallback (secondary)

        # ---------- State ----------
        self.estop        = False
        self.prev_err     = 0.0
        self.i_err        = 0.0
        self.prev_t       = rospy.get_time()
        self.last_ok_time = rospy.Time(0.0)
        self.last_cmd     = AckermannDrive()

        # ---------- IO ----------
        self.pub_cmd = rospy.Publisher("cmd", AckermannDrive, queue_size=10)
        self.pub_err = rospy.Publisher("lateral_error", Float32, queue_size=10)
        self.pub_dbg = rospy.Publisher("debug", Image, queue_size=1)
        rospy.Subscriber("image", Image, self.on_image, queue_size=1)             # remap in launch
        rospy.Subscriber("/gem/safety/stop", Bool, self.on_stop, queue_size=1)

        rospy.loginfo(
            "vision_lka params: ts=%.2f kp=%.3f kd=%.3f ki=%.4f kh=%.3f steer=%.2f "
            "HSV(S<=%d,V>=%d) HLS(L>=%d) LAB(b>=%d) roi=%.2f scans=%s "
            "min_mask=%d min_w=%d min_rows=%d hold=%dms center_bias=%.1f "
            "lane_init=%.0f yellow_offset=%.0f w_edges=%.2f w_mid=%.2f w_yellow=%.2f",
            self.target_speed, self.kp, self.kd, self.ki, self.k_heading, self.steer_limit,
            self.s_thresh, self.v_thresh, self.hls_L_min, self.lab_b_min,
            self.roi_top, str(self.scan_rows),
            self.min_mask_px, self.min_lane_w, self.min_valid_rows, self.hold_bad_ms,
            self.center_bias_px, self.lane_px_width, self.offset_from_yellow_px,
            self.edges_weight, self.mid_weight, self.yellow_weight
        )

    # -------------------- Callbacks --------------------

    def on_stop(self, msg: Bool):
        self.estop = bool(msg.data)

    def publish_cmd(self, speed, steer):
        if self.estop:
            speed = 0.0
            steer = 0.0
        cmd = AckermannDrive()
        cmd.speed = float(speed)
        cmd.steering_angle = float(clamp(steer, -self.steer_limit, self.steer_limit))
        self.pub_cmd.publish(cmd)
        self.last_cmd = cmd

    def on_image(self, msg: Image):
        # 1) Convert and crop ROI
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "cv_bridge error: %s", e)
            return

        h, w = bgr.shape[:2]
        y0 = min(max(0, int(h * self.roi_top)), h - 2)
        roi = bgr[y0:h, :]

        # 2) Build lane mask (HSV/HLS for white, LAB for yellow)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

        white_hsv  = cv2.inRange(hsv, (0, 0, self.v_thresh), (179, self.s_thresh, 255))
        white_hls  = cv2.inRange(hls[:, :, 1], self.hls_L_min, 255)
        yellow_lab = cv2.inRange(lab[:, :, 2], self.lab_b_min, 255)

        mask_bi = cv2.bitwise_or(white_hsv, white_hls)   # dashed + white edges
        mask    = cv2.bitwise_or(mask_bi, yellow_lab)    # add yellow edge
        mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask    = cv2.medianBlur(mask, 5)

        ys = mask.shape[0]
        nz = int(np.count_nonzero(mask))
        if nz < self.min_mask_px:
            self._hold_or_stop("NO LANE (area)")
            self._publish_debug(self._label(mask, "NO LANE (area)"), w, h, y0)
            return

        # 3) Multi-row edge scanning (both borders)
        centers, used_y, lane_widths = [], [], []
        for r in self.scan_rows:
            y_scan = np.clip(int(ys * float(r)), 0, ys - 1)
            xs = np.where(mask[y_scan, :] > 0)[0]
            if xs.size >= self.min_lane_w:
                L, R = int(xs[0]), int(xs[-1])
                if (R - L) >= self.min_lane_w:
                    centers.append(0.5 * (L + R))
                    lane_widths.append(float(R - L))
                    used_y.append(y_scan)

        have_both = (len(centers) >= self.min_valid_rows)
        lane_center_edges = None
        heading = 0.0

        if have_both:
            # Average on lower half rows (closer to the car)
            lane_center_edges = float(np.mean(centers[-max(1, len(centers)//2):]))
            # Adapt lane width (only when both edges are seen)
            lw = float(np.median(lane_widths))
            lw = float(np.clip(lw, self.lane_w_min, self.lane_w_max))
            self.lane_px_width = (1.0 - self.lane_w_alpha) * self.lane_px_width + self.lane_w_alpha * lw
            # Heading from slope of centers vs. row index
            z = np.polyfit(np.array(used_y, np.float32), np.array(centers, np.float32), 1)
            slope = z[0]
            # Sign convention: turning right -> slope>0 -> negative heading (to steer right)
            heading = float(-slope) / (w * 0.5)

        # 4) Fallback from left yellow edge with FIXED offset to our lane center
        lane_center_left = None
        ys_idx, xs_idx = np.where(yellow_lab > 0)
        if xs_idx.size >= self.min_lane_w:
            # Fit line through yellow pixels (points are (x,y) pairs)
            pts = np.column_stack([xs_idx.astype(np.float32), ys_idx.astype(np.float32)])
            try:
                vx, vy, x0, y0f = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).squeeze()
                y_query = float(ys - 1)
                if abs(vy) < 1e-6:
                    x_left = float(x0)
                else:
                    x_left = float(x0 + (vx / vy) * (y_query - y0f))
                lane_center_left = float(x_left + self.offset_from_yellow_px)
                # If edges were not available, derive heading from yellow slope as well
                if not have_both and abs(vy) > 1e-6:
                    heading = float(-(vx / vy)) / (w * 0.5)
            except cv2.error:
                lane_center_left = None

        # 5) Mid-lane dashed white cue inside central window
        mid_lo, mid_hi = int(w * 0.35), int(w * 0.65)
        mid_centers = []
        for r in (0.60, 0.75, 0.90):
            y_scan = np.clip(int(ys * r), 0, ys - 1)
            xs = np.where(mask_bi[y_scan, mid_lo:mid_hi] > 0)[0]
            if xs.size >= 6:
                mid_centers.append(mid_lo + float(xs.mean()))
        lane_center_mid = float(np.mean(mid_centers)) if mid_centers else None

        # 6) Fuse candidates with weights and compute confidence
        cands, wts, conf = [], [], 0.0
        if lane_center_edges is not None:
            cands.append(lane_center_edges); wts.append(self.edges_weight); conf += 0.6
        if lane_center_mid is not None:
            cands.append(lane_center_mid);   wts.append(self.mid_weight);   conf += 0.3
        if lane_center_left is not None:
            cands.append(lane_center_left);  wts.append(self.yellow_weight); conf += 0.2

        if not cands:
            self._hold_or_stop("NO LANE (rows)")
            self._publish_debug(self._label(mask, "NO LANE (rows)"), w, h, y0)
            return

        lane_center_px = float(np.average(np.array(cands), weights=np.array(wts)))
        img_center_px  = 0.5 * w + self.center_bias_px

        # Guard rails: when confidence is low, reject far-out centers
        if (lane_center_px < self.guard_lo_frac * w or lane_center_px > self.guard_hi_frac * w) and conf < 0.6:
            self._hold_or_stop("OUT OF BOUNDS")
            self._publish_debug(self._label(mask, "OUT OF BOUNDS"), w, h, y0)
            return

        # Bias center gently toward geometric center at low confidence
        if conf < 0.6:
            lane_center_px = 0.8 * lane_center_px + 0.2 * img_center_px

        # 7) Controller: PD(+I) + heading, with integral reset on low confidence
        err = (img_center_px - lane_center_px) / (w * 0.5)  # >0 => steer LEFT
        t = rospy.get_time()
        dt = max(1e-3, t - self.prev_t)
        d_err = (err - self.prev_err) / dt
        self.prev_err, self.prev_t = err, t

        if conf < 0.5:
            self.i_err = 0.0
        else:
            self.i_err = float(np.clip(self.i_err + err * dt, -0.5, 0.5))

        steer = self.kp * err + self.kd * d_err + self.ki * self.i_err + self.k_heading * heading

        # Speed governor: slow down for large steering and low confidence
        steer_ratio = min(1.0, abs(steer) / max(1e-6, self.steer_limit))
        conf_scale  = 0.5 + 0.5 * min(1.0, conf)      # [0.5 .. 1.0]
        speed = max(self.min_speed, self.target_speed * (1.0 - 0.9 * steer_ratio) * conf_scale)

        # Publish
        self.pub_err.publish(float(err))
        self.publish_cmd(speed, steer)
        self.last_ok_time = rospy.Time.now()

        # 8) Debug canvas (mask + annotations pasted back into full frame)
        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for y_scan, cx in zip(used_y, centers):
            cv2.line(dbg, (int(cx), y_scan), (int(cx), max(0, y_scan - 30)), (0, 0, 255), 2)
        ys_h = mask.shape[0]
        # Mark fused center (green) and geometric center (blue bar)
        cv2.circle(dbg, (int(lane_center_px), ys_h - 5), 5, (0, 255, 0), -1)
        cv2.line(dbg, (int(0.5 * w), ys_h - 5), (int(0.5 * w), ys_h - 45), (255, 0, 0), 1)
        # Mark mid-lane and yellow-derived centers if present
        if lane_center_mid  is not None: cv2.circle(dbg, (int(lane_center_mid),  ys_h - 16), 4, (255, 255, 0), -1)
        if lane_center_left is not None: cv2.circle(dbg, (int(lane_center_left), ys_h - 26), 4, (0, 255, 255), -1)

        txt = ("nz={:d} conf={:.2f} err={:+.2f} de={:+.2f} I={:+.2f} "
               "hd={:+.3f} st={:+.2f} v={:.2f} lw={:.0f}").format(
                   nz, conf, err, d_err, self.i_err, heading, steer, speed, self.lane_px_width)
        cv2.putText(dbg, txt, (10, dbg.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        self._publish_debug(dbg, w, h, y0)

    # -------------------- Helpers --------------------

    def _hold_or_stop(self, reason: str):
        # Hold last command briefly; then stop (softly) if vision is gone
        if (rospy.Time.now() - self.last_ok_time).to_sec() * 1000.0 < self.hold_bad_ms:
            self.publish_cmd(self.last_cmd.speed, self.last_cmd.steering_angle)
        else:
            self.publish_cmd(0.0, 0.0)

    def _label(self, mask, text):
        bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2, cv2.LINE_AA)
        return bgr

    def _publish_debug(self, roi_bgr, w, h, y0):
        # Paste ROI back into a full-frame black canvas for easy viewing in rqt/image_view
        canv = np.zeros((h, w, 3), dtype=np.uint8)
        canv[y0:h, :] = cv2.resize(roi_bgr, (w, h - y0))
        try:
            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(canv, encoding='bgr8'))
        except Exception as e:
            rospy.logwarn_throttle(2.0, "vision_lka: debug publish failed: %s", e)


if __name__ == "__main__":
    VisionLKA()
    rospy.spin()

