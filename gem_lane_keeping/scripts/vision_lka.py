#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lane Keeping Assist (vision-based) for Polaris GEM simulator.

Inputs
------
- "image" (sensor_msgs/Image)  : front camera (remap to /gem/front_single_camera/image_raw)
- "/gem/safety/stop" (std_msgs/Bool) : safety stop from supervisor

Outputs
-------
- "cmd" (ackermann_msgs/AckermannDrive)  : steering + target speed
- "lateral_error" (std_msgs/Float32)     : normalized lateral error (+ > steer right)
- "debug" (sensor_msgs/Image)            : visualization of ROI mask and features

Design notes
------------
1) Robust mask from HSV/HLS whites + LAB yellow (road edge).
2) Multi-row scanners across ROI bottom to estimate lane center from both borders.
3) Fallback when only left yellow edge is visible: center = left + estimated lane width / 2.
   Lane width is adapted online within reasonable bounds.
4) Optional center cue from dashed white mid-lane (if visible in middle band).
5) Controller = PD (lateral error) + small I-term (bias/offset compensation)
   + heading term (slope of lane center vs row index).
6) Graceful stop if lane is not confidently detected.

Tune on the fly with rosparam, e.g.:
  rosparam set /gem/vision_lka/v_thresh 30
  rosparam set /gem/vision_lka/s_thresh 120
  rosparam set /gem/vision_lka/hls_L_min 180
  rosparam set /gem/vision_lka/lab_b_min 130
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
        self.kp           = rospy.get_param("~kp", 0.020)
        self.kd           = rospy.get_param("~kd", 0.060)
        self.ki           = rospy.get_param("~ki", 0.002)      # very small integral term
        self.k_heading    = rospy.get_param("~k_heading", 0.012)
        self.steer_limit  = rospy.get_param("~steer_limit", 0.50)
        self.center_bias_px = rospy.get_param("~center_bias_px", 0.0)  # manual bias in pixels

        # ---------- Color thresholds (mask building) ----------
        # HSV whiteness: high V, limited S (keeps bright whites)
        self.s_thresh     = rospy.get_param("~s_thresh", 120)
        self.v_thresh     = rospy.get_param("~v_thresh", 30)
        # HLS whiteness: high Lightness
        self.hls_L_min    = rospy.get_param("~hls_L_min", 180)
        # LAB yellow edge: high b channel
        self.lab_b_min    = rospy.get_param("~lab_b_min", 130)

        # ---------- Geometry & scanning ----------
        # ROI top as fraction of image height (keep bottom of image)
        self.roi_top         = rospy.get_param("~roi_top", 0.58)
        # Scan rows inside the ROI (fractions of ROI height 0..1)
        self.scan_rows       = rospy.get_param("~scan_rows", [0.35, 0.55, 0.70, 0.85])
        self.min_valid_rows  = rospy.get_param("~min_valid_rows", 1)
        self.min_mask_px     = rospy.get_param("~min_mask_px", 150)    # minimal area to consider valid
        self.min_lane_w      = rospy.get_param("~min_lane_width_px", 16)

        # Lane width adaptation (used by fallback from left edge)
        self.lane_px_width   = rospy.get_param("~lane_px_width_init", 110.0)
        self.lane_w_min      = rospy.get_param("~lane_w_min_px", 80.0)
        self.lane_w_max      = rospy.get_param("~lane_w_max_px", 150.0)
        self.lane_w_alpha    = rospy.get_param("~lane_w_alpha", 0.40)  # adaptation rate

        # Hold last command briefly when vision is lost (ms)
        self.hold_bad_ms     = rospy.get_param("~hold_bad_ms", 400)

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
        rospy.Subscriber("/gem/safety/stop", Bool, self.on_stop, queue_size=1)    # global safety

        rospy.loginfo(
            "vision_lka params: ts=%.2f kp=%.3f kd=%.3f ki=%.3f kh=%.3f steer=%.2f "
            "HSV(S<=%d,V>=%d) HLS(L>=%d) LAB(b>=%d) roi=%.2f scans=%s "
            "min_mask=%d min_w=%d min_rows=%d lane_w=%.0f[%.0f..%.0f] a=%.2f",
            self.target_speed, self.kp, self.kd, self.ki, self.k_heading, self.steer_limit,
            self.s_thresh, self.v_thresh, self.hls_L_min, self.lab_b_min,
            self.roi_top, str(self.scan_rows),
            self.min_mask_px, self.min_lane_w, self.min_valid_rows,
            self.lane_px_width, self.lane_w_min, self.lane_w_max, self.lane_w_alpha
        )

    # -------------------- Callbacks --------------------

    def on_stop(self, msg: Bool):
        """ Safety stop from supervisor – zero out command if true. """
        self.estop = bool(msg.data)

    def publish_cmd(self, speed, steer):
        """ Publish final Ackermann command with steer clamping and e-stop override. """
        if self.estop:
            speed = 0.0
            steer = 0.0
        cmd = AckermannDrive()
        cmd.speed = float(speed)
        cmd.steering_angle = float(clamp(steer, -self.steer_limit, self.steer_limit))
        self.pub_cmd.publish(cmd)
        self.last_cmd = cmd

    def on_image(self, msg: Image):
        """ Main processing: build mask -> find lane center -> compute steering -> publish. """
        # 1) Convert and crop ROI
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "cv_bridge err: %s", e)
            return

        h, w = bgr.shape[:2]
        y0 = min(max(0, int(h * self.roi_top)), h - 2)
        roi = bgr[y0:h, :]

        # 2) Color masks (HSV/HLS whites, LAB yellow)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

        white_hsv  = cv2.inRange(hsv, (0, 0, self.v_thresh), (179, self.s_thresh, 255))
        white_hls  = cv2.inRange(hls[:, :, 1], self.hls_L_min, 255)
        yellow_lab = cv2.inRange(lab[:, :, 2], self.lab_b_min, 255)

        mask_bi = cv2.bitwise_or(white_hsv, white_hls)            # all whites (edges + dashes)
        mask    = cv2.bitwise_or(mask_bi, yellow_lab)             # add yellow edge (left)
        mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask    = cv2.medianBlur(mask, 5)

        ys = mask.shape[0]
        nz_total = int(np.count_nonzero(mask))
        if nz_total < self.min_mask_px:
            # Not enough signal -> brief hold then stop
            if (rospy.Time.now() - self.last_ok_time).to_sec() * 1000.0 < self.hold_bad_ms:
                self.publish_cmd(self.last_cmd.speed, self.last_cmd.steering_angle)
            else:
                self.publish_cmd(0.0, 0.0)
            self._debug_and_stop(mask, w, h, y0, "NO LANE (area)")
            return

        # 3) Primary center from both borders using multi-row scans
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
            # Average of lower half of valid centers (gives more weight to near field)
            lane_center_edges = float(np.mean(centers[-max(1, len(centers)//2):]))
            # Adapt lane width (median across rows), clamped to reasonable limits
            if lane_widths:
                lw = float(np.median(lane_widths))
                lw = float(np.clip(lw, self.lane_w_min, self.lane_w_max))
                self.lane_px_width = (1.0 - self.lane_w_alpha) * self.lane_px_width + self.lane_w_alpha * lw
            # Heading: slope of center vs row index
            z = np.polyfit(np.array(used_y, dtype=np.float32),
                           np.array(centers, dtype=np.float32), 1)
            slope = z[0]
            heading = float(slope) / (w * 0.5)

        # 4) Fallback from left yellow edge only (fit line)
        lane_center_left = None
        ys_idx, xs_idx = np.where(yellow_lab > 0)
        if xs_idx.size >= self.min_lane_w:
            fit = cv2.fitLine(
                np.column_stack([ys_idx.astype(np.float32), xs_idx.astype(np.float32)]),
                cv2.DIST_L2, 0, 0.01, 0.01
            )
            vy, vx, y0f, x0f = fit.squeeze()
            y_query = float(ys - 1)
            x_left  = x0f + (vx / vy) * (y_query - y0f) if abs(vy) > 1e-6 else x0f
            lane_center_left = float(x_left + 0.5 * self.lane_px_width)
            if not have_both and abs(vy) > 1e-6:
                heading = float(vx / vy) / (w * 0.5)

        # 5) Middle dashed white center (optional extra cue)
        mid_lo, mid_hi = int(w * 0.35), int(w * 0.65)
        mid_centers = []
        for r in (0.55, 0.70, 0.85):
            y_scan = np.clip(int(ys * r), 0, ys - 1)
            xs = np.where(mask_bi[y_scan, mid_lo:mid_hi] > 0)[0]
            if xs.size >= 6:
                mid_centers.append(mid_lo + float(xs.mean()))
        lane_center_mid = float(np.mean(mid_centers)) if mid_centers else None

        # 6) Fuse available center candidates
        candidates, weights = [], []
        if lane_center_edges is not None:
            candidates.append(lane_center_edges); weights.append(0.6)
        if lane_center_left is not None:
            candidates.append(lane_center_left);  weights.append(0.3)
        if lane_center_mid is not None:
            candidates.append(lane_center_mid);   weights.append(0.4)

        if not candidates:
            if (rospy.Time.now() - self.last_ok_time).to_sec() * 1000.0 < self.hold_bad_ms:
                self.publish_cmd(self.last_cmd.speed, self.last_cmd.steering_angle)
            else:
                self.publish_cmd(0.0, 0.0)
            self._debug_and_stop(mask, w, h, y0, "NO LANE (rows)")
            return

        lane_center_px = float(np.average(np.array(candidates), weights=np.array(weights)))

        # 7) Errors and controller (PD + I + heading)
        img_center_px = 0.5 * w + self.center_bias_px
        err = (img_center_px - lane_center_px) / (w * 0.5)   # normalized lateral error

        t = rospy.get_time()
        dt = max(1e-3, t - self.prev_t)
        d_err = (err - self.prev_err) / dt
        self.prev_err, self.prev_t = err, t

        # Anti-windup integral with soft clamping
        self.i_err = float(np.clip(self.i_err + err * dt, -0.5, 0.5))

        steer = self.kp * err + self.kd * d_err + self.ki * self.i_err + self.k_heading * heading
        speed = self.target_speed

        # 8) Publish
        self.pub_err.publish(float(err))
        self.publish_cmd(speed, steer)
        self.last_ok_time = rospy.Time.now()

        # 9) Debug image (full-frame with ROI pasted at the bottom)
        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for y_scan, cx in zip(used_y, centers):
            cv2.line(dbg, (int(cx), y_scan), (int(cx), max(0, y_scan - 30)), (0, 0, 255), 2)
        if lane_center_mid is not None:
            cv2.circle(dbg, (int(lane_center_mid), ys - 15), 4, (255, 255, 0), -1)
        if lane_center_left is not None:
            cv2.circle(dbg, (int(lane_center_left), ys - 25), 4, (0, 255, 255), -1)
        cv2.line(dbg, (int(0.5 * w), ys - 5), (int(0.5 * w), ys - 45), (255, 0, 0), 1)
        cv2.circle(dbg, (int(lane_center_px), ys - 5), 5, (0, 255, 0), -1)

        txt = (
            f"nz={nz_total} err={err:+.2f} de={d_err:+.2f} I={self.i_err:+.2f} "
            f"hd={heading:+.3f} st={steer:+.2f} v={speed:.2f} lw={self.lane_px_width:.0f}"
        )
        cv2.putText(dbg, txt, (10, dbg.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        self._publish_debug(dbg, w, h, y0)

    # -------------------- Helpers --------------------

    def _debug_and_stop(self, mask, w, h, y0, label):
        """ Compose a labeled debug image and publish (we already handled command outside). """
        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(dbg, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        self._publish_debug(dbg, w, h, y0)

    def _publish_debug(self, roi_bgr, w, h, y0):
        """ Paste ROI back into a black canvas (original size) and publish /vision_lka/debug. """
        canv = np.zeros((h, w, 3), dtype=np.uint8)
        try:
            canv[y0:h, :] = cv2.resize(roi_bgr, (w, h - y0))
        except Exception:
            # Fallback if resize fails due to tiny ROI
            canv[y0:h, :] = roi_bgr
        try:
            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(canv, encoding='bgr8'))
        except Exception as e:
            rospy.logwarn_throttle(2.0, "vision_lka: debug publish failed: %s", e)


if __name__ == "__main__":
    VisionLKA()
    rospy.spin()

