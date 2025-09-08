#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision LKA with PID like in the simulator:
  δ = -(Kp*e_lat + Ki*∫e + Kd*de + K_heading*e_head), with steering rate limit,
  EMA-filtered derivative, simple anti-windup, and speed slowdown in curves.
Also: color-mask primary detector, Canny+Hough fallback, recovery mode.
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

        # ======== CONTROL (PID like simulator) ========
        self.kp        = rospy.get_param("~kp",        0.03)
        self.ki        = rospy.get_param("~ki",        0.001)
        self.kd        = rospy.get_param("~kd",        0.08)
        self.k_heading = rospy.get_param("~k_heading", 0.10)   
        self.der_alpha = rospy.get_param("~der_alpha", 0.7)    # EMA filter D (0..1)
        self.i_leak    = rospy.get_param("~i_decay",   0.02)   # "leaky I", 0..0.1

        # ======== STEERING SHAPING ========
        self.steer_limit       = rospy.get_param("~steer_limit", 0.90)
        self.steer_rate_limit  = rospy.get_param("~steer_rate_limit", 2.0)  # rad/sec
        self.steer_alpha       = rospy.get_param("~steer_alpha", 0.0)       # 
        self.steer_sign        = rospy.get_param("~steer_sign",  -1.0)      # -1 inverted steering

        # ======== SPEED PROFILE ========
        self.target_speed   = rospy.get_param("~target_speed", 1.5)
        self.min_speed      = rospy.get_param("~min_speed",    0.7)
        self.recovery_speed = rospy.get_param("~recovery_speed", 0.6)
        self.k_curve_speed  = rospy.get_param("~k_curve_speed", 8.0)  
        self.steer_slowdown = rospy.get_param("~steer_slowdown", 0.15)

        # ======== COLOR MASK  ========
        self.s_thresh   = rospy.get_param("~s_thresh",  rospy.get_param("~white_s_max", 100))
        self.v_thresh   = rospy.get_param("~v_thresh",  rospy.get_param("~white_v_min", 35))
        self.hls_L_min  = rospy.get_param("~hls_L_min", 190)
        self.lab_b_min  = rospy.get_param("~lab_b_min", 140)
        
        self.yellow_h_lo = rospy.get_param("~yellow_h_lo", 15)
        self.yellow_h_hi = rospy.get_param("~yellow_h_hi", 40)
        self.yellow_s_min= rospy.get_param("~yellow_s_min", 80)
        self.yellow_v_min= rospy.get_param("~yellow_v_min", 80)
        self.use_yellow_hsv = rospy.get_param("~use_yellow_hsv", False)

        # ======== GEOMETRY / SCANS ========
        self.roi_top      = rospy.get_param("~roi_top", rospy.get_param("~roi_y_top", 0.62))
        self.roi_top_reco = rospy.get_param("~roi_top_recovery", 0.55)
        self.scan_rows    = rospy.get_param("~scan_rows", [0.60, 0.70, 0.80, 0.90])
        self.min_mask_px  = rospy.get_param("~min_mask_px", rospy.get_param("~min_mask_area", 150))
        self.min_lane_w   = rospy.get_param("~min_lane_width_px", 22)
        self.min_valid_rows = rospy.get_param("~min_valid_rows", 2)
        self.hold_bad_ms  = rospy.get_param("~hold_bad_ms", 600)
        self.stop_if_lost = rospy.get_param("~stop_if_lost", False)
        
        # ======== LANE GEOMETRY FILTERS (center-line ignore) ========
        self.ignore_center_band = rospy.get_param("~ignore_center_band", True)  
        self.center_band_frac   = rospy.get_param("~center_band_frac", 0.20)    
        self.min_run_px         = rospy.get_param("~min_run_px", 12)            
        self.prefer_edges       = rospy.get_param("~prefer_edges", True)       

        # ======== HOUGH/CANNY (fallback) ========
        self.canny1 = rospy.get_param("~canny1", rospy.get_param("~canny_low", 60))
        self.canny2 = rospy.get_param("~canny2", rospy.get_param("~canny_high",150))
        self.hough_threshold   = rospy.get_param("~hough_threshold", 25)
        self.hough_min_length  = rospy.get_param("~hough_min_length", 60)
        self.hough_max_gap     = rospy.get_param("~hough_max_gap", 20)
        self.hough_min_angle_deg = rospy.get_param("~hough_min_angle_deg", 15)

        # ======== STATE ========
        self.estop        = False
        self.prev_t       = rospy.get_time()
        self.prev_err     = 0.0
        self.prev_de_f    = 0.0
        self.int_err      = 0.0
        self.prev_delta   = 0.0
        self.steer_filt   = 0.0
        self.last_ok_time = rospy.Time(0.0)
        self.last_cmd     = AckermannDrive()
        self.last_lane_w_px = 120.0
        self.last_center_px = None
        self.lost_frames  = 0

        # ======== I/O ========
        self.pub_cmd = rospy.Publisher("cmd", AckermannDrive, queue_size=10)
        self.pub_err = rospy.Publisher("lateral_error", Float32, queue_size=10)
        self.pub_dbg = rospy.Publisher("debug", Image, queue_size=1)
        rospy.Subscriber("image", Image, self.on_image, queue_size=1)
        rospy.Subscriber("/gem/safety/stop", Bool, self.on_stop, queue_size=1)

        rospy.loginfo("vision_lka gains: P=%.3f I=%.3f D=%.3f Kh=%.3f der_alpha=%.2f",
                      self.kp, self.ki, self.kd, self.k_heading, self.der_alpha)

    # ---------- callbacks ----------
    def on_stop(self, msg: Bool):
        self.estop = bool(msg.data)

    def publish_cmd(self, speed, steer):
        if self.estop:
            speed = 0.0; steer = 0.0
        cmd = AckermannDrive()
        cmd.speed = float(speed)
        cmd.steering_angle = float(clamp(steer, -self.steer_limit, self.steer_limit))
        self.pub_cmd.publish(cmd)

    def on_image(self, msg: Image):
        # Image → ROI (adaptive in recovery)
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "cv_bridge: %s", e); return
        h, w = bgr.shape[:2]

        recovering = (rospy.Time.now() - self.last_ok_time).to_sec()*1000.0 >= self.hold_bad_ms
        roi_top_eff = self.roi_top_reco if recovering else self.roi_top
        y0 = int(h * roi_top_eff); y0 = min(max(y0, 0), h-2)
        roi = bgr[y0:h, :]

        # Primary color mask
        mask_color = self._color_mask(roi)
        if self.ignore_center_band:
            cx0 = int((0.5 - 0.5*self.center_band_frac) * w)
            cx1 = int((0.5 + 0.5*self.center_band_frac) * w)
            cx0 = max(0, min(w-1, cx0));  cx1 = max(0, min(w, cx1))
            mask_color[:, cx0:cx1] = 0
        
        centers, used_y, lane_w_px = self._scan_centers(mask_color)

        # Fallback (Canny+Hough)
        used_fallback = False
        lines_draw = None
        if len(centers) < self.min_valid_rows:
            center_h, lane_w_h, lines_draw = self._hough_center(roi)
            if center_h is not None:
                ys = mask_color.shape[0]
                centers = [center_h]*max(2, self.min_valid_rows)
                used_y  = [int(ys*0.85), int(ys*0.95)]
                lane_w_px = lane_w_h
                used_fallback = True

        # Recovery if still nothing
        if len(centers) < self.min_valid_rows:
            self._recovery_publish(mask_color, w, h, y0, "NO LANE")
            self.lost_frames += 1
            return
        else:
            self.lost_frames = 0

        # Errors: lateral (normalized) and heading (rad, like simulator)
        img_center_px  = 0.5 * w
        lane_center_px = float(np.mean(centers[-max(1, len(centers)//2):]))
        if lane_w_px is not None and lane_w_px > 5:
            self.last_lane_w_px = 0.8*self.last_lane_w_px + 0.2*float(lane_w_px)
        self.last_center_px = lane_center_px

        
        err = (lane_center_px - img_center_px) / (w * 0.5)

        # Heading: fit centers vs y in normalized coords and take arctan of slope → рад
        ys = mask_color.shape[0]
        y_norm = (np.array(used_y, dtype=np.float32) / max(1.0, ys))  # 0..1 вниз
        c_norm = (np.array(centers, dtype=np.float32) - img_center_px) / (w * 0.5)
        if y_norm.ptp() < 1e-6:
            slope_norm = 0.0
        else:
            z = np.polyfit(y_norm, c_norm, 1)
            slope_norm = float(z[0])
        heading = math.atan(slope_norm)  # рад

        # PID like simulator
        t  = rospy.get_time()
        dt = max(1e-3, t - self.prev_t)
        de = (err - self.prev_err) / dt
        de_f = self.der_alpha * self.prev_de_f + (1.0 - self.der_alpha) * de

        
        delta_fb = -(self.kp*err + self.ki*self.int_err + self.kd*de_f + self.k_heading*heading)
        raw = delta_fb 

        # rate limit 
        max_step = self.steer_rate_limit * dt
        raw = clamp(raw, self.prev_delta - max_step, self.prev_delta + max_step)

        # saturation
        delta = clamp(raw, -self.steer_limit, self.steer_limit)

        # anti-windup
        if abs(delta) < self.steer_limit - 1e-6:
            self.int_err = (1.0 - self.i_leak) * self.int_err + err * dt
            self.int_err = clamp(self.int_err, -0.8, 0.8)

        self.prev_err   = err
        self.prev_de_f  = de_f
        self.prev_delta = delta

        
        alpha = float(self.steer_alpha)
        new_delta = delta * self.steer_sign

        if alpha <= 0.0:              # сглаживание выключено -> сразу берём новое
            steer_cmd = new_delta
        elif alpha >= 1.0:            # 1.0 = без инерции (полный пропуск)
            steer_cmd = new_delta
        else:                         # 0 < alpha < 1: EMA, где alpha — доля НОВОГО
            self.steer_filt = (1.0 - alpha) * self.steer_filt + alpha * new_delta
            steer_cmd = self.steer_filt

        self.steer_filt = steer_cmd

        # Speed schedule 
        v_curve = 1.0 / (1.0 + self.k_curve_speed * abs(heading))
        v_steer = 1.0 - self.steer_slowdown * abs(self.steer_filt)
        v_scale = max(0.25, min(v_curve, v_steer))
        speed   = clamp(self.target_speed * v_scale, self.min_speed, self.target_speed)

        # Publish + state
        self.pub_err.publish(float(err))
        self.publish_cmd(speed, self.steer_filt)
        self.last_ok_time = rospy.Time.now()
        self.last_cmd.speed = float(speed)
        self.last_cmd.steering_angle = float(self.steer_filt)
        self.prev_t = t
        
        if self.ignore_center_band:
            cv2.rectangle(dbg,
                          (cx0, 0), (cx1, dbg.shape[0]-1),
                          (0, 128, 255), 1)

        # 8) Debug overlay
        dbg = cv2.cvtColor(mask_color, cv2.COLOR_GRAY2BGR)
        for y_scan, cx in zip(used_y, centers):
            cv2.line(dbg, (int(cx), y_scan), (int(cx), max(0, y_scan-30)), (0, 0, 255), 2)
        if used_fallback and lines_draw is not None:
            dbg = cv2.addWeighted(dbg, 1.0, lines_draw, 0.8, 0.0)
        cv2.line(dbg, (int(img_center_px), ys-5), (int(img_center_px), ys-55), (255, 0, 0), 1)
        txt = f"err={err:+.2f} de={de_f:+.2f} I={self.int_err:+.2f} hd={heading:+.2f} δ={self.steer_filt:+.2f} v={speed:.2f}"
        cv2.putText(dbg, txt, (10, dbg.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1, cv2.LINE_AA)
        self._publish_debug(dbg, w, h, y0)

    # ---------- detectors ----------
    def _color_mask(self, roi_bgr):
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        hls = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HLS)
        lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)

        white_hsv  = cv2.inRange(hsv, (0, 0, self.v_thresh), (179, self.s_thresh, 255))
        white_hls  = cv2.inRange(hls[:, :, 1], self.hls_L_min, 255)

        if self.use_yellow_hsv:
            yellow_hsv = cv2.inRange(hsv,
                                     (self.yellow_h_lo, self.yellow_s_min, self.yellow_v_min),
                                     (self.yellow_h_hi, 255, 255))
            mask = cv2.bitwise_or(white_hsv, white_hls)
            mask = cv2.bitwise_or(mask, yellow_hsv)
        else:
            yellow_lab = cv2.inRange(lab[:, :, 2], self.lab_b_min, 255)
            mask = cv2.bitwise_or(white_hsv, white_hls)
            mask = cv2.bitwise_or(mask, yellow_lab)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        mask = cv2.medianBlur(mask, 5)
        return mask

    def _scan_centers(self, mask):

        H, W = mask.shape[:2]
        centers, used_y, widths = [], [], []

        
        cx0 = int((0.5 - 0.5*self.center_band_frac) * W)
        cx1 = int((0.5 + 0.5*self.center_band_frac) * W)

        for r in self.scan_rows:
            y_scan = int(H * float(r))
            y_scan = 0 if y_scan < 0 else H-1 if y_scan >= H else y_scan
            row = mask[y_scan, :]

            xs = np.flatnonzero(row > 0)
            if xs.size == 0:
                continue
    
            
            split_idx = np.where(np.diff(xs) > 1)[0] + 1
            runs = np.split(xs, split_idx)

            
            runs = [run for run in runs if run.size >= int(self.min_run_px)]
    
            if self.ignore_center_band and runs:
                runs = [run for run in runs
                        if not (run[0] >= cx0 and run[-1] <= cx1)]  

            if not runs:
                continue

            
            left_run  = runs[0]
            right_run = runs[-1]
            if left_run is right_run:
                
                mid = 0.5 * (left_run[0] + left_run[-1])
                if mid < W * 0.5:
                    
                    center = mid + 0.5 * float(self.last_lane_w_px)
                    width  = float(self.last_lane_w_px)
                else:
                 
                    center = mid - 0.5 * float(self.last_lane_w_px)
                    width  = float(self.last_lane_w_px)
            else:
                
                left_edge  = left_run[-1]    
                right_edge = right_run[0]     
                width = right_edge - left_edge
                if width < self.min_lane_w:
                    
                    continue
                center = 0.5 * (left_edge + right_edge)

            
            center = float(np.clip(center, 0, W-1))
            centers.append(center)
            used_y.append(y_scan)
            if width is not None:
                widths.append(float(width))

        lane_w = float(np.median(widths)) if len(widths) > 0 else None
        return centers, used_y, lane_w

    def _hough_center(self, roi_bgr):
        h, w = roi_bgr.shape[:2]
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, self.canny1, self.canny2)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, self.hough_threshold,
                                minLineLength=self.hough_min_length, maxLineGap=self.hough_max_gap)
        overlay = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        if lines is None:
            return None, None, overlay

        left_xb, right_xb = [], []
        min_ang = np.deg2rad(self.hough_min_angle_deg)
        for l in lines[:,0,:]:
            x1,y1,x2,y2 = map(int, l)
            dy = y2 - y1; dx = x2 - x1
            if dy == 0: continue
            ang = abs(np.arctan2(dy, dx))
            if ang < min_ang:  # near-horizontal → ignore
                continue
            x_bottom = x1 + (h-1 - y1) * (dx / float(dy))
            if dx * dy < 0:   # negative slope (left boundary in image coords)
                left_xb.append(x_bottom);  cv2.line(overlay, (x1,y1), (x2,y2), (0,255,0), 2)
            else:
                right_xb.append(x_bottom); cv2.line(overlay, (x1,y1), (x2,y2), (255,0,0), 2)

        center_px, lane_w = None, None
        if left_xb and right_xb:
            xl = float(np.median(left_xb)); xr = float(np.median(right_xb))
            if xr - xl > 5:
                center_px = 0.5*(xl + xr); lane_w = xr - xl
        elif left_xb and self.last_lane_w_px is not None:
            xl = float(np.median(left_xb)); center_px = xl + 0.5*self.last_lane_w_px; lane_w = self.last_lane_w_px
        elif right_xb and self.last_lane_w_px is not None:
            xr = float(np.median(right_xb)); center_px = xr - 0.5*self.last_lane_w_px; lane_w = self.last_lane_w_px

        return center_px, lane_w, overlay

    # ---------- recovery & debug ----------
    def _recovery_publish(self, mask, w, h, y0, label):
        if self.stop_if_lost:
            self.publish_cmd(0.0, 0.0)
        else:
            speed = self.recovery_speed
            steer = getattr(self.last_cmd, "steering_angle", 0.0)
            self.publish_cmd(speed, steer)
        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(dbg, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        self._publish_debug(dbg, w, h, y0)

    def _publish_debug(self, roi_bgr, w, h, y0):
        canv = np.zeros((h, w, 3), dtype=np.uint8)
        canv[y0:h, :] = cv2.resize(roi_bgr, (w, h - y0))
        try:
            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(canv, encoding='bgr8'))
        except Exception as e:
            rospy.logwarn_throttle(2.0, "debug publish failed: %s", e)

if __name__ == "__main__":
    VisionLKA()
    rospy.spin()

