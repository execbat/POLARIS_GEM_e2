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
        
        # ======== SPEED/ACC LIMITS (anti-slip) ========
        self.wheelbase     = rospy.get_param("~wheelbase", 1.2)   # m
        self.a_accel_max   = rospy.get_param("~a_accel_max", 0.6) # m/s^2 
        self.a_brake_max   = rospy.get_param("~a_brake_max", 1.5) # m/s^2 
        self.a_lat_max     = rospy.get_param("~a_lat_max", 1.8)   # m/s^2 
        self.err_slow_k    = rospy.get_param("~err_slow_k", 0.6)  
        self.err_slow_th   = rospy.get_param("~err_slow_th", 0.25)

        # 
        self.v_cmd         = 0.0        
        self.prev_t_speed  = rospy.get_time()

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
        self.center_band_frac   = rospy.get_param("~center_band_frac", 0.18)    
        self.min_run_px         = rospy.get_param("~min_run_px", 12)            
        self.prefer_edges       = rospy.get_param("~prefer_edges", True)     
        
        self.center_beta   = rospy.get_param("~center_beta", 0.30)  
        self.heading_beta  = rospy.get_param("~heading_beta", 0.50)  

        # ======== HOUGH/CANNY (fallback) ========
        self.canny1 = rospy.get_param("~canny1", rospy.get_param("~canny_low", 60))
        self.canny2 = rospy.get_param("~canny2", rospy.get_param("~canny_high",150))
        self.hough_threshold   = rospy.get_param("~hough_threshold", 25)
        self.hough_min_length  = rospy.get_param("~hough_min_length", 60)
        self.hough_max_gap     = rospy.get_param("~hough_max_gap", 20)
        self.hough_min_angle_deg = rospy.get_param("~hough_min_angle_deg", 15)
        
        
        # --- ignore center line ---
        self.center_band_mode     = rospy.get_param("~center_band_mode", "dynamic")  # off|static|dynamic
        self.center_band_frac     = rospy.get_param("~center_band_frac", 0.22)       #  static: width part ROI
        self.center_band_k        = rospy.get_param("~center_band_k", 0.60)          #  dynamic: width = k*last_lane_w_px
        self.center_band_min_px   = rospy.get_param("~center_band_min_px", 30)       # low lim px
        # limits
        self.seg_min_frac         = rospy.get_param("~seg_min_frac", 0.30)  
        self.center_reject_margin = rospy.get_param("~center_reject_margin", 0.15) 

        # filter state
        self.center_filt  = None
        self.heading_filt = 0.0


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

    def publish_cmd(self, speed, steer, dt=None):
        if self.estop:
            speed = 0.0; steer = 0.0
        now = rospy.get_time()
        if dt is None:
            dt = max(1e-3, now - getattr(self, "_last_pub_t", now))
        acc = (speed - getattr(self, "_last_pub_speed", 0.0)) / dt
        jerk = (acc - getattr(self, "_last_pub_acc", 0.0)) / dt

        cmd = AckermannDrive()
        cmd.speed = float(speed)
        cmd.steering_angle = float(clamp(steer, -self.steer_limit, self.steer_limit))
        
        cmd.acceleration = float(clamp(acc, -self.a_brake_max, self.a_accel_max))
        cmd.jerk = float(jerk)

        self.pub_cmd.publish(cmd)
        self._last_pub_t = now
        self._last_pub_speed = float(speed)
        self._last_pub_acc = float(acc)


    def on_image(self, msg: Image):
    
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "cv_bridge: %s", e)
            return

        h, w = bgr.shape[:2]
        recovering = (rospy.Time.now() - self.last_ok_time).to_sec()*1000.0 >= self.hold_bad_ms
        roi_top_eff = self.roi_top_reco if recovering else self.roi_top
        y0 = int(h * roi_top_eff); y0 = min(max(y0, 0), h-2)
        roi = bgr[y0:h, :]

   
        mask_color = self._color_mask(roi)
        H_roi, W_roi = mask_color.shape[:2]
    

        forbid_band = (None, None)
        if self.center_band_mode != "off":
          
            if self.center_filt is not None:
                cx_est = float(self.center_filt)
            elif self.last_center_px is not None:
                cx_est = float(self.last_center_px)
            else:
                cx_est = 0.5 * W_roi

            if self.center_band_mode == "static":
                half = 0.5 * float(self.center_band_frac) * W_roi
            else:
                base_w = self.last_lane_w_px if self.last_lane_w_px is not None else (0.4 * W_roi)
                half = 0.5 * max(float(self.center_band_min_px), float(self.center_band_k) * float(base_w))
    
            cx0 = int(max(0, min(W_roi-1, cx_est - half)))
            cx1 = int(max(0, min(W_roi,   cx_est + half)))
            if cx1 > cx0:
                mask_color[:, cx0:cx1] = 0
                forbid_band = (cx0, cx1)

 
        try:
            centers, used_y, lane_w_px = self._scan_centers(mask_color, forbid_band)
        except TypeError:
        
            centers, used_y, lane_w_px = self._scan_centers(mask_color)

     
        used_fallback, lines_draw = False, None
        if len(centers) < self.min_valid_rows:
            center_h, lane_w_h, lines_draw = self._hough_center(roi)
            if center_h is not None:
                ys = mask_color.shape[0]
                centers = [center_h] * max(2, self.min_valid_rows)
                used_y  = [int(ys*0.85), int(ys*0.95)]
                lane_w_px = lane_w_h
                used_fallback = True

 
        if len(centers) < self.min_valid_rows:
            self._recovery_publish(mask_color, w, h, y0, "NO LANE")
            self.lost_frames += 1
            return
        else:
            self.lost_frames = 0

     
        img_center_px  = 0.5 * w
        lane_center_px = float(np.mean(centers[-max(1, len(centers)//2):]))
        if lane_w_px is not None and lane_w_px > 5:
            self.last_lane_w_px = 0.8 * self.last_lane_w_px + 0.2 * float(lane_w_px)

     
        if self.center_filt is None:
            self.center_filt = lane_center_px
        else:
            b = float(self.center_beta)  # доля нового
            self.center_filt = (1.0 - b) * self.center_filt + b * lane_center_px

      
        err = (self.center_filt - img_center_px) / (w * 0.5)
    
       
        ys = mask_color.shape[0]
        y_norm = (np.array(used_y, dtype=np.float32) / max(1.0, ys))                # 0..1 
        c_norm = (np.array(centers, dtype=np.float32) - img_center_px) / (w * 0.5)  # -1..1
        if y_norm.ptp() < 1e-6:
        slope_norm = 0.0
        else:
            z = np.polyfit(y_norm, c_norm, 1)
            slope_norm = float(z[0])
        heading_raw = math.atan(slope_norm)
        a = float(self.heading_beta)
        self.heading_filt = (1.0 - a) * self.heading_filt + a * heading_raw
        heading = self.heading_filt

      
        t  = rospy.get_time()
        dt = max(1e-3, t - self.prev_t)

        de   = (err - self.prev_err) / dt
        de_f = self.der_alpha * self.prev_de_f + (1.0 - self.der_alpha) * de

        delta_fb = -(self.kp*err + self.ki*self.int_err + self.kd*de_f + self.k_heading*heading)
        raw = delta_fb  

      
        max_step = self.steer_rate_limit * dt
        raw = clamp(raw, self.prev_delta - max_step, self.prev_delta + max_step)

       
        delta = clamp(raw, -self.steer_limit, self.steer_limit)

      
        if abs(delta) < self.steer_limit - 1e-6:
            self.int_err = (1.0 - self.i_leak) * self.int_err + err * dt
            self.int_err = clamp(self.int_err, -0.8, 0.8)

      
        self.prev_err   = err
        self.prev_de_f  = de_f
        self.prev_delta = delta
        self.prev_t     = t

     
        alpha = float(self.steer_alpha)
        new_delta = delta * self.steer_sign
        if alpha <= 0.0 or alpha >= 1.0:
            steer_cmd = new_delta
        else:
            self.steer_filt = (1.0 - alpha) * self.steer_filt + alpha * new_delta
            steer_cmd = self.steer_filt
        self.steer_filt = steer_cmd

   
        v_curve = 1.0 / (1.0 + self.k_curve_speed * abs(heading))
        v_steer = 1.0 - self.steer_slowdown * abs(self.steer_filt)
        v_scale = max(0.25, min(v_curve, v_steer))

        
        if abs(err) > self.err_slow_th:
            v_scale *= max(0.35, 1.0 - self.err_slow_k * (abs(err) - self.err_slow_th))

        
        L = max(0.2, float(self.wheelbase))
        kappa = abs(math.tan(self.steer_filt) / L)
        if kappa > 1e-4:
            v_lat_cap = math.sqrt(max(1e-6, self.a_lat_max / kappa))
        else:
            v_lat_cap = self.target_speed

        # desired spd
        v_des = min(self.target_speed * v_scale, v_lat_cap)
        v_des = clamp(v_des, self.min_speed, self.target_speed)

        # smooth ramp
        speed, dt_pub = self._ramped_speed(v_des, now=rospy.get_time())


        self.pub_err.publish(float(err))
        self.publish_cmd(speed, self.steer_filt, dt_pub)
        self.last_ok_time = rospy.Time.now()
        self.last_cmd.speed = float(speed)
        self.last_cmd.steering_angle = float(self.steer_filt)
        self.last_center_px = float(self.center_filt)

     
        dbg = cv2.cvtColor(mask_color, cv2.COLOR_GRAY2BGR)

     
        if forbid_band[0] is not None and forbid_band[1] is not None:
            cv2.rectangle(dbg,
                          (int(forbid_band[0]), 0),
                          (int(forbid_band[1]), dbg.shape[0]-1),
                          (0,128,255), 1)

      
        for y_scan, cx in zip(used_y, centers):
            cv2.line(dbg, (int(cx), y_scan), (int(cx), max(0, y_scan-30)), (0, 0, 255), 2)

      
        if used_fallback and lines_draw is not None:
            dbg = cv2.addWeighted(dbg, 1.0, lines_draw, 0.7, 0.0)

        
        cv2.line(dbg, (int(0.5*w), ys-5), (int(0.5*w), ys-55), (255, 0, 0), 1)

        txt = f"err={err:+.2f} de={de_f:+.2f} I={self.int_err:+.2f} hd={heading:+.2f} δ={self.steer_filt:+.2f} v={speed:.2f}"
        cv2.putText(dbg, txt, (10, dbg.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

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

    def _scan_centers(self, mask, forbid_band=(None, None)):

        H, W = mask.shape[:2]
        centers, used_y = [], []
        lane_w = None

        expect_w = float(self.last_lane_w_px) if self.last_lane_w_px is not None else None
        seg_min = max(6.0, (expect_w or 0.4*W) * float(self.seg_min_frac))  # px

        fb0, fb1 = forbid_band
        center_forbid_lo = int((0.5 - self.center_reject_margin) * W)
        center_forbid_hi = int((0.5 + self.center_reject_margin) * W)

        def _forbidden(cx):
            
            if fb0 is not None and fb1 is not None and fb0 <= cx <= fb1:
                return True
            if center_forbid_lo <= cx <= center_forbid_hi:
                return True
            return False

        for r in self.scan_rows:
            y = int(H * float(r))
            y = 0 if y < 0 else H-1 if y >= H else y
            row = (mask[y, :] > 0).astype(np.uint8)

       
            diff = np.diff(np.pad(row, (1,1), 'constant'))
            starts = np.where(diff == 1)[0]
            ends   = np.where(diff == -1)[0] - 1
            segs = []
            for s,e in zip(starts, ends):
                if e >= s:
                    wseg = e - s + 1
                    cx   = 0.5 * (s + e)
                    segs.append((s, e, wseg, cx))

           
            keep = []
            for (s,e,wseg,cx) in segs:
                if wseg < seg_min:
                    continue
                if _forbidden(cx):
                    continue
                keep.append((s,e,wseg,cx))

            
            left = right = None
            if keep:
                keep.sort(key=lambda t: t[3]) 
                left  = keep[0]
                right = keep[-1]
                if right[3] <= left[3] + 1.0:
                    
                    right = None

            
            if left is not None and right is not None:
                xl = 0.5*(left[0] + left[1])
                xr = 0.5*(right[0] + right[1])
                w_est = xr - xl
                if w_est >= max(self.min_lane_w, seg_min):
                    centers.append(0.5*(xl + xr)); used_y.append(y); lane_w = w_est
            elif (left is not None or right is not None) and expect_w is not None:
                
                if left is not None:
                    xl = 0.5*(left[0] + left[1]); xr = xl + expect_w
                else:
                    xr = 0.5*(right[0] + right[1]); xl = xr - expect_w
                centers.append(0.5*(xl + xr)); used_y.append(y); lane_w = expect_w

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
        now = rospy.get_time()
        v_des = 0.0 if self.stop_if_lost else self.recovery_speed
        speed, dt_pub = self._ramped_speed(v_des, now=now)

        steer = getattr(self.last_cmd, "steering_angle", 0.0)
        self.publish_cmd(speed, steer, dt_pub)

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
            
    def _ramped_speed(self, v_des, now=None):        
        if now is None:
            now = rospy.get_time()
        dt = max(1e-3, now - self.prev_t_speed)

        dv = float(v_des) - float(self.v_cmd)
        
        if dv >= 0.0:
            dv = min(dv, self.a_accel_max * dt)
        else:
            dv = max(dv, -self.a_brake_max * dt)

        self.v_cmd = clamp(self.v_cmd + dv, 0.0, self.target_speed)
        self.prev_t_speed = now
        return self.v_cmd, dt
            

if __name__ == "__main__":
    VisionLKA()
    rospy.spin()

