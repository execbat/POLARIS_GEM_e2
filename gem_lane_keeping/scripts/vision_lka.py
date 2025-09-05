#!/usr/bin/env python3
import rospy, cv2, numpy as np
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge
from std_msgs.msg import String

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

class VisionLKA:
    def __init__(self):
        rospy.init_node("vision_lka")
        self.bridge = CvBridge()

        # --- control params ---
        self.target_speed = rospy.get_param("~target_speed", 1.5)
        self.kp           = rospy.get_param("~kp", 0.020)
        self.kd           = rospy.get_param("~kd", 0.060)
        self.k_heading    = rospy.get_param("~k_heading", 0.012)
        self.steer_limit  = rospy.get_param("~steer_limit", 0.40)

        # --- color/mask ---
        # HSV-white 
        self.s_thresh     = rospy.get_param("~s_thresh", 100)
        self.v_thresh     = rospy.get_param("~v_thresh", 35)
        # HLS-white 
        self.hls_L_min    = rospy.get_param("~hls_L_min", 190)
        # LAB-yellow 
        self.lab_b_min    = rospy.get_param("~lab_b_min", 140)

        # --- scan geometry ---
        self.roi_top      = rospy.get_param("~roi_top", 0.58)               # top margin of the observation frame
        self.scan_rows    = rospy.get_param("~scan_rows", [0.35, 0.55, 0.70, 0.85])  # scan scales by height
        self.min_mask_px  = rospy.get_param("~min_mask_px", 150)            # min masked square
        self.min_lane_w   = rospy.get_param("~min_lane_width_px", 16)       # min wide in pixels
        self.min_valid_rows = rospy.get_param("~min_valid_rows", 1)         # min number of valid lines
        self.hold_bad_ms  = rospy.get_param("~hold_bad_ms", 400)            # hold last cmd
        
        self.lane_px_width = 120.0 # reminded 	pixel line width
        
        # auto-switch mux (if temporaryly lost valid observation)
        self.auto_switch_mux = rospy.get_param("~auto_switch_mux", True)
        self.lost_to_pp_s    = rospy.get_param("~lost_to_pp_s", 1.0)
        self.last_seen_ok    = rospy.Time(0)
        self.current_mode    = "vision_lka"
        self.mux_sel_pub     = rospy.Publisher("/gem/controller_mux/select", String, queue_size=1, latch=True)

        # --- state ---
        self.estop        = False
        self.prev_err     = 0.0
        self.prev_t       = rospy.get_time()
        self.last_ok_time = rospy.Time(0.0)
        self.last_cmd     = AckermannDrive()

        # --- I/O ---
        self.pub_cmd = rospy.Publisher("cmd", AckermannDrive, queue_size=10)
        self.pub_err = rospy.Publisher("lateral_error", Float32, queue_size=10)
        self.pub_dbg = rospy.Publisher("debug", Image, queue_size=1)
        rospy.Subscriber("image", Image, self.on_image, queue_size=1)
        rospy.Subscriber("/gem/safety/stop", Bool, self.on_stop, queue_size=1)

        rospy.loginfo("vision_lka params: ts=%.2f kp=%.3f kd=%.3f kh=%.3f steer=%.2f "
                      "HSV(S<=%d,V>=%d) HLS(L>=%d) LAB(b>=%d) roi=%.2f scans=%s "
                      "min_mask=%d min_w=%d min_rows=%d hold=%dms",
                      self.target_speed, self.kp, self.kd, self.k_heading, self.steer_limit,
                      self.s_thresh, self.v_thresh, self.hls_L_min, self.lab_b_min,
                      self.roi_top, str(self.scan_rows),
                      self.min_mask_px, self.min_lane_w, self.min_valid_rows, self.hold_bad_ms)

    # -------------------- callbacks --------------------

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

    def on_image(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "cv_bridge err: %s", e); return

        h, w = bgr.shape[:2]
        y0   = int(h * self.roi_top)
        y0   = min(max(0, y0), h-2)
        roi  = bgr[y0:h, :]

        # --- MASKS: white + yellow ---
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

        white_hsv = cv2.inRange(hsv, (0, 0, self.v_thresh), (179, self.s_thresh, 255))
        white_hls = cv2.inRange(hls[:, :, 1], self.hls_L_min, 255)
        yellow_lab = cv2.inRange(lab[:, :, 2], self.lab_b_min, 255)

        mask_bi = cv2.bitwise_or(white_hsv, white_hls)   # any white lines
        mask    = cv2.bitwise_or(mask_bi, yellow_lab)    # + yellow margin
        mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        mask    = cv2.medianBlur(mask, 5)

        ys = mask.shape[0]
        nz_total = int(np.count_nonzero(mask))
        if nz_total < self.min_mask_px:
            self._debug_and_stop(mask, w, h, y0, "NO LANE (area)")
            self._maybe_switch_mux(lost=True)
            return

        # --- polyline scan: trying to find both margins ---
        centers, used_y = [], []
        lane_widths = []
        for r in self.scan_rows:
            y_scan = int(ys * float(r))
            y_scan = np.clip(y_scan, 0, ys-1)
            scan = mask[y_scan, :]
            xs = np.where(scan > 0)[0]
            if xs.size >= self.min_lane_w:
                left_idx, right_idx = int(xs[0]), int(xs[-1])
                if (right_idx - left_idx) >= self.min_lane_w:
                    centers.append(0.5 * (left_idx + right_idx))
                    lane_widths.append(float(right_idx - left_idx))
                    used_y.append(int(y_scan))

        have_both_edges = (len(centers) >= self.min_valid_rows)
        lane_center_px = None
        heading = 0.0

        if have_both_edges:
            # cented by low lanes + update center line evaluation
            lane_center_px = float(np.mean(centers[-max(1, len(centers)//2):]))
            if len(lane_widths) > 0:
                lw = float(np.median(lane_widths))
                # smooth update
                self.lane_px_width = 0.9*self.lane_px_width + 0.1*lw

            # slope (heading) by regression
            z = np.polyfit(np.array(used_y, dtype=np.float32),
                       np.array(centers, dtype=np.float32), 1)
            slope = z[0]
            heading = float(slope) / (w*0.5)

        else:
            # --- FALLBACK: only LEFT (yellow) margin ---
            ys_idx, xs_idx = np.where(yellow_lab > 0)
            if xs_idx.size >= self.min_lane_w:
                # fit of the line x(y) = m*y + c по (y, x)
                fit = cv2.fitLine(np.column_stack([ys_idx.astype(np.float32),
                                                   xs_idx.astype(np.float32)]),
                                  cv2.DIST_L2, 0, 0.01, 0.01)
                vy, vx, y0f, x0f = fit.squeeze()
                # x of low margin
                y_query = float(ys-1)
                x_left  = x0f + (vx/vy)*(y_query - y0f) if abs(vy) > 1e-6 else x0f
                # cnter ≈ left margin + half of measured before
                lane_center_px = float(x_left + 0.5*self.lane_px_width)
                # heading left margin
                heading = float(vx/vy) / (w*0.5) if abs(vy) > 1e-6 else 0.0

        if lane_center_px is None:
            # observation is not valid -  soft stop
            self._debug_and_stop(mask, w, h, y0, "NO LANE (rows)")
            self._maybe_switch_mux(lost=True)
            return

        # --- error and PD ---
        img_center_px = 0.5 * w
        err   = (img_center_px - lane_center_px) / (w*0.5)

        t   = rospy.get_time()
        dt  = max(1e-3, t - self.prev_t)
        d_e = (err - self.prev_err) / dt
        self.prev_err, self.prev_t = err, t

        steer = self.kp*err + self.kd*d_e + self.k_heading*heading
        speed = self.target_speed

        # publications + "observation valid" -> remind the timestamp
        self.pub_err.publish(float(err))
        self.publish_cmd(speed, steer)
        self.last_ok_time = rospy.Time.now()

        # write debug
        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for y_scan, cx in zip(used_y, centers):
            cv2.line(dbg, (int(cx), y_scan), (int(cx), max(0, y_scan-30)), (0,0,255), 2)
        cv2.line(dbg, (int(img_center_px), ys-5), (int(img_center_px), ys-40), (255,0,0), 1)
        cv2.circle(dbg, (int(lane_center_px), ys-5), 5, (0,255,0), -1)
        txt = f"nz={nz_total} err={err:+.2f} de={d_e:+.2f} hd={heading:+.3f} st={steer:+.2f} v={speed:.2f} lw={self.lane_px_width:.1f}"
        cv2.putText(dbg, txt, (10, dbg.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        self._publish_debug(dbg, w, h, y0)

        # if observation is valid — change mode to vision_lka
        self._maybe_switch_mux(lost=False)



    # -------------------- helpers --------------------

    def _debug_and_stop(self, mask, w, h, y0, label):
        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(dbg, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        self._publish_debug(dbg, w, h, y0)
        

    def _publish_debug(self, roi_bgr, w, h, y0):
        canv = np.zeros((h, w, 3), dtype=np.uint8)
        canv[y0:h, :] = cv2.resize(roi_bgr, (w, h - y0))
        try:
            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(canv, encoding='bgr8'))
        except Exception as e:
            rospy.logwarn_throttle(2.0, "vision_lka: debug publish failed: %s", e)
            
    def _maybe_switch_mux(self, lost: bool):
        if not self.auto_switch_mux:
            return
        now = rospy.Time.now()
        if lost:
            # держим «потерю» чуть дольше, прежде чем уйти на PP
            lost_long = (now - self.last_ok_time).to_sec() > self.lost_to_pp_s
            if lost_long and self.current_mode != "pure_pursuit":
                self.mux_sel_pub.publish(String(data="pure_pursuit"))
                self.current_mode = "pure_pursuit"
        else:
            if self.current_mode != "vision_lka":
                self.mux_sel_pub.publish(String(data="vision_lka"))
                self.current_mode = "vision_lka"            

if __name__ == "__main__":
    VisionLKA()
    rospy.spin()

