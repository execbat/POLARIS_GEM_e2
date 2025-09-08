#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy, cv2, numpy as np, math
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge

def clamp(v, lo, hi): 
    return lo if v < lo else hi if v > hi else v

class VisionLKA:
    """
    Простой и надёжный LKA:
      1) ROI → HSV/HLS → маски yellow/white → морфология.
      2) По ряду y ищем левую/правую ЖЁЛТУЮ кромку (центр белой игнорим полосой).
      3) Центр = среднее(left,right) с EMA; ширина полосы с EMA; heading из наклона центра.
      4) Управление: delta = -(Kp*e + Kd*de + Kh*heading), ограничение d(delta)/dt, EMA руля.
      5) Скорость: таргет с замедлением от |heading| и |delta|, плюс лимиты по a_lat и dv/dt.
    """

    # ---------- init ----------
    def __init__(self):
        rospy.init_node("vision_lka")
        self.bridge = CvBridge()

        # --- PD + heading ---
        self.kp = rospy.get_param("~kp", 0.035)
        self.kd = rospy.get_param("~kd", 0.10)
        self.kh = rospy.get_param("~k_heading", 0.12)
        self.der_alpha = rospy.get_param("~der_alpha", 0.7)   # EMA деривативы (0..1)

        # --- рулёжка ---
        self.steer_limit      = rospy.get_param("~steer_limit", 0.90)
        self.steer_rate_limit = rospy.get_param("~steer_rate_limit", 2.0)  # рад/с
        self.steer_alpha      = rospy.get_param("~steer_alpha", 0.15)      # EMA руля (0..1)
        self.steer_sign       = rospy.get_param("~steer_sign", -1.0)       # если инверт рулевого

        # --- скорость/ограничения ---
        self.target_speed  = rospy.get_param("~target_speed", 1.6)
        self.min_speed     = rospy.get_param("~min_speed", 0.6)
        self.reco_speed    = rospy.get_param("~recovery_speed", 0.6)
        self.k_curve_speed = rospy.get_param("~k_curve_speed", 7.0)  # чем больше, тем сильней замедл.
        self.steer_slowdown= rospy.get_param("~steer_slowdown", 0.18)

        self.wheelbase   = rospy.get_param("~wheelbase", 1.2)     # м
        self.a_acc_max   = rospy.get_param("~a_accel_max", 0.6)   # м/с^2
        self.a_brake_max = rospy.get_param("~a_brake_max", 1.5)   # м/с^2
        self.a_lat_max   = rospy.get_param("~a_lat_max", 1.8)     # м/с^2

        # --- ROI и сканы ---
        self.roi_top     = rospy.get_param("~roi_top", 0.62)      # доля высоты от верха
        self.scan_rows   = rospy.get_param("~scan_rows", [0.62, 0.72, 0.82, 0.92])
        self.center_ignore_frac = rospy.get_param("~center_ignore_frac", 0.12)  # блок вокруг центра (белая прерывистая)
        self.min_valid_rows = rospy.get_param("~min_valid_rows", 2)

        # --- пороги и морфология ---
        # yellow (HSV)
        self.yellow_h_lo = rospy.get_param("~yellow_h_lo", 15)
        self.yellow_h_hi = rospy.get_param("~yellow_h_hi", 40)
        self.yellow_s_min= rospy.get_param("~yellow_s_min", 70)
        self.yellow_v_min= rospy.get_param("~yellow_v_min", 70)
        # white (HSV/HLS)
        self.s_thresh    = rospy.get_param("~s_thresh", 110)      # upper S threshold for white
        self.v_thresh    = rospy.get_param("~v_thresh", 35)       # lower V threshold for white
        self.hls_L_min   = rospy.get_param("~hls_L_min", 190)

        self.kernel_close = np.ones((5,5), np.uint8)
        self.kernel_open  = np.ones((3,3), np.uint8)

        # --- ширина полосы (пикс) ---
        self.lw_min = rospy.get_param("~lane_w_min_px", 90)
        self.lw_max = rospy.get_param("~lane_w_max_px", 220)
        self.lw_ema = rospy.get_param("~lane_w_ema", 0.15)  # EMA для ширины
        self.last_lane_w_px = None

        # --- состояния ---
        self.prev_t = rospy.get_time()
        self.prev_err = 0.0
        self.prev_de_f = 0.0
        self.prev_delta = 0.0
        self.steer_filt = 0.0
        self.center_px_ema = None
        self.center_alpha = rospy.get_param("~lane_center_alpha", 0.25)

        self.last_cmd = AckermannDrive()
        self.estop = False
        self.hold_bad_ms = rospy.get_param("~hold_bad_ms", 600)
        self.stop_if_lost = rospy.get_param("~stop_if_lost", False)
        self.last_ok_time = rospy.Time(0.0)
        self.v_cmd = 0.0
        self.prev_t_speed = rospy.get_time()

        # --- IO ---
        self.pub_cmd = rospy.Publisher("cmd", AckermannDrive, queue_size=10)
        self.pub_err = rospy.Publisher("lateral_error", Float32, queue_size=10)
        self.pub_dbg = rospy.Publisher("debug", Image, queue_size=1)
        rospy.Subscriber("image", Image, self.on_image, queue_size=1)
        rospy.Subscriber("/gem/safety/stop", Bool, self.on_stop, queue_size=1)

        rospy.loginfo("vision_lka: KP=%.3f KD=%.3f KH=%.3f steer_alpha=%.2f", 
                      self.kp, self.kd, self.kh, self.steer_alpha)

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
        # 0) BGR + ROI
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "cv_bridge: %s", e)
            return
        h, w = bgr.shape[:2]

        y0 = int(h * self.roi_top); y0 = min(max(y0, 0), h-2)
        roi = bgr[y0:h, :]
        H, W = roi.shape[:2]

        # 1) Маски цветов
        mask_yellow, mask_white = self._color_masks(roi)

        # 2) Игнор центральной полосы для поиска кромок
        cx = int(0.5 * W)
        ignore = int(W * self.center_ignore_frac)
        xL_blk, xR_blk = max(0, cx - ignore), min(W, cx + ignore)
        mask_edges = mask_yellow.copy()
        if xR_blk > xL_blk:
            mask_edges[:, xL_blk:xR_blk] = 0

        # 3) Поиск кромок на скан-рядках
        centers, used_y, lane_w = self._scan_centers(mask_edges, W, H, cx, ignore)

        if len(centers) < self.min_valid_rows:
            self._recovery(mask_white | mask_yellow, w, h, y0, label="NO LANE")
            return

        # 4) Центр + ширина с EMA
        lane_center_px = float(np.median(centers[-max(2, len(centers)):]))

        if lane_w is not None:
            if self.last_lane_w_px is None:
                self.last_lane_w_px = float(lane_w)
            else:
                a = self.lw_ema
                self.last_lane_w_px = (1.0 - a)*self.last_lane_w_px + a*float(lane_w)

        if self.center_px_ema is None:
            self.center_px_ema = lane_center_px
        else:
            a = self.center_alpha
            self.center_px_ema = (1.0 - a)*self.center_px_ema + a*lane_center_px
        lane_center_px = self.center_px_ema

        # 5) Ошибки: поперечная (норм.) и heading (наклон центра по y)
        img_center_px = 0.5 * w
        err = (lane_center_px - img_center_px) / (w * 0.5)

        ys = np.asarray(used_y, dtype=np.float32)
        cs = np.asarray(centers, dtype=np.float32)
        if ys.size >= 2 and ys.max() - ys.min() > 1e-3:
            # нормируем x в [-1,1] (от центра экрана), y в [0,1]
            c_norm = (cs - (0.5*w)) / (w*0.5)
            y_norm = ys / max(1.0, H)
            k = np.polyfit(y_norm, c_norm, 1)[0]
        else:
            k = 0.0
        heading = math.atan(float(k))

        # 6) PD + ограничение скорости поворота руля + EMA
        t  = rospy.get_time(); dt = max(1e-3, t - self.prev_t)
        de = (err - self.prev_err) / dt
        de_f = self.der_alpha * self.prev_de_f + (1.0 - self.der_alpha) * de

        delta = -(self.kp*err + self.kd*de_f + self.kh*heading)
        max_step = self.steer_rate_limit * dt
        delta = clamp(delta, self.prev_delta - max_step, self.prev_delta + max_step)
        delta = clamp(delta, -self.steer_limit, self.steer_limit)

        self.prev_err = err
        self.prev_de_f = de_f
        self.prev_delta = delta
        self.prev_t = t

        alpha = self.steer_alpha
        new_delta = delta * self.steer_sign
        self.steer_filt = (1.0 - alpha)*self.steer_filt + alpha*new_delta

        # 7) Скорость: базовая * (кривая по heading и по рулю) + лимиты
        v_curve = 1.0 / (1.0 + self.k_curve_speed * abs(heading))
        v_steer = 1.0 - self.steer_slowdown * abs(self.steer_filt)
        v_scale = max(0.25, min(v_curve, v_steer))
        v_des = clamp(self.target_speed * v_scale, self.min_speed, self.target_speed)

        # ограничение по попер. ускорению и dv/dt
        v_des = self._limit_by_lateral(v_des, self.steer_filt)
        speed, dt_pub = self._ramped_speed(v_des)

        # 8) Публикация
        self.pub_err.publish(float(err))
        self.publish_cmd(speed, self.steer_filt, dt_pub)
        self.last_ok_time = rospy.Time.now()
        self.last_cmd.speed = float(speed)
        self.last_cmd.steering_angle = float(self.steer_filt)

        # 9) Дебаг-оверлей
        dbg = self._draw_debug(roi, mask_edges, mask_white, centers, used_y, lane_center_px, xL_blk, xR_blk, w)
        self._publish_debug(dbg, w, h, y0)

    # ---------- detectors ----------
    def _color_masks(self, roi_bgr):
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        hls = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HLS)

        # white (широкий порог по V low + L high)
        white_hsv  = cv2.inRange(hsv, (0, 0, self.v_thresh), (179, self.s_thresh, 255))
        white_hls  = cv2.inRange(hls[:,:,1], self.hls_L_min, 255)
        mask_white = cv2.bitwise_or(white_hsv, white_hls)

        # yellow
        y_lo = (self.yellow_h_lo, self.yellow_s_min, self.yellow_v_min)
        y_hi = (self.yellow_h_hi, 255, 255)
        mask_yellow = cv2.inRange(hsv, y_lo, y_hi)

        # морфология
        mask_white  = cv2.morphologyEx(mask_white,  cv2.MORPH_CLOSE, self.kernel_close)
        mask_white  = cv2.morphologyEx(mask_white,  cv2.MORPH_OPEN,  self.kernel_open)
        mask_white  = cv2.medianBlur(mask_white, 5)

        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, self.kernel_close)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN,  self.kernel_open)
        mask_yellow = cv2.medianBlur(mask_yellow, 5)

        return mask_yellow, mask_white

    def _scan_centers(self, mask_edges, W, H, cx, ignore):
        centers, used_y = [], []
        lane_w = None

        xL_max = cx - ignore
        xR_min = cx + ignore

        for r in self.scan_rows:
            y = int(H * float(r))
            y = 0 if y < 0 else H-1 if y >= H else y
            row = mask_edges[y, :]

            # идём от центра к краям — ищем первую жёлтую слева и справа
            left_idx = None
            for x in range(xL_max-1, -1, -1):
                if row[x] > 0: left_idx = x; break

            right_idx = None
            for x in range(xR_min, W):
                if row[x] > 0: right_idx = x; break

            if left_idx is not None and right_idx is not None:
                wpx = right_idx - left_idx
                if self.lw_min <= wpx <= self.lw_max:
                    centers.append(0.5*(left_idx + right_idx))
                    used_y.append(y)
                    lane_w = wpx
            else:
                # частичный вид: используем прошлую ширину
                if self.last_lane_w_px is not None:
                    if left_idx is not None:
                        centers.append(left_idx + 0.5*self.last_lane_w_px); used_y.append(y)
                    elif right_idx is not None:
                        centers.append(right_idx - 0.5*self.last_lane_w_px); used_y.append(y)

        return centers, used_y, lane_w

    # ---------- speed limiting ----------
    def _limit_by_lateral(self, v_des, delta):
        # a_lat ≈ v^2 / L * |tan(delta)|  → ограничим v
        tan_d = abs(math.tan(delta))
        if tan_d > 1e-4:
            v_lat_max = math.sqrt(max(0.0, self.a_lat_max * self.wheelbase / tan_d))
            v_des = min(v_des, v_lat_max)
        return v_des

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

    # ---------- recovery & debug ----------
    def _recovery(self, mask, w, h, y0, label):
        now = rospy.get_time()
        v_des = 0.0 if self.stop_if_lost else self.reco_speed
        speed, dt_pub = self._ramped_speed(v_des, now=now)

        steer = getattr(self.last_cmd, "steering_angle", 0.0)
        self.publish_cmd(speed, steer, dt_pub)

        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(dbg, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        self._publish_debug(dbg, w, h, y0)

    def _draw_debug(self, roi_bgr, mask_edges, mask_white, centers, used_y, lane_center_px, xL_blk, xR_blk, w_full):
        dbg = roi_bgr.copy()
        # heatmap маски
        edges_bgr = cv2.cvtColor(mask_edges, cv2.COLOR_GRAY2BGR)
        white_bgr = cv2.cvtColor(mask_white,  cv2.COLOR_GRAY2BGR)
        mix = cv2.addWeighted(dbg, 0.7, edges_bgr, 0.4, 0.0)
        mix = cv2.addWeighted(mix, 1.0, white_bgr, 0.25, 0.0)

        # игнорируемая центральная зона
        cv2.rectangle(mix, (xL_blk, 0), (xR_blk, mix.shape[0]-1), (0, 60, 0), 2)

        # центры по рядам
        for y_scan, cx in zip(used_y, centers):
            cv2.circle(mix, (int(cx), int(y_scan)), 4, (0, 0, 255), -1)

        # финальный центр (на нижней части)
        cv2.line(mix, (int(0.5*w_full), mix.shape[0]-5), (int(0.5*w_full), mix.shape[0]-55), (255, 0, 0), 2)
        cv2.line(mix, (int(lane_center_px), mix.shape[0]-5), (int(lane_center_px), mix.shape[0]-55), (0, 255, 255), 2)

        txt = f"delta={self.steer_filt:+.2f} v={getattr(self.last_cmd,'speed',0.0):.2f}"
        cv2.putText(mix, txt, (10, mix.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1, cv2.LINE_AA)
        return mix

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

