"""실시간 모니터 화면 합성 (scripts/live_monitor.py가 사용). ROS·카메라와 무관한 순수 함수.

화면 구성
  [카메라 640x480] [LiDAR 위에서 본 그림 480x480]
  [상태 줄: 카메라 fps, 스캔 Hz·점 수·가장 가까운 장애물, odom, 카메라 노출·초점 설정]

보정 파일(H)이 있으면 카메라 화면에 근거리 ROI와 'LiDAR 점의 바닥 위치'를 겹친다.
LiDAR 점은 높이 약 17 cm에서 잰 것이지만, 바닥 평면으로 투영하면 그 물체가 바닥에 닿는 지점이 된다.
의자 다리의 LiDAR 점이 화면 속 의자 다리 밑동에 겹치면 보정·좌표 변환이 맞다는 뜻이다.
"""

import math
import time
from collections import deque
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from floor_dirt.projection import _robot_to_pixel_checked, roi_polygon

CAM_SIZE = (640, 480)
LIDAR_SIZE = 480
BG = (24, 24, 24)
COL_RING = (70, 70, 70)
COL_POINT = (60, 60, 255)
COL_CLOSE = (0, 220, 255)
COL_ROI = (0, 140, 255)
COL_ROBOT = (230, 160, 40)
COL_TEXT = (235, 235, 235)


class RateMeter:
    """최근 window초 동안의 수신 주기 [Hz]."""

    def __init__(self, window=2.0, clock=time.monotonic):
        self.window = window
        self.clock = clock
        self.times = deque()
        self.last = None  # 마지막 수신 시각 (주기 계산용 기록이 지워져도 남김)

    def tick(self, t=None):
        t = self.clock() if t is None else t
        self.times.append(t)
        self.last = t
        while self.times and t - self.times[0] > self.window:
            self.times.popleft()

    def rate(self, now=None):
        now = self.clock() if now is None else now
        while self.times and now - self.times[0] > self.window:
            self.times.popleft()
        if len(self.times) < 2:
            return 0.0
        span = self.times[-1] - self.times[0]
        return (len(self.times) - 1) / span if span > 0 else 0.0

    def age(self, now=None):
        """마지막 수신 후 지난 시간 [s]. 받은 적 없으면 inf."""
        now = self.clock() if now is None else now
        return now - self.last if self.last is not None else float("inf")


# ---------------------------------------------------------------- LiDAR 그림

def _robot_to_top(xy, size, range_m):
    s = (size / 2.0) / range_m
    c = size / 2.0
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    return np.stack([c - xy[:, 1] * s, c - xy[:, 0] * s], axis=1)  # 전방(x) = 위, 왼쪽(y) = 왼쪽


def closest_point(points):
    # type: (np.ndarray) -> Optional[Tuple[float, float]]
    """가장 가까운 점의 (거리 [m], 방향 [deg], 왼쪽 +). 점이 없으면 None."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return None
    d = np.hypot(pts[:, 0], pts[:, 1])
    i = int(np.argmin(d))
    return float(d[i]), math.degrees(math.atan2(pts[i, 1], pts[i, 0]))


def lidar_topdown(points, size=LIDAR_SIZE, range_m=2.0, roi=None, robot_radius=0.15):
    # type: (Optional[np.ndarray], int, float, Optional[dict], float) -> np.ndarray
    """로봇 중심, 전방이 위인 LiDAR 그림. points = base_footprint 기준 (N, 2) [m]."""
    img = np.full((size, size, 3), BG, np.uint8)
    c = size // 2
    s = (size / 2.0) / range_m
    step = 0.5
    for k in range(1, int(range_m / step) + 1):
        r = k * step
        cv2.circle(img, (c, c), int(round(r * s)), COL_RING, 1, cv2.LINE_AA)
        if abs(r - round(r)) < 1e-9:
            cv2.putText(img, "{:.0f} m".format(r), (c + 4, c - int(round(r * s)) + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 130, 130), 1, cv2.LINE_AA)
    cv2.line(img, (c, 0), (c, size), COL_RING, 1)
    cv2.line(img, (0, c), (size, c), COL_RING, 1)
    if roi is not None:
        poly = np.round(_robot_to_top(roi_polygon(roi), size, range_m)).astype(np.int32)
        cv2.polylines(img, [poly.reshape(-1, 1, 2)], True, COL_ROI, 2, cv2.LINE_AA)
    cv2.circle(img, (c, c), max(int(round(robot_radius * s)), 3), COL_ROBOT, 2, cv2.LINE_AA)
    cv2.line(img, (c, c), (c, c - max(int(round(robot_radius * s)), 3)), COL_ROBOT, 2, cv2.LINE_AA)

    if points is not None and len(points):
        uv = np.round(_robot_to_top(points, size, range_m)).astype(np.int32)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < size) & (uv[:, 1] >= 0) & (uv[:, 1] < size)
        for u, v in uv[inside]:
            cv2.circle(img, (int(u), int(v)), 2, COL_POINT, -1)
        cl = closest_point(points)
        if cl is not None:
            i = int(np.argmin(np.hypot(points[:, 0], points[:, 1])))
            if inside[i]:
                cv2.circle(img, tuple(int(v) for v in uv[i]), 6, COL_CLOSE, 2, cv2.LINE_AA)
    cv2.putText(img, "LiDAR (top view, forward = up)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (180, 180, 180), 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------- 카메라 화면

def lidar_on_camera(H, points, img_shape):
    # type: (np.ndarray, np.ndarray, Tuple[int, int]) -> np.ndarray
    """LiDAR 점의 바닥 위치를 (보정된) 카메라 픽셀로. 카메라 앞쪽이면서 화면 안인 점만 (M, 2)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return np.zeros((0, 2))
    h, w = img_shape[:2]
    uv, front = _robot_to_pixel_checked(H, pts, w, h)
    ok = front & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv[ok]


def camera_panel(frame, H=None, roi=None, points=None):
    # type: (Optional[np.ndarray], Optional[np.ndarray], Optional[dict], Optional[np.ndarray]) -> np.ndarray
    """카메라 화면 (CAM_SIZE). frame이 없으면 'no camera'. H가 있으면 ROI·LiDAR 바닥 위치를 겹친다."""
    w, h = CAM_SIZE
    if frame is None:
        img = np.full((h, w, 3), BG, np.uint8)
        cv2.putText(img, "no camera", (w // 2 - 70, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 120, 120), 2)
        return img
    img = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    sx, sy = w / img.shape[1], h / img.shape[0]
    src_shape = img.shape
    if (img.shape[1], img.shape[0]) != (w, h):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    else:
        img = img.copy()
    if H is not None:
        if roi is not None:
            uv, front = _robot_to_pixel_checked(H, roi_polygon(roi), src_shape[1], src_shape[0])
            if front.all():
                poly = np.round(uv * [sx, sy]).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [poly], True, COL_ROI, 2, cv2.LINE_AA)
        if points is not None and len(points):
            for u, v in lidar_on_camera(H, points, src_shape) * [sx, sy]:
                cv2.circle(img, (int(round(u)), int(round(v))), 4, COL_POINT, -1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------- 상태 줄

OPENCR_LOW_V = 11.1     # OpenCR 배터리 LED가 켜지는 전압
OPENCR_ALARM_V = 11.0   # OpenCR 경고음이 울리는 전압


def battery_percent(voltage, empty=10.5, full=12.3):
    # type: (float, float, float) -> float
    """3셀 배터리 전압 → 대략 %. OpenCR 펌웨어와 같은 직선 (셀당 3.5 V = 0%, 4.1 V = 100%)."""
    return float(min(max((voltage - empty) / (full - empty) * 100.0, 0.0), 100.0))


def power_line(opencr=None):
    # type: (Optional[dict]) -> str
    """전원 줄. opencr = {"voltage", "percentage"(0~100 또는 None)}."""
    if opencr is None:
        return "OpenCR bat --"
    v = opencr["voltage"]
    pct = opencr.get("percentage")
    line = "OpenCR bat {:.2f} V".format(v)
    if pct is not None:
        line += " ({:.0f}%)".format(pct)
    if v < OPENCR_ALARM_V:
        line += " CHARGE NOW"
    elif v < OPENCR_LOW_V:
        line += " LOW"
    return line


def status_lines(cam_fps=None, scan_hz=None, n_points=0, closest=None, odom=None, cam_ctrl=None, scan_age=None,
                 calibrated=False, power=None):
    # type: (Optional[float], Optional[float], int, Optional[Tuple[float, float]], Optional[dict], Optional[dict], Optional[float], bool, Optional[str]) -> list
    """화면 아래 상태 줄 (cv2.putText는 한글이 안 되므로 영어). power = power_line() 결과."""
    cam = "cam --" if cam_fps is None else "cam {:4.1f} fps".format(cam_fps)
    if scan_hz is None:
        scan = "scan off"
    elif scan_age is not None and scan_age > 1.0:
        scan = "scan NO DATA ({:.0f} s)".format(min(scan_age, 999))
    else:
        scan = "scan {:4.1f} Hz {} pts".format(scan_hz, n_points)
    if closest is not None:
        scan += " | closest {:.2f} m @ {:+.0f} deg".format(*closest)
    line1 = "{} | {}{}".format(cam, scan, " | calib ON" if calibrated else " | calib off")
    if odom is None:
        line2 = "odom --"
    else:
        line2 = "odom x {:+.2f} y {:+.2f} yaw {:+6.1f} deg | v {:+.2f} m/s w {:+5.1f} deg/s".format(
            odom["x"], odom["y"], math.degrees(odom["yaw"]), odom["v"], math.degrees(odom["w"]))
    lines = [line1, line2]
    if power:
        lines.append(power)
    if cam_ctrl:
        lines.append(" | ".join("{} {}".format(k, v) for k, v in cam_ctrl.items()))
    return lines


def compose(cam_img, lidar_img, lines):
    # type: (np.ndarray, np.ndarray, Sequence[str]) -> np.ndarray
    """카메라 + LiDAR를 나란히, 아래에 상태 줄."""
    h = max(cam_img.shape[0], lidar_img.shape[0])
    top = np.full((h, cam_img.shape[1] + lidar_img.shape[1], 3), BG, np.uint8)
    top[:cam_img.shape[0], :cam_img.shape[1]] = cam_img
    top[:lidar_img.shape[0], cam_img.shape[1]:] = lidar_img
    bar = np.full((10 + 22 * max(len(lines), 1), top.shape[1], 3), (40, 40, 40), np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(bar, line, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_TEXT, 1, cv2.LINE_AA)
    return np.vstack([top, bar])


# ---------------------------------------------------------------- ROS 메시지 변환 (rclpy 없이 테스트 가능)

def image_msg_to_bgr(encoding, height, width, step, data):
    # type: (str, int, int, int, bytes) -> np.ndarray
    """sensor_msgs/Image → BGR. cv_bridge 없이 처리 (conda 환경에서 cv_bridge가 안 될 수 있어서)."""
    buf = np.frombuffer(bytes(data), np.uint8)
    enc = encoding.lower()
    if enc in ("bgr8", "rgb8"):
        img = buf.reshape(height, step)[:, :width * 3].reshape(height, width, 3)
        return img.copy() if enc == "bgr8" else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if enc == "mono8":
        return cv2.cvtColor(buf.reshape(height, step)[:, :width].copy(), cv2.COLOR_GRAY2BGR)
    if enc in ("yuv422", "uyvy"):  # ROS 'yuv422' = UYVY
        return cv2.cvtColor(buf.reshape(height, step)[:, :width * 2].reshape(height, width, 2), cv2.COLOR_YUV2BGR_UYVY)
    if enc in ("yuv422_yuy2", "yuyv", "yuy2"):
        return cv2.cvtColor(buf.reshape(height, step)[:, :width * 2].reshape(height, width, 2), cv2.COLOR_YUV2BGR_YUY2)
    raise ValueError("지원하지 않는 이미지 인코딩: {}".format(encoding))


def quaternion_to_yaw(x, y, z, w):
    # type: (float, float, float, float) -> float
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# ---------------------------------------------------------------- 계기판 (상태 줄을 그림으로)

C_OK = (90, 200, 90)       # 초록
C_WARN = (0, 190, 255)     # 노랑 (BGR)
C_BAD = (70, 70, 235)      # 빨강
C_OFF = (110, 110, 110)    # 회색 (꺼짐·모름)
C_PANEL = (36, 36, 36)
C_DIM = (150, 150, 150)

WAFFLE_MAX_V = 0.26        # Waffle Pi 최대 직진 속도 [m/s]
WAFFLE_MAX_W = 1.82        # 최대 회전 속도 [rad/s]
DASH_H = 132


def _text(img, s, org, scale=0.45, color=COL_TEXT, thick=1):
    cv2.putText(img, s, (int(org[0]), int(org[1])), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def level(value, good, warn, higher_is_better=True):
    """값 → 신호등 색. value None이면 회색."""
    if value is None:
        return C_OFF
    if higher_is_better:
        return C_OK if value >= good else C_WARN if value >= warn else C_BAD
    return C_OK if value <= good else C_WARN if value <= warn else C_BAD


def _light(img, x, y, color, name, value):
    cv2.circle(img, (x + 9, y), 8, color, -1, cv2.LINE_AA)
    _text(img, name, (x + 24, y + 5), 0.5, COL_TEXT)
    _text(img, value, (x + 92, y + 5), 0.45, color if color != C_OFF else C_DIM)


def _hbar(img, x, y, w, h, frac, color, centered=False):
    """막대. centered면 가운데 0에서 frac(-1~1)만큼 좌우로."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (70, 70, 70), 1)
    f = float(np.clip(frac, -1.0, 1.0)) if frac is not None else 0.0
    if centered:
        cx = x + w // 2
        x1 = cx + int(round(f * (w // 2 - 2)))
        if x1 != cx:
            cv2.rectangle(img, (min(cx, x1), y + 2), (max(cx, x1), y + h - 2), color, -1)
        cv2.line(img, (cx, y - 3), (cx, y + h + 3), C_DIM, 1)
    elif f > 0:
        cv2.rectangle(img, (x + 2, y + 2), (x + 2 + int(round(f * (w - 4))), y + h - 2), color, -1)


def _battery_icon(img, x, y, w, h, pct, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), C_DIM, 2)
    cv2.rectangle(img, (x + w, y + h // 3), (x + w + 5, y + 2 * h // 3), C_DIM, -1)
    if pct is not None:
        fill = int(round((w - 6) * max(min(pct, 100.0), 0.0) / 100.0))
        if fill > 0:
            cv2.rectangle(img, (x + 3, y + 3), (x + 3 + fill, y + h - 3), color, -1)


def dashboard(width, st):
    # type: (int, dict) -> np.ndarray
    """상태 dict → 계기판 그림 (width x DASH_H).

    st 키 (없으면 회색 '--'):
      cam_fps, cam_on, scan_hz, scan_age, ros_on, n_points, closest (dist, deg),
      odom {x, y, yaw, v, w}, odom_age, battery {voltage, percentage},
      cam_ctrl {AE, AF, AWB, ...}, calibrated
    """
    img = np.full((DASH_H, width, 3), C_PANEL, np.uint8)
    secs = [0, int(width * 0.26), int(width * 0.50), int(width * 0.76), width]
    for xs in secs[1:-1]:
        cv2.line(img, (xs, 8), (xs, DASH_H - 8), (60, 60, 60), 1)

    # 1) 연결 상태 신호등
    x = secs[0] + 14
    _text(img, "SENSORS", (x, 20), 0.45, C_DIM)
    fps = st.get("cam_fps")
    cam_c = level(fps, 20, 1) if st.get("cam_on") else C_OFF
    _light(img, x, 40, cam_c, "CAM", "{:.0f} fps".format(fps) if (st.get("cam_on") and fps is not None) else "off")
    if not st.get("ros_on"):
        lid_c, lid_v = C_OFF, "off"
    elif st.get("scan_age") is None or st["scan_age"] > 1.0:
        lid_c, lid_v = C_BAD, "NO DATA"
    else:
        hz = st.get("scan_hz") or 0.0
        lid_c, lid_v = level(hz, 4, 1), "{:.1f} Hz {} pts".format(hz, st.get("n_points", 0))
    _light(img, x, 64, lid_c, "LiDAR", lid_v)
    if not st.get("ros_on"):
        od_c, od_v = C_OFF, "off"
    elif st.get("odom") is None or (st.get("odom_age") is not None and st["odom_age"] > 1.0):
        od_c, od_v = C_BAD, "NO DATA"
    else:
        od_c, od_v = C_OK, "OK"
    _light(img, x, 88, od_c, "ODOM", od_v)
    _light(img, x, 112, C_OK if st.get("calibrated") else C_OFF, "CALIB", "ON" if st.get("calibrated") else "off")

    # 2) 전원
    x = secs[1] + 14
    _text(img, "POWER", (x, 20), 0.45, C_DIM)
    bat = st.get("battery")
    v = None if bat is None else bat["voltage"]
    pct = None if bat is None else bat.get("percentage")
    bc = level(v, 11.5, 11.1)
    _battery_icon(img, x, 32, 70, 26, pct, bc)
    _text(img, "OpenCR", (x + 86, 42), 0.45, C_DIM)
    _text(img, "--" if v is None else "{:.2f} V  {:.0f}%".format(v, pct if pct is not None else 0), (x + 86, 60), 0.5,
          bc if v is not None else C_DIM)
    if v is not None and v < 11.0:
        _text(img, "CHARGE NOW", (x + 86, 78), 0.5, C_BAD, 2)

    # 3) 움직임
    x = secs[2] + 14
    w_bar = secs[3] - secs[2] - 28
    _text(img, "MOTION", (x, 20), 0.45, C_DIM)
    od = st.get("odom")
    lin = None if od is None else od["v"]
    ang = None if od is None else od["w"]
    _text(img, "speed" + ("" if lin is None else "  {:+.2f} m/s".format(lin)), (x, 40), 0.45, COL_TEXT)
    _hbar(img, x, 46, w_bar, 14, None if lin is None else lin / WAFFLE_MAX_V, (230, 160, 40), centered=True)
    _text(img, "turn" + ("" if ang is None else "  {:+.0f} deg/s".format(math.degrees(ang))), (x, 80), 0.45, COL_TEXT)
    # 왼쪽 회전(+)이 왼쪽으로 차오르게 부호를 뒤집는다
    _hbar(img, x, 86, w_bar, 14, None if ang is None else -ang / WAFFLE_MAX_W, (230, 160, 40), centered=True)
    if od is not None:
        _text(img, "x {:+.2f}  y {:+.2f}  yaw {:+.0f} deg".format(od["x"], od["y"], math.degrees(od["yaw"])),
              (x, 122), 0.42, C_DIM)

    # 4) 장애물 + 카메라 설정
    x = secs[3] + 14
    w_bar = secs[4] - secs[3] - 28
    _text(img, "NEAREST OBSTACLE", (x, 20), 0.45, C_DIM)
    cl = st.get("closest")
    if cl is None:
        _text(img, "--", (x, 44), 0.5, C_DIM)
        _hbar(img, x, 50, w_bar, 14, 0.0, C_OFF)
    else:
        d, deg = cl
        c = level(d, 0.5, 0.25)
        side = "front" if abs(deg) <= 45 else "back" if abs(deg) >= 135 else ("left" if deg > 0 else "right")
        _text(img, "{:.2f} m  {}".format(d, side), (x, 44), 0.55, c, 2)
        _hbar(img, x, 50, w_bar, 14, max(1.0 - d / 2.0, 0.0), c)   # 가까울수록 차오름 (2 m 이상 = 빈 막대)
    _text(img, "far", (x, 76), 0.38, C_DIM)
    _text(img, "near", (x + w_bar - 30, 76), 0.38, C_DIM)
    _text(img, "CAMERA", (x, 98), 0.45, C_DIM)
    ctrl = st.get("cam_ctrl") or {}
    cx = x
    for key, fixed in (("AE", "MANUAL"), ("AF", "OFF"), ("AWB", "OFF")):
        val = ctrl.get(key)
        c = C_OFF if val is None else (C_OK if val == fixed else C_WARN)
        label = "{} {}".format(key, "--" if val is None else ("fixed" if val == fixed else "auto"))
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0] + 12
        cv2.rectangle(img, (cx, 104), (cx + tw, 126), c, -1 if val is not None else 1, cv2.LINE_AA)
        _text(img, label, (cx + 6, 120), 0.42, (20, 20, 20) if val is not None else C_DIM)
        cx += tw + 6
    return img


def compose_dashboard(cam_img, lidar_img, st):
    # type: (np.ndarray, np.ndarray, dict) -> np.ndarray
    """카메라 + LiDAR를 나란히, 아래에 계기판."""
    h = max(cam_img.shape[0], lidar_img.shape[0])
    top = np.full((h, cam_img.shape[1] + lidar_img.shape[1], 3), BG, np.uint8)
    top[:cam_img.shape[0], :cam_img.shape[1]] = cam_img
    top[:lidar_img.shape[0], cam_img.shape[1]:] = lidar_img
    return np.vstack([top, dashboard(top.shape[1], st)])
