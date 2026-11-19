"""테스트용 합성 카메라. 실제 카메라 없이 '이 자세에서 찍었다면 나왔을 이미지'를 만든다.

핀홀 카메라를 로봇 앞쪽 위에 아래로 기울여 달았다고 가정하고,
지도 좌표에 그린 바닥 텍스처를 카메라 이미지로 렌더링한다.
"""

import math
from typing import NamedTuple, Tuple

import cv2
import numpy as np


class PinholeCamera(NamedTuple):
    K: np.ndarray          # 3x3
    width: int
    height: int
    cam_x: float           # base_footprint 기준 카메라 전방 위치 [m]
    cam_z: float           # 바닥에서 카메라 높이 [m]
    pitch: float           # 아래로 기울인 각 [rad]

    def R_robot_from_cam(self):
        # 광학 좌표(x 오른쪽, y 아래, z 전방) → 로봇 좌표(x 전방, y 왼쪽, z 위)
        R0 = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        c, s = math.cos(self.pitch), math.sin(self.pitch)
        Ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        return Ry @ R0

    def t(self):
        return np.array([self.cam_x, 0.0, self.cam_z])

    def floor_homography(self):
        """정답 H: 보정된 이미지 픽셀 → base_footprint 바닥 좌표."""
        Rt = self.R_robot_from_cam().T
        G = self.K @ np.column_stack([Rt[:, 0], Rt[:, 1], -Rt @ self.t()])
        H = np.linalg.inv(G)
        return H / H[2, 2]

    def project_robot_points(self, xy):
        """바닥점 (N, 2) → 픽셀 (N, 2)와 카메라 앞쪽 여부 (H를 쓰지 않는 독립 계산)."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        P = np.column_stack([xy, np.zeros(len(xy))])
        Pc = (P - self.t()) @ self.R_robot_from_cam()  # = R^T (P - t)
        in_front = Pc[:, 2] > 1e-9
        with np.errstate(divide="ignore", invalid="ignore"):
            uvw = Pc @ self.K.T
            uv = uvw[:, :2] / uvw[:, 2:3]
        return uv, in_front


def c920_like(width=640, height=480, cam_x=0.05, cam_z=0.20, pitch_deg=40.0):
    """C920 640x480 수준(수평 화각 약 65도)의 카메라."""
    f = (width / 2.0) / math.tan(math.radians(65.0 / 2.0))
    K = np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])
    return PinholeCamera(K, width, height, cam_x, cam_z, math.radians(pitch_deg))


def wide_camera(width=640, height=480, cam_x=0.0, cam_z=0.40, pitch_deg=60.0, f=230.0):
    """ROI 전체가 화면에 들어오는 넓은 화각 카메라 (유효 칸 개수 검사용)."""
    K = np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])
    return PinholeCamera(K, width, height, cam_x, cam_z, math.radians(pitch_deg))


class FloorTexture(NamedTuple):
    """지도 좌표에 그린 바닥 그림. image[row, col], row는 +y 방향 (바닥 격자와 같은 규칙)."""

    image: np.ndarray
    origin_x: float
    origin_y: float
    resolution: float


def render(camera, texture, pose):
    # type: (PinholeCamera, FloorTexture, Tuple[float, float, float]) -> np.ndarray
    """자세 pose(map→base_footprint)에서 카메라가 볼 이미지. 바닥이 아닌 부분(지평선 위)은 0."""
    u, v = np.meshgrid(np.arange(camera.width, dtype=np.float64), np.arange(camera.height, dtype=np.float64))
    # 픽셀 광선과 바닥의 교점을 직접 계산 (H를 쓰지 않는 독립 경로)
    Kinv = np.linalg.inv(camera.K)
    rays_c = np.stack([u.ravel(), v.ravel(), np.ones(u.size)], axis=1) @ Kinv.T
    rays_r = rays_c @ camera.R_robot_from_cam().T
    t = camera.t()
    with np.errstate(divide="ignore", invalid="ignore"):
        s = -t[2] / rays_r[:, 2]
    hit = s > 0
    xr = t[0] + s * rays_r[:, 0]
    yr = t[1] + s * rays_r[:, 1]
    x, y, yaw = pose
    c, sn = math.cos(yaw), math.sin(yaw)
    xm = x + c * xr - sn * yr
    ym = y + sn * xr + c * yr
    map_x = ((xm - texture.origin_x) / texture.resolution - 0.5).astype(np.float32)
    map_y = ((ym - texture.origin_y) / texture.resolution - 0.5).astype(np.float32)
    map_x[~hit] = -1e6
    map_y[~hit] = -1e6
    out = cv2.remap(texture.image, map_x.reshape(u.shape), map_y.reshape(u.shape),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out


# ---------------------------------------------------------------- 체커보드

def chessboard_image(pattern, square_px, margin_px):
    """흰 여백을 둔 체커보드 그림 (uint8). 내부 코너 (r, c)의 픽셀 위치 = margin + (c+1, r+1) * square_px."""
    cols, rows = pattern
    w = (cols + 1) * square_px + 2 * margin_px
    h = (rows + 1) * square_px + 2 * margin_px
    img = np.full((h, w), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y0, x0 = margin_px + r * square_px, margin_px + c * square_px
                img[y0:y0 + square_px, x0:x0 + square_px] = 30
    return img


def distort_image(ideal, K, D):
    """왜곡 없는 이미지 → 렌즈 왜곡 D가 있는 카메라로 찍은 이미지."""
    h, w = ideal.shape[:2]
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    pts = np.stack([u.ravel(), v.ravel()], axis=1).reshape(-1, 1, 2)
    src = cv2.undistortPoints(pts, K, D, P=K).reshape(h, w, 2).astype(np.float32)
    return cv2.remap(ideal, src[..., 0], src[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=128)


def board_view(K, D, size, rvec, tvec, pattern, square, square_px=40, margin_px=40, background=128):
    """보드 좌표(첫 내부 코너가 원점, z=0)를 rvec, tvec 자세로 찍은 이미지."""
    tex = chessboard_image(pattern, square_px, margin_px)
    s = square / square_px
    # 텍스처 픽셀 → 보드 좌표 [m] (첫 내부 코너 = 원점)
    T = np.array([[s, 0.0, -(margin_px + square_px) * s], [0.0, s, -(margin_px + square_px) * s], [0.0, 0.0, 1.0]])
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    G = K @ np.column_stack([R[:, 0], R[:, 1], np.asarray(tvec, dtype=np.float64).ravel()]) @ T
    w, h = size
    ideal = cv2.warpPerspective(tex, G, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=background)
    return distort_image(ideal, K, D) if D is not None and np.any(D) else ideal


def floor_board_texture(pattern, square, origin_xy, yaw=0.0, res=0.001, extent=((-0.2, 1.2), (-0.8, 0.8))):
    """바닥(로봇 좌표 = 지도 좌표, pose 0)에 놓인 체커보드 텍스처.

    기준 코너(order_floor_corners의 [0, 0])가 origin_xy, 가로 코너가 +y, 세로 코너가 +x.
    """
    (x0, x1), (y0, y1) = extent
    w, h = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    xs = x0 + (np.arange(w) + 0.5) * res
    ys = y0 + (np.arange(h) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = X - origin_xy[0], Y - origin_xy[1]
    bx = (c * dx + s * dy) / square      # 보드 세로 축(+x) 칸 단위
    by = (-s * dx + c * dy) / square     # 보드 가로 축(+y) 칸 단위
    cols, rows = pattern
    inside = (bx >= -1) & (bx < rows) & (by >= -1) & (by < cols)
    black = (np.floor(bx).astype(int) + np.floor(by).astype(int)) % 2 == 0
    margin = (bx >= -2) & (bx < rows + 1) & (by >= -2) & (by < cols + 1)
    img = np.full((h, w), 110, np.uint8)
    img[margin] = 255
    img[inside & black] = 30
    return FloorTexture(img, x0, y0, res)
