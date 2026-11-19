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
