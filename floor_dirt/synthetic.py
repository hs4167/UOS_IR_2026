"""합성 카메라·장면 (테스트와 WSL 확인용 make_synthetic_run.py가 사용). 실제 카메라 없이 '이 자세에서 찍었다면 나왔을 이미지'를 만든다.

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


# ---------------------------------------------------------------- 합성 강의실 (E2E)

class Item(NamedTuple):
    name: str
    x: float
    y: float
    w: float      # x 방향 크기 [m]
    h: float      # y 방향 크기 [m]
    color: tuple  # BGR


class Scene(NamedTuple):
    clean: FloorTexture        # 기준 주행 때 바닥 (청소 직후)
    dirty: FloorTexture        # 정찰 주행 때 바닥 (오염 + 물체 + 조명 변화)
    dirt: list                 # 찾아야 할 오염 (Item)
    bag: Item                  # LiDAR에 걸리는 물체
    shoes: Item                # detectNet이 'person'으로 찾는 물체 (보라색)
    poses: list                # 주행 자세 (참값)


SHOE_COLOR = (200, 30, 200)


def _paint(img, res, item, origin=(0.0, 0.0)):
    x0, x1 = int((item.x - item.w / 2 - origin[0]) / res), int((item.x + item.w / 2 - origin[0]) / res)
    y0, y1 = int((item.y - item.h / 2 - origin[1]) / res), int((item.y + item.h / 2 - origin[1]) / res)
    img[y0:y1, x0:x1] = item.color


def classroom_scene(seed=0, light_gain=1.10, res=0.002, size=(3.6, 3.0), origin=(-0.5, 0.0)):
    """30 cm 타일(2 mm 이음새, 타일마다 색 조금씩 다름) + 오래된 얼룩 + 미세 무늬 바닥.

    바닥 그림은 지도 x -0.5~3.1, y 0~3 m. 주행 중 ROI가 바닥 그림 밖으로 나가지 않게 넉넉히 둔다.
    """
    rng = np.random.RandomState(seed)
    w, h = int(size[0] / res), int(size[1] / res)
    base = np.array([168, 178, 188], np.float32)  # 베이지 (BGR)
    img = np.empty((h, w, 3), np.float32)
    tile = int(0.30 / res)
    for ty in range(0, h, tile):
        for tx in range(0, w, tile):
            img[ty:ty + tile, tx:tx + tile] = base + rng.uniform(-5, 5, 3)
    img += rng.uniform(-3, 3, (h, w, 1))                     # 미세 무늬
    img[::tile, :] = (120, 125, 130)                          # 이음새
    img[:, ::tile] = (120, 125, 130)
    for _ in range(6):                                        # 오래된 얼룩 (기준에도 있음 → 무시해야 함)
        _paint(img, res, Item("old", rng.uniform(0.5, 2.5), rng.uniform(1.1, 1.9), 0.03, 0.03, (130, 145, 160)), origin)
    clean = np.clip(img, 0, 255).astype(np.uint8)

    dirt = [Item("red_paper", 1.13, 1.42, 0.06, 0.06, (50, 50, 215)),
            Item("gray_dust", 1.86, 1.63, 0.05, 0.04, (95, 100, 105)),
            Item("green_paper", 2.37, 1.25, 0.05, 0.05, (70, 170, 70))]
    bag = Item("bag", 1.55, 1.15, 0.12, 0.20, (45, 50, 60))
    shoes = Item("shoes", 2.05, 1.86, 0.10, 0.08, SHOE_COLOR)
    d = img.copy()
    for it in dirt + [bag, shoes]:
        _paint(d, res, it, origin)
    dirty = np.clip(d * light_gain, 0, 255).astype(np.uint8)

    poses = []
    for lane, y in enumerate((1.32, 1.68)):                   # 2줄 왕복, 간격 36 cm < ROI 폭 50 cm
        xs = np.arange(0.35, 2.45, 0.04)
        if lane % 2:
            xs = xs[::-1]
        yaw0 = 0.0 if lane % 2 == 0 else math.pi
        for i, x in enumerate(xs):
            poses.append((float(x), y + 0.02 * math.sin(i / 4.0), yaw0 + 0.03 * math.sin(i / 6.0)))
    return Scene(FloorTexture(clean, origin[0], origin[1], res), FloorTexture(dirty, origin[0], origin[1], res),
                 dirt, bag, shoes, poses)


def item_outline(item, n=24):
    """물체 둘레 점 (지도 좌표) — LiDAR가 보는 점 흉내."""
    pts = []
    for t in np.linspace(0, 1, n, endpoint=False):
        p = t * 4
        k, f = int(p), p - int(p)
        hx, hy = item.w / 2, item.h / 2
        corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy), (-hx, -hy)]
        (ax, ay), (bx, by) = corners[k], corners[k + 1]
        pts.append((item.x + ax + (bx - ax) * f, item.y + ay + (by - ay) * f))
    return np.array(pts)


def noisy(pose, rng, xy_sigma, yaw_sigma_deg):
    return (pose[0] + rng.normal(0, xy_sigma), pose[1] + rng.normal(0, xy_sigma),
            pose[2] + math.radians(rng.normal(0, yaw_sigma_deg)))


def add_sensor_noise(img, rng, amp=3):
    return np.clip(img.astype(np.int16) + rng.randint(-amp, amp + 1, img.shape), 0, 255).astype(np.uint8)


class ColorBlobDetector:
    """detectNet 대역: 보라색 덩어리를 'person' 박스로 보고한다 (ImageExcluder의 detectnet 자리에 끼움)."""

    def __init__(self, classes=("person",), margin_px=10, min_px=30):
        self.classes = list(classes)
        self.margin_px = margin_px
        self.min_px = min_px

    def exclude_mask(self, bgr):
        from floor_dirt.masks import Detection, boxes_to_mask
        m = ((bgr[..., 2].astype(int) > 170) & (bgr[..., 0].astype(int) > 170) & (bgr[..., 1].astype(int) < 90))
        n, _, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
        dets = [Detection("person", 0.9, float(s[0]), float(s[1]), float(s[0] + s[2]), float(s[1] + s[3]))
                for s in stats[1:] if s[4] >= self.min_px]
        return boxes_to_mask(bgr.shape, dets, self.classes, self.margin_px), dets
