"""[0] 바닥 투영: 이미지 픽셀 → 로봇(base_footprint) → 지도 → 바닥 격자.

좌표 규칙 (CLAUDE.md §4)
- 로봇 좌표: base_footprint 기준 (x 전방, y 왼쪽) [m]
- H: 왜곡 보정된 이미지 픽셀 (u, v) → 로봇 바닥 좌표 (x, y)
- pose: map → base_footprint 자세 (x, y, yaw), 이미지 촬영 시점 값
- 바닥 격자: LiDAR 지도와 원점·방향 공유, 해상도만 다름.
  배열 인덱스 [row, col]에서 col은 +x, row는 +y 방향으로 증가한다
  (ROS 지도 PNG처럼 위아래를 뒤집지 않는다. 뒤집기는 시각화에서만 한다).
  칸 (row, col)의 중심 = (origin_x + (col + 0.5) * res, origin_y + (row + 0.5) * res)

핵심: 프레임마다 점을 하나씩 옮기지 않고, 이미지 → 격자 창을 바로 잇는
합성 호모그래피 M = Wofs · S_grid · T_pose · H 하나로 cv2.warpPerspective를 한 번 호출한다.
"""

import math
import os
from typing import NamedTuple, Optional, Tuple

import cv2
import numpy as np
import yaml

Pose = Tuple[float, float, float]


class GridMeta(NamedTuple):
    """바닥 격자 정의. 원점은 격자 (row 0, col 0) 칸의 바깥 모서리(최소 x, 최소 y)."""

    resolution: float
    origin_x: float
    origin_y: float
    width: int   # col 개수 (x 방향)
    height: int  # row 개수 (y 방향)

    @classmethod
    def from_lidar_map(cls, map_yaml_path, resolution):
        # type: (str, float) -> GridMeta
        """LiDAR 지도(map_server 형식 yaml + pgm)와 같은 영역을 덮는 바닥 격자."""
        with open(map_yaml_path, "r", encoding="utf-8") as f:
            info = yaml.safe_load(f)
        origin = info["origin"]
        if abs(float(origin[2])) > 1e-9:
            raise ValueError("지도 origin yaw가 0이 아닙니다 ({}). 회전된 지도는 지원하지 않습니다".format(origin[2]))
        image_path = info["image"]
        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(os.path.abspath(map_yaml_path)), image_path)
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError("지도 이미지를 읽을 수 없습니다: {}".format(image_path))
        lidar_res = float(info["resolution"])
        h_px, w_px = img.shape[:2]
        return cls(
            resolution=float(resolution),
            origin_x=float(origin[0]),
            origin_y=float(origin[1]),
            width=int(round(w_px * lidar_res / resolution)),
            height=int(round(h_px * lidar_res / resolution)),
        )


class WarpResult(NamedTuple):
    """warp_to_grid 결과. 격자 전체에서 [r0:r0+h, c0:c0+w] 창에 해당한다."""

    patch: np.ndarray   # (h, w[, C]) 이미지를 격자 창으로 옮긴 것
    valid: np.ndarray   # (h, w) bool. ROI 안이면서 이미지 범위 안인 칸
    r0: int
    c0: int
    M: np.ndarray       # 3x3, 이미지 픽셀 → 창 인덱스 (다른 이미지 마스크를 같은 방식으로 옮길 때 사용)


# ---------------------------------------------------------------- 왜곡 보정

def undistort_maps(K, D, size):
    # type: (np.ndarray, np.ndarray, Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]
    """cv2.remap용 왜곡 보정 테이블을 미리 계산한다. size = (width, height).

    보정 후 이미지의 내부 행렬은 K 그대로다 (H는 이 보정된 이미지 기준으로 구한다).
    """
    K = np.asarray(K, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    return cv2.initUndistortRectifyMap(K, D, None, K, tuple(int(s) for s in size), cv2.CV_16SC2)


def undistort(img, maps):
    # type: (np.ndarray, Tuple[np.ndarray, np.ndarray]) -> np.ndarray
    return cv2.remap(img, maps[0], maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def undistort_points(K, D, uv):
    # type: (np.ndarray, np.ndarray, np.ndarray) -> np.ndarray
    """원본(왜곡된) 이미지의 픽셀 좌표 → 왜곡 보정된 이미지의 픽셀 좌표. (N, 2)."""
    pts = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
    K = np.asarray(K, dtype=np.float64)
    out = cv2.undistortPoints(pts, K, np.asarray(D, dtype=np.float64), P=K)
    return out.reshape(-1, 2)


# ---------------------------------------------------------------- 점 변환

def apply_homography(H, pts):
    # type: (np.ndarray, np.ndarray) -> np.ndarray
    """(N, 2) 점에 3x3 행렬을 적용한다 (동차 좌표 나눗셈 포함)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ np.asarray(H, dtype=np.float64).T
    return hom[:, :2] / hom[:, 2:3]


def pixel_to_robot(H, uv):
    # type: (np.ndarray, np.ndarray) -> np.ndarray
    """보정된 이미지 픽셀 → base_footprint 바닥 좌표 [m]."""
    return apply_homography(H, uv)


def robot_to_pixel(H, xy):
    # type: (np.ndarray, np.ndarray) -> np.ndarray
    """base_footprint 바닥 좌표 → 보정된 이미지 픽셀. 카메라 뒤쪽 점은 의미 없는 값이 나온다."""
    return apply_homography(np.linalg.inv(H), xy)


def pose_matrix(pose):
    # type: (Pose) -> np.ndarray
    """로봇 좌표 → 지도 좌표 3x3 (SE2)."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])


def robot_to_map(pose, xy):
    # type: (Pose, np.ndarray) -> np.ndarray
    return apply_homography(pose_matrix(pose), xy)


def map_to_robot(pose, xy):
    # type: (Pose, np.ndarray) -> np.ndarray
    return apply_homography(np.linalg.inv(pose_matrix(pose)), xy)


def grid_matrix(grid):
    # type: (GridMeta) -> np.ndarray
    """지도 좌표 [m] → 연속 격자 인덱스 (col, row). 정수 값이 칸 중심이다 (warpPerspective 규칙)."""
    r = grid.resolution
    return np.array([
        [1.0 / r, 0.0, -grid.origin_x / r - 0.5],
        [0.0, 1.0 / r, -grid.origin_y / r - 0.5],
        [0.0, 0.0, 1.0],
    ])


def map_to_cell(grid, xy):
    # type: (GridMeta, np.ndarray) -> np.ndarray
    """지도 좌표 → 정수 칸 인덱스 (N, 2) = (row, col). 격자 밖일 수 있다."""
    cr = np.floor(apply_homography(grid_matrix(grid), xy) + 0.5).astype(np.int64)
    return cr[:, ::-1]


def cell_to_map(grid, rc):
    # type: (GridMeta, np.ndarray) -> np.ndarray
    """칸 인덱스 (row, col) → 칸 중심의 지도 좌표 (N, 2)."""
    rc = np.asarray(rc, dtype=np.float64).reshape(-1, 2)
    x = grid.origin_x + (rc[:, 1] + 0.5) * grid.resolution
    y = grid.origin_y + (rc[:, 0] + 0.5) * grid.resolution
    return np.stack([x, y], axis=1)


# ---------------------------------------------------------------- 격자로 옮기기

def roi_polygon(roi):
    # type: (dict) -> np.ndarray
    """params의 roi(x_min, x_max, y_min, y_max) → 로봇 좌표 꼭짓점 4개 (4, 2)."""
    return np.array([
        [roi["x_min"], roi["y_min"]],
        [roi["x_max"], roi["y_min"]],
        [roi["x_max"], roi["y_max"]],
        [roi["x_min"], roi["y_max"]],
    ], dtype=np.float64)


def _robot_to_pixel_checked(H, xy, img_w, img_h):
    # type: (np.ndarray, np.ndarray, int, int) -> Tuple[np.ndarray, np.ndarray]
    """robot_to_pixel + 카메라 앞쪽 점인지 여부.

    지평선 너머(카메라 뒤쪽) 바닥점도 호모그래피로는 그럴듯한 픽셀이 나오므로
    동차 좌표 w의 부호를 이미지 맨 아래 가운데 픽셀(항상 바닥을 본다고 가정)과 비교한다.
    """
    H = np.asarray(H, dtype=np.float64)
    ref_w = (H @ np.array([img_w / 2.0, img_h - 1.0, 1.0]))[2]
    hom = np.hstack([xy, np.ones((len(xy), 1))]) @ np.linalg.inv(H).T
    in_front = np.sign(hom[:, 2]) == np.sign(ref_w)
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = hom[:, :2] / hom[:, 2:3]
    return uv, in_front


def warp_to_grid(img, H, pose, grid, roi, interpolation=cv2.INTER_LINEAR):
    # type: (np.ndarray, np.ndarray, Pose, GridMeta, dict, int) -> Optional[WarpResult]
    """보정된 이미지를 근거리 ROI가 덮는 격자 창으로 옮긴다.

    반환값이 None이면 ROI가 격자와 전혀 겹치지 않는 경우다.
    valid는 칸 중심이 ROI 안에 있고, 이미지 안쪽(테두리 1픽셀 제외)으로 투영되는 칸이다.
    """
    S_T = grid_matrix(grid) @ pose_matrix(pose)
    corners = apply_homography(S_T, roi_polygon(roi))  # (col, row) 연속 인덱스
    c0 = max(int(math.floor(corners[:, 0].min())), 0)
    r0 = max(int(math.floor(corners[:, 1].min())), 0)
    c1 = min(int(math.ceil(corners[:, 0].max())) + 1, grid.width)
    r1 = min(int(math.ceil(corners[:, 1].max())) + 1, grid.height)
    if c1 <= c0 or r1 <= r0:
        return None
    w, h = c1 - c0, r1 - r0

    offset = np.array([[1.0, 0.0, -c0], [0.0, 1.0, -r0], [0.0, 0.0, 1.0]])
    M = offset @ S_T @ np.asarray(H, dtype=np.float64)
    patch = cv2.warpPerspective(img, M, (w, h), flags=interpolation, borderMode=cv2.BORDER_CONSTANT)

    # 칸 중심을 로봇 좌표와 이미지 픽셀로 되돌려 유효 여부를 판정한다
    cols, rows = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    centers = np.stack([cols.ravel(), rows.ravel()], axis=1)
    robot_xy = apply_homography(np.linalg.inv(offset @ S_T), centers)
    in_roi = ((robot_xy[:, 0] >= roi["x_min"]) & (robot_xy[:, 0] <= roi["x_max"])
              & (robot_xy[:, 1] >= roi["y_min"]) & (robot_xy[:, 1] <= roi["y_max"]))
    img_h, img_w = img.shape[:2]
    uv, in_front = _robot_to_pixel_checked(H, robot_xy, img_w, img_h)
    in_img = (in_front
              & (uv[:, 0] >= 1) & (uv[:, 0] <= img_w - 2)
              & (uv[:, 1] >= 1) & (uv[:, 1] <= img_h - 2))
    valid = (in_roi & in_img).reshape(h, w)
    return WarpResult(patch=patch, valid=valid, r0=r0, c0=c0, M=M)


def warp_mask(mask, result):
    # type: (np.ndarray, WarpResult) -> np.ndarray
    """이미지 크기의 bool 마스크(바닥, 제외 영역 등)를 같은 격자 창으로 옮긴다."""
    h, w = result.valid.shape
    out = cv2.warpPerspective(mask.astype(np.uint8), result.M, (w, h),
                              flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    return out.astype(bool)


def window_slices(result):
    # type: (WarpResult) -> Tuple[slice, slice]
    """격자 전체 배열에서 이 창에 해당하는 (row, col) 슬라이스."""
    h, w = result.valid.shape
    return slice(result.r0, result.r0 + h), slice(result.c0, result.c0 + w)


def roi_corners_pixel(H, roi):
    # type: (np.ndarray, dict) -> np.ndarray
    """ROI 꼭짓점의 보정된 이미지 픽셀 좌표 (디버그 오버레이용)."""
    return robot_to_pixel(H, roi_polygon(roi))


__all__ = [
    "GridMeta", "WarpResult", "Pose",
    "undistort_maps", "undistort", "undistort_points",
    "apply_homography", "pixel_to_robot", "robot_to_pixel",
    "pose_matrix", "robot_to_map", "map_to_robot",
    "grid_matrix", "map_to_cell", "cell_to_map",
    "roi_polygon", "warp_to_grid", "warp_mask", "window_slices", "roi_corners_pixel",
]
