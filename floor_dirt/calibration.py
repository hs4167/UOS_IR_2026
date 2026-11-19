"""[0] 카메라 보정: 체커보드 → 내부 보정(K, D) → 바닥 호모그래피(H) → 투영 오차 검사.

바닥 보정 시 체커보드 놓는 규칙 (CLAUDE.md §5 [0])
- 로봇 앞 바닥에 평평하게 놓는다. 가로 코너(pattern[0]개) 방향이 로봇 좌우(y)가 되게 한다.
- "기준 코너" = 로봇에서 가장 가깝고 오른쪽에 있는 내부 코너 (이미지에서 오른쪽 아래).
  이 코너의 base_footprint 좌표 (x 전방, y 왼쪽)를 줄자로 잰다.
- 기준 코너에서 로봇 왼쪽(+y)으로 가로 코너가, 앞쪽(+x)으로 세로 코너가 이어진다.
  보드가 로봇 축과 비뚤어져 있으면 그 각도를 yaw로 준다 (반시계 +).
"""

import datetime
import math
import os
from typing import List, NamedTuple, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from floor_dirt.config import REPO_ROOT
from floor_dirt.projection import apply_homography, undistort_points

DEFAULT_INTRINSICS_PATH = os.path.join(REPO_ROOT, "config", "camera_intrinsics.yaml")
DEFAULT_FLOOR_PATH = os.path.join(REPO_ROOT, "config", "floor_homography.yaml")

Pattern = Tuple[int, int]  # (가로 코너 수, 세로 코너 수)

_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)


class Intrinsics(NamedTuple):
    K: np.ndarray       # 3x3
    D: np.ndarray       # (5,) k1 k2 p1 p2 k3
    width: int
    height: int
    rms: float          # 재투영 오차 RMS [px]


class FloorCalibration(NamedTuple):
    H: np.ndarray       # 3x3, 보정된 이미지 픽셀 → base_footprint 바닥 좌표 [m]
    width: int
    height: int
    fit_mean: float     # 보정에 쓴 코너들의 평균 오차 [m]
    fit_max: float


class ErrorStats(NamedTuple):
    errors: np.ndarray  # 점별 오차 [m]
    mean: float
    max: float
    n: int
    passed: bool


# ---------------------------------------------------------------- 체커보드

def to_gray(img):
    # type: (np.ndarray) -> np.ndarray
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def find_chessboard(img, pattern):
    # type: (np.ndarray, Pattern) -> Optional[np.ndarray]
    """내부 코너를 찾아 서브픽셀로 다듬는다. (N, 2), N = cols * rows. 못 찾으면 None."""
    gray = to_gray(img)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, tuple(pattern), flags=flags)
    if not found:
        return None
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), _SUBPIX_CRITERIA)
    return corners.reshape(-1, 2).astype(np.float64)


def board_object_points(pattern, square):
    # type: (Pattern, float) -> np.ndarray
    """보드 자체 좌표의 코너 위치 (N, 3), z = 0. OpenCV 코너 순서(가로 우선)와 같다."""
    cols, rows = pattern
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    return np.column_stack([c.ravel() * square, r.ravel() * square, np.zeros(cols * rows)]).astype(np.float32)


# ---------------------------------------------------------------- 내부 보정

def calibrate_intrinsics(images, pattern, square):
    # type: (Sequence[np.ndarray], Pattern, float) -> Tuple[Intrinsics, List[int]]
    """여러 각도에서 찍은 체커보드 이미지로 K, D를 구한다. (결과, 코너를 찾은 이미지 번호)."""
    corners_list, used = [], []
    size = None
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        if size is None:
            size = (w, h)
        elif size != (w, h):
            raise ValueError("이미지 해상도가 섞여 있습니다: {} vs {}".format(size, (w, h)))
        corners = find_chessboard(img, pattern)
        if corners is not None:
            corners_list.append(corners)
            used.append(i)
    return calibrate_from_corners(corners_list, size, pattern, square), used


def calibrate_from_corners(corners_list, size, pattern, square):
    # type: (Sequence[np.ndarray], Tuple[int, int], Pattern, float) -> Intrinsics
    """이미지별 검출 코너 목록으로 K, D를 구한다. size = (width, height)."""
    if len(corners_list) < 3:
        raise ValueError("체커보드를 찾은 이미지가 {}장뿐입니다 (최소 3장, 권장 15~20장)".format(len(corners_list)))
    obj = board_object_points(pattern, square)
    obj_pts = [obj] * len(corners_list)
    img_pts = [np.asarray(c, dtype=np.float32).reshape(-1, 1, 2) for c in corners_list]
    rms, K, D, _, _ = cv2.calibrateCamera(obj_pts, img_pts, tuple(size), None, None)
    return Intrinsics(K=K, D=D.ravel()[:5], width=int(size[0]), height=int(size[1]), rms=float(rms))


def corner_coverage(corners_list, size, cells=(3, 3)):
    # type: (Sequence[np.ndarray], Tuple[int, int], Tuple[int, int]) -> np.ndarray
    """화면을 cells(가로, 세로)로 나눴을 때 칸별 검출 코너 개수 (세로, 가로).

    0인 칸이 있으면 그 구석의 렌즈 왜곡이 추정되지 않는다 → 보드를 그쪽에 두고 더 찍어야 한다.
    """
    w, h = size
    nx, ny = cells
    counts = np.zeros((ny, nx), np.int64)
    for c in corners_list:
        c = np.asarray(c).reshape(-1, 2)
        ix = np.clip((c[:, 0] / w * nx).astype(int), 0, nx - 1)
        iy = np.clip((c[:, 1] / h * ny).astype(int), 0, ny - 1)
        np.add.at(counts, (iy, ix), 1)
    return counts


def save_intrinsics(path, intr, note=""):
    # type: (str, Intrinsics, str) -> None
    data = {
        "width": intr.width, "height": intr.height,
        "K": np.asarray(intr.K).tolist(), "D": np.asarray(intr.D).ravel().tolist(),
        "rms_px": float(intr.rms),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "note": note,
    }
    _write_yaml(path, data)


def load_intrinsics(path=DEFAULT_INTRINSICS_PATH):
    # type: (str) -> Intrinsics
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return Intrinsics(K=np.array(d["K"], dtype=np.float64), D=np.array(d["D"], dtype=np.float64),
                      width=int(d["width"]), height=int(d["height"]), rms=float(d.get("rms_px", float("nan"))))


# ---------------------------------------------------------------- 바닥 호모그래피

def order_floor_corners(corners, pattern):
    # type: (np.ndarray, Pattern) -> np.ndarray
    """검출 순서와 관계없이 [세로 r, 가로 c] 격자로 정렬한다.

    [0, 0] = 이미지 오른쪽 아래(로봇에 가깝고 오른쪽), c 증가 = 이미지 왼쪽(로봇 +y),
    r 증가 = 이미지 위쪽(로봇 +x). 반환 (rows, cols, 2).
    OpenCV는 같은 보드를 180도 뒤집힌 순서로 돌려줄 수 있어서 직접 맞춘다.
    """
    cols, rows = pattern
    g = np.asarray(corners, dtype=np.float64).reshape(rows, cols, 2)
    candidates = [g, g[::-1], g[:, ::-1], g[::-1, ::-1]]
    if rows == cols:
        t = g.transpose(1, 0, 2)
        candidates += [t, t[::-1], t[:, ::-1], t[::-1, ::-1]]
    for c in candidates:
        dc = c[0, -1] - c[0, 0]   # 가로 방향: 왼쪽(u 감소)이 주성분
        dr = c[-1, 0] - c[0, 0]   # 세로 방향: 위쪽(v 감소)이 주성분
        if dc[0] < 0 and abs(dc[0]) > abs(dc[1]) and dr[1] < 0 and abs(dr[1]) > abs(dr[0]):
            return c
    raise ValueError("체커보드 방향을 알 수 없습니다. 가로 코너 {}개 방향이 로봇 좌우가 되도록 놓으세요"
                     " (90도 돌아가 있으면 pattern 가로·세로를 바꾸거나 보드를 돌리세요)".format(cols))


def floor_board_world(pattern, square, origin_xy, yaw=0.0):
    # type: (Pattern, float, Tuple[float, float], float) -> np.ndarray
    """order_floor_corners 순서에 맞는 코너의 로봇 좌표 (rows, cols, 2)."""
    cols, rows = pattern
    r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    local = np.stack([r * square, c * square], axis=-1)  # (x 전방, y 왼쪽)
    cy, sy = math.cos(yaw), math.sin(yaw)
    R = np.array([[cy, -sy], [sy, cy]])
    return local @ R.T + np.asarray(origin_xy, dtype=np.float64)


def detect_floor_board(img, intr, pattern, square, origin_xy, yaw=0.0):
    # type: (np.ndarray, Intrinsics, Pattern, float, Tuple[float, float], float) -> Tuple[np.ndarray, np.ndarray]
    """원본(왜곡된) 바닥 보드 이미지 → (보정된 픽셀 (N, 2), 로봇 좌표 (N, 2)).

    코너는 원본 이미지에서 찾고 좌표만 왜곡 보정한다 (remap한 이미지에서 찾는 것보다 정확).
    """
    corners = find_chessboard(img, pattern)
    if corners is None:
        raise ValueError("바닥 체커보드를 찾지 못했습니다 (pattern={}). 보드 전체가 화면에 들어오는지 확인".format(tuple(pattern)))
    uv = undistort_points(intr.K, intr.D, corners)
    uv = order_floor_corners(uv, pattern).reshape(-1, 2)
    xy = floor_board_world(pattern, square, origin_xy, yaw).reshape(-1, 2)
    return uv, xy


def fit_floor_homography(uv, xy, width, height):
    # type: (np.ndarray, np.ndarray, int, int) -> FloorCalibration
    """대응점 (보정된 픽셀 ↔ 로봇 바닥 좌표)으로 H를 최소제곱 추정한다."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if len(uv) < 4:
        raise ValueError("대응점이 4개 이상 필요합니다")
    H, _ = cv2.findHomography(uv, xy, 0)
    if H is None:
        raise ValueError("호모그래피 추정 실패 (점들이 한 직선 위에 있는지 확인)")
    H = H / H[2, 2]
    err = np.linalg.norm(apply_homography(H, uv) - xy, axis=1)
    return FloorCalibration(H=H, width=int(width), height=int(height),
                            fit_mean=float(err.mean()), fit_max=float(err.max()))


def save_floor_calibration(path, calib, meta=None):
    # type: (str, FloorCalibration, Optional[dict]) -> None
    data = {
        "width": calib.width, "height": calib.height,
        "H": np.asarray(calib.H).tolist(),
        "fit_mean_m": float(calib.fit_mean), "fit_max_m": float(calib.fit_max),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if meta:
        data["board"] = meta
    _write_yaml(path, data)


def load_floor_calibration(path=DEFAULT_FLOOR_PATH):
    # type: (str) -> FloorCalibration
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return FloorCalibration(H=np.array(d["H"], dtype=np.float64), width=int(d["width"]), height=int(d["height"]),
                            fit_mean=float(d.get("fit_mean_m", float("nan"))),
                            fit_max=float(d.get("fit_max_m", float("nan"))))


# ---------------------------------------------------------------- 오차 검사

def in_roi(xy, roi):
    # type: (np.ndarray, dict) -> np.ndarray
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    return ((xy[:, 0] >= roi["x_min"]) & (xy[:, 0] <= roi["x_max"])
            & (xy[:, 1] >= roi["y_min"]) & (xy[:, 1] <= roi["y_max"]))


def projection_error(H, uv, xy, pass_mean=0.01, min_points=5):
    # type: (np.ndarray, np.ndarray, np.ndarray, float, int) -> ErrorStats
    """보정된 픽셀 uv를 H로 투영한 위치와 실측 xy의 거리. 평균 ≤ pass_mean, 점 ≥ min_points면 통과."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if len(uv) == 0:
        return ErrorStats(np.zeros(0), float("nan"), float("nan"), 0, False)
    err = np.linalg.norm(apply_homography(H, uv) - xy, axis=1)
    mean = float(err.mean())
    return ErrorStats(errors=err, mean=mean, max=float(err.max()), n=len(err),
                      passed=bool(len(err) >= min_points and mean <= pass_mean))


def _write_yaml(path, data):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=None)
