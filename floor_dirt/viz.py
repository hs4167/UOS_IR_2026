"""시각화: 웹에서 볼 결과 지도(dirt_map.png)와 디버그용 카메라 오버레이.

결과 지도 (지도처럼 위쪽 = +y. 격자 배열은 row = +y라 위아래를 뒤집어 그린다)
- 배경: LiDAR 지도 (빈 곳 밝은 회색, 벽·장애물 검정, 모름 중간 회색). 없으면 흰색
- 깨끗함: 연두색, 오염 의심: 확률에 따라 노랑(0.7) → 빨강(1.0)
- 봤지만 판정 못 한 칸(기준 없음·관측 부족): 회색 빗금. 안 간 곳은 배경 그대로
- 오염 구역: 자홍색 사각형 + 번호, 로봇: 파란 원 + 방향, 경로: 하늘색 선
cv2.putText는 한글을 못 그리므로 그림 안 글자는 영어로 쓴다.
"""

import os
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from floor_dirt.dirt_map import CLEAN, SUSPECT, DirtMap, Region
from floor_dirt.masks import Detection
from floor_dirt.projection import GridMeta, Pose, WarpResult, apply_homography, grid_matrix

COL_FREE = 235
COL_OCC = 40
COL_UNK = 190
COL_CLEAN = (140, 215, 140)
COL_SUSPECT_LO = (0, 220, 255)   # 노랑 (BGR)
COL_SUSPECT_HI = (0, 0, 220)     # 빨강
COL_HATCH = (120, 120, 120)
COL_REGION = (200, 0, 200)
COL_ROBOT = (200, 90, 0)
COL_PATH = (230, 180, 60)
ROBOT_RADIUS = 0.15              # Waffle Pi 대략 반경 [m]


# ---------------------------------------------------------------- 결과 지도

def lidar_background(map_yaml_path, grid):
    # type: (str, GridMeta) -> np.ndarray
    """LiDAR 지도를 바닥 격자 크기의 회색 배경으로 (격자 규칙: row = +y)."""
    with open(map_yaml_path, "r", encoding="utf-8") as f:
        info = yaml.safe_load(f)
    path = info["image"]
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(map_yaml_path)), path)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("지도 이미지를 읽을 수 없습니다: {}".format(path))
    if int(info.get("negate", 0)):
        img = 255 - img
    occ_t = float(info.get("occupied_thresh", 0.65))
    free_t = float(info.get("free_thresh", 0.196))
    p_occ = (255.0 - img.astype(np.float32)) / 255.0
    bg = np.full(img.shape, COL_UNK, np.uint8)
    bg[p_occ > occ_t] = COL_OCC
    bg[p_occ < free_t] = COL_FREE
    bg = bg[::-1]
    return cv2.resize(bg, (grid.width, grid.height), interpolation=cv2.INTER_NEAREST)


def _suspect_color(p, suspect_p):
    t = np.clip((p - suspect_p) / max(1.0 - suspect_p, 1e-6), 0.0, 1.0)[..., None]
    lo = np.array(COL_SUSPECT_LO, np.float32)
    hi = np.array(COL_SUSPECT_HI, np.float32)
    return (lo * (1 - t) + hi * t).astype(np.uint8)


def _to_px(grid, xy, scale, height_px):
    """지도 좌표 → 그림 픽셀 (위아래 뒤집힘 반영). (N, 2) float."""
    cr = apply_homography(grid_matrix(grid), xy)  # 연속 (col, row), 정수 = 칸 중심
    u = (cr[:, 0] + 0.5) * scale
    v = height_px - (cr[:, 1] + 0.5) * scale
    return np.stack([u, v], axis=1)


def render_dirt_map(dm, regions=(), background=None, robot_pose=None, path_xy=None, scale=1, crop=None,
                    legend=True):
    # type: (DirtMap, Sequence[Region], Optional[np.ndarray], Optional[Pose], Optional[np.ndarray], int, Optional[Tuple[int, int, int, int]], bool) -> np.ndarray
    """결과 지도 BGR 이미지.

    background: lidar_background 결과 (격자 크기 회색). scale: 칸 1개를 몇 픽셀로 그릴지 (정수).
    crop: (r0, c0, r1, c1) 격자 일부만 그릴 때.
    """
    g = dm.grid
    if background is None:
        img = np.full((g.height, g.width, 3), 255, np.uint8)
    else:
        if background.shape[:2] != (g.height, g.width):
            raise ValueError("배경 크기 {}가 격자 {}와 다릅니다".format(background.shape[:2], (g.height, g.width)))
        img = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR) if background.ndim == 2 else background.copy()

    st = dm.states()
    clean = st == CLEAN
    img[clean] = (0.5 * img[clean] + 0.5 * np.array(COL_CLEAN)).astype(np.uint8)
    sus = st == SUSPECT
    if sus.any():
        img[sus] = _suspect_color(dm.probability()[sus], dm.suspect_p)
    pending = (dm.n_seen > 0) & ~clean & ~sus   # 봤지만 판정 못 함

    r0, c0, r1, c1 = crop if crop is not None else (0, 0, g.height, g.width)
    img = img[r0:r1, c0:c1]
    pending = pending[r0:r1, c0:c1]
    sub = g._replace(origin_x=g.origin_x + c0 * g.resolution, origin_y=g.origin_y + r0 * g.resolution,
                     width=c1 - c0, height=r1 - r0)

    img = np.ascontiguousarray(img[::-1])  # cv2 그리기 함수는 뒤집힌(음수 stride) 배열을 못 받는다
    pending = pending[::-1]
    s = max(int(scale), 1)
    if s > 1:
        img = cv2.resize(img, (img.shape[1] * s, img.shape[0] * s), interpolation=cv2.INTER_NEAREST)
        pending = cv2.resize(pending.astype(np.uint8), (img.shape[1], img.shape[0]),
                             interpolation=cv2.INTER_NEAREST).astype(bool)
    if pending.any():
        yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        img[pending & ((xx + yy) % 6 == 0)] = COL_HATCH

    H_px = img.shape[0]
    if path_xy is not None and len(path_xy) >= 2:
        pts = np.round(_to_px(sub, np.asarray(path_xy, float), s, H_px)).astype(np.int32)
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, COL_PATH, max(1, s // 2), cv2.LINE_AA)

    thick = max(1, s // 2)
    for reg in regions:
        rr0, cc0, rr1, cc1 = reg.bbox_cells
        x0 = (cc0 - c0) * s
        x1 = (cc1 - c0) * s - 1
        y0 = H_px - (rr1 - r0) * s
        y1 = H_px - (rr0 - r0) * s - 1
        pad = 2 * s
        cv2.rectangle(img, (x0 - pad, y0 - pad), (x1 + pad, y1 + pad), COL_REGION, thick)
        cv2.putText(img, str(reg.id), (x1 + pad + 2, max(y0 - pad, 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4 + 0.1 * s, COL_REGION, thick, cv2.LINE_AA)

    if robot_pose is not None:
        x, y, yaw = robot_pose
        ctr = _to_px(sub, [[x, y]], s, H_px)[0]
        tip = _to_px(sub, [[x + ROBOT_RADIUS * np.cos(yaw), y + ROBOT_RADIUS * np.sin(yaw)]], s, H_px)[0]
        rad = max(int(round(ROBOT_RADIUS / g.resolution * s)), 3)
        cv2.circle(img, tuple(np.round(ctr).astype(int)), rad, COL_ROBOT, thick, cv2.LINE_AA)
        cv2.line(img, tuple(np.round(ctr).astype(int)), tuple(np.round(tip).astype(int)), COL_ROBOT, thick, cv2.LINE_AA)

    if legend:
        _draw_legend(img, dm)
    return img


def _draw_legend(img, dm):
    sm = dm.summary()
    items = [(COL_CLEAN, "clean {:.2f} m2".format(sm["clean_m2"])),
             (COL_SUSPECT_HI, "suspect {:.3f} m2".format(sm["suspect_m2"])),
             (COL_HATCH, "seen, no decision"),
             (COL_REGION, "region")]
    x, y = 6, 6
    w = 150
    if img.shape[0] < 70 or img.shape[1] < w + 10:
        return
    over = img[y:y + 16 * len(items) + 6, x:x + w]
    over[:] = (0.3 * over + 0.7 * 255).astype(np.uint8)
    for i, (col, text) in enumerate(items):
        yy = y + 6 + 16 * i
        cv2.rectangle(img, (x + 4, yy), (x + 14, yy + 10), col, -1)
        cv2.putText(img, text, (x + 20, yy + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 30, 30), 1, cv2.LINE_AA)


def save_png_atomic(path, img):
    # type: (str, np.ndarray) -> None
    """임시 파일에 쓴 뒤 교체한다 (웹에서 읽는 중에 반쯤 쓰인 파일을 받지 않게)."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, ".tmp_" + os.path.basename(path))
    if not cv2.imwrite(tmp, img):
        raise IOError("이미지 저장 실패: {}".format(tmp))
    os.replace(tmp, path)


def region_crop(frame_bgr, result, region, margin_cells=5):
    # type: (np.ndarray, WarpResult, Region, int) -> Optional[np.ndarray]
    """대표 사진: 구역 바운딩 박스(+여백)를 카메라 원본 위치로 되돌려 잘라낸다. 화면 밖이면 None."""
    r0, c0, r1, c1 = region.bbox_cells
    cells = np.array([[c0 - margin_cells, r0 - margin_cells], [c1 + margin_cells, r0 - margin_cells],
                      [c1 + margin_cells, r1 + margin_cells], [c0 - margin_cells, r1 + margin_cells]], float)
    cells -= [[result.c0 + 0.5, result.r0 + 0.5]]
    uv = apply_homography(np.linalg.inv(result.M), cells)
    h, w = frame_bgr.shape[:2]
    u0, v0 = np.clip(np.floor(uv.min(0)), 0, [w, h]).astype(int)
    u1, v1 = np.clip(np.ceil(uv.max(0)), 0, [w, h]).astype(int)
    if u1 - u0 < 4 or v1 - v0 < 4:
        return None
    crop = frame_bgr[v0:v1, u0:u1].copy()
    poly = np.round(uv - [u0, v0]).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(crop, [poly], True, COL_REGION, 2, cv2.LINE_AA)
    return crop


# ---------------------------------------------------------------- 디버그 오버레이 (카메라 화면)

def window_to_image(mask, result, image_shape):
    # type: (np.ndarray, WarpResult, Tuple[int, int]) -> np.ndarray
    """창 칸 마스크(hit 등)를 카메라 이미지 픽셀로 되돌린다."""
    h, w = image_shape[:2]
    out = cv2.warpPerspective(mask.astype(np.uint8), result.M, (w, h),
                              flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
    return out.astype(bool)


def _tint(img, mask, color, alpha):
    if mask is not None and mask.any():
        img[mask] = ((1 - alpha) * img[mask] + alpha * np.array(color, np.float32)).astype(np.uint8)


def debug_overlay(frame_bgr, floor_mask=None, exclude_mask=None, roi_poly_px=None, detections=(), hit_mask=None,
                  text_lines=()):
    # type: (np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Iterable[Detection], Optional[np.ndarray], Iterable[str]) -> np.ndarray
    """카메라 화면 위에 처리 과정을 겹친다 (RTSP·파일 디버그용). 모든 입력은 왜곡 보정된 이미지 기준.

    바닥 = 초록, 제외 = 빨강, hit(기준과 달라진 곳) = 노랑, ROI = 주황 테두리, detectNet 박스 = 하늘색.
    """
    img = frame_bgr.copy() if frame_bgr.ndim == 3 else cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
    _tint(img, floor_mask, (0, 200, 0), 0.25)
    _tint(img, exclude_mask, (0, 0, 230), 0.4)
    _tint(img, hit_mask, (0, 230, 255), 0.6)
    if roi_poly_px is not None:
        poly = np.round(np.asarray(roi_poly_px, float)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [poly], True, (0, 140, 255), 2, cv2.LINE_AA)
    for d in detections:
        p0 = (int(round(d.left)), int(round(d.top)))
        p1 = (int(round(d.right)), int(round(d.bottom)))
        cv2.rectangle(img, p0, p1, (255, 200, 0), 2)
        cv2.putText(img, "{} {:.2f}".format(d.name, d.confidence), (p0[0], max(p0[1] - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA)
    for i, line in enumerate(text_lines):
        y = 18 + 18 * i
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img

