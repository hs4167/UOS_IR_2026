"""[2] 기준 바닥 격자: 칸별 바닥 색(Lab) 평균·분산·관측 횟수를 누적·저장한다.

색 공간: OpenCV 8비트 Lab 값을 float로 쓴다 (L 0~255, a·b는 128이 무채색).
compare.min_thr 등 임계값도 이 단위를 따른다.

누적 방식: 칸마다 Welford 방식(평균과 편차 제곱합을 바로 갱신)으로 float32에 누적한다.
합·제곱합 방식은 float32에서 관측이 많아지면 분산 계산 시 자릿수 손실이 생겨서 쓰지 않는다.
격자 1000 x 800칸이면 층 7개 x 4 B = 약 22 MB.
"""

import datetime
import os
from typing import List, NamedTuple, Optional

import cv2
import numpy as np
import yaml

from floor_dirt.projection import GridMeta, WarpResult

NPZ_NAME = "reference.npz"
META_NAME = "reference.yaml"
COLOR_SPACE = "opencv_lab_8bit"


def to_lab(bgr):
    # type: (np.ndarray) -> np.ndarray
    """uint8 BGR (h, w, 3) → float32 Lab (h, w, 3)."""
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("uint8 BGR 3채널 이미지가 필요합니다 (받은 값: {} {})".format(bgr.dtype, bgr.shape))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


class Reference(NamedTuple):
    """비교 단계에서 쓰는 기준 값 (창 단위로 잘라 쓸 수 있다)."""

    mean: np.ndarray    # (h, w, 3) Lab 평균
    var: np.ndarray     # (h, w, 3) 분산 (모분산)
    count: np.ndarray   # (h, w) 관측 횟수
    valid: np.ndarray   # (h, w) bool, count >= min_count


class FloorGrid:
    """칸별 Lab 통계. 배열 인덱스 규칙은 projection.GridMeta와 같다 ([row, col], row = +y)."""

    def __init__(self, grid, min_count=1):
        # type: (GridMeta, int) -> None
        self.grid = grid
        self.min_count = int(min_count)
        shape = (grid.height, grid.width)
        self.mean = np.zeros(shape + (3,), np.float32)
        self.m2 = np.zeros(shape + (3,), np.float32)
        self.count = np.zeros(shape, np.float32)
        self.runs = []  # type: List[str]

    # ------------------------------------------------------------ 누적

    def add_patch(self, lab, mask, r0, c0):
        # type: (np.ndarray, np.ndarray, int, int) -> int
        """Lab 패치를 격자 [r0:, c0:] 위치에 누적한다. mask가 True인 칸만. 누적한 칸 수를 돌려준다."""
        h, w = mask.shape
        if lab.shape[:2] != (h, w):
            raise ValueError("패치 {}와 마스크 {} 크기가 다릅니다".format(lab.shape[:2], (h, w)))
        if r0 < 0 or c0 < 0 or r0 + h > self.grid.height or c0 + w > self.grid.width:
            raise ValueError("패치가 격자 밖으로 나갑니다: r0={} c0={} size={}".format(r0, c0, (h, w)))
        rs, cs = slice(r0, r0 + h), slice(c0, c0 + w)
        m = mask.astype(bool)
        n_new = int(m.sum())
        if n_new == 0:
            return 0
        cnt = self.count[rs, cs]
        mean = self.mean[rs, cs]
        m2 = self.m2[rs, cs]
        x = lab[m].astype(np.float32)
        n = cnt[m] + 1.0
        delta = x - mean[m]
        new_mean = mean[m] + delta / n[:, None]
        m2[m] = m2[m] + delta * (x - new_mean)
        mean[m] = new_mean
        cnt[m] = n
        return n_new

    def add_warp(self, result, mask=None):
        # type: (WarpResult, Optional[np.ndarray]) -> int
        """warp_to_grid 결과(BGR 패치)를 누적한다. mask: 추가 제외 후 남길 칸 (창 크기 bool)."""
        keep = result.valid if mask is None else (result.valid & mask)
        return self.add_patch(to_lab(result.patch), keep, result.r0, result.c0)

    def merge(self, other):
        # type: (FloorGrid) -> None
        """다른 주행에서 만든 격자를 합친다 (두 집단의 평균·분산 병합 공식)."""
        if other.grid != self.grid:
            raise ValueError("격자 정의가 다릅니다: {} vs {}".format(self.grid, other.grid))
        na = self.count[..., None]
        nb = other.count[..., None]
        n = na + nb
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = other.mean - self.mean
            frac = np.where(n > 0, nb / n, 0.0)
            self.mean = (self.mean + delta * frac).astype(np.float32)
            self.m2 = (self.m2 + other.m2 + delta * delta * np.where(n > 0, na * nb / n, 0.0)).astype(np.float32)
        self.count = (self.count + other.count).astype(np.float32)
        self.runs = self.runs + [r for r in other.runs if r not in self.runs]

    # ------------------------------------------------------------ 조회

    def var(self):
        # type: () -> np.ndarray
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(self.count[..., None] > 0, self.m2 / self.count[..., None], 0.0)
        return np.maximum(v, 0.0).astype(np.float32)

    def reference(self, rows=slice(None), cols=slice(None)):
        # type: (slice, slice) -> Reference
        """기준 값. rows, cols로 창만 잘라 받을 수 있다 (비교 단계에서 window_slices 결과를 넘김)."""
        cnt = self.count[rows, cols]
        m2 = self.m2[rows, cols]
        with np.errstate(divide="ignore", invalid="ignore"):
            var = np.where(cnt[..., None] > 0, m2 / cnt[..., None], 0.0)
        return Reference(mean=self.mean[rows, cols], var=np.maximum(var, 0.0).astype(np.float32),
                         count=cnt, valid=cnt >= self.min_count)

    def coverage(self, floor_mask=None):
        # type: (Optional[np.ndarray]) -> dict
        """기준이 있는 칸 비율. floor_mask(바닥으로 갈 수 있는 칸)가 있으면 그 안에서만 센다."""
        covered = self.count >= self.min_count
        area = np.ones_like(covered) if floor_mask is None else floor_mask.astype(bool)
        n_floor = int(area.sum())
        n_cov = int((covered & area).sum())
        res2 = self.grid.resolution ** 2
        return {
            "floor_cells": n_floor,
            "covered_cells": n_cov,
            "covered_ratio": (n_cov / n_floor) if n_floor else 0.0,
            "covered_area_m2": n_cov * res2,
            "missing_area_m2": (n_floor - n_cov) * res2,
        }

    def coverage_image(self, floor_mask=None):
        # type: (Optional[np.ndarray]) -> np.ndarray
        """확인용 BGR 그림 (위쪽 = +y, 지도처럼 보이게 위아래 뒤집음).

        기준 있음 = 기준 색, 바닥인데 기준 없음 = 빨강(재주행 필요), 그 외 = 어두운 회색.
        """
        covered = self.count >= self.min_count
        lab = np.clip(self.mean, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        img[~covered] = (40, 40, 40)
        if floor_mask is not None:
            img[floor_mask.astype(bool) & ~covered] = (0, 0, 220)
        return img[::-1].copy()

    # ------------------------------------------------------------ 저장

    def save(self, directory, extra=None):
        # type: (str, Optional[dict]) -> None
        """directory/reference.npz (배열) + reference.yaml (격자 정의·메타)."""
        os.makedirs(directory, exist_ok=True)
        tmp = os.path.join(directory, NPZ_NAME + ".tmp.npz")
        np.savez_compressed(tmp, mean=self.mean, m2=self.m2, count=self.count)
        os.replace(tmp, os.path.join(directory, NPZ_NAME))
        meta = {
            "grid": dict(self.grid._asdict()),
            "color_space": COLOR_SPACE,
            "min_count": self.min_count,
            "runs": list(self.runs),
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "coverage": self.coverage(),
        }
        if extra:
            meta.update(extra)
        with open(os.path.join(directory, META_NAME), "w", encoding="utf-8") as f:
            yaml.safe_dump(meta, f, allow_unicode=True, sort_keys=False)

    @classmethod
    def load(cls, directory, min_count=None):
        # type: (str, Optional[int]) -> FloorGrid
        with open(os.path.join(directory, META_NAME), "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)
        if meta.get("color_space") != COLOR_SPACE:
            raise ValueError("색 공간이 다릅니다: {}".format(meta.get("color_space")))
        g = meta["grid"]
        grid = GridMeta(float(g["resolution"]), float(g["origin_x"]), float(g["origin_y"]),
                        int(g["width"]), int(g["height"]))
        fg = cls(grid, meta.get("min_count", 1) if min_count is None else min_count)
        with np.load(os.path.join(directory, NPZ_NAME)) as d:
            fg.mean = d["mean"].astype(np.float32)
            fg.m2 = d["m2"].astype(np.float32)
            fg.count = d["count"].astype(np.float32)
        if fg.count.shape != (grid.height, grid.width):
            raise ValueError("배열 크기 {}가 격자 정의 {}와 다릅니다".format(fg.count.shape, (grid.height, grid.width)))
        fg.runs = list(meta.get("runs") or [])
        return fg


def lidar_free_mask(map_yaml_path, grid, free_value=254):
    # type: (str, GridMeta, int) -> np.ndarray
    """LiDAR 지도(pgm)에서 '비어 있는(갈 수 있는) 칸'을 바닥 격자 크기로 키운 bool 마스크.

    map_server 저장 형식: 흰색(254) = 빈 곳, 검정 = 벽·장애물, 회색(205) = 모름.
    pgm은 위쪽 행이 +y 끝이므로 위아래를 뒤집어 격자 규칙(row = +y)에 맞춘다.
    """
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
    free = (img >= free_value).astype(np.uint8)[::-1]
    return cv2.resize(free, (grid.width, grid.height), interpolation=cv2.INTER_NEAREST).astype(bool)
