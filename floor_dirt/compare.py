"""[3] 기준과 현재 바닥 비교: 칸별 차이, 칸별 임계값, '달라짐(hit)' 판정.

비교 방식 (params compare.method)
- sad_gray: 밝기(L) 절대 차이. 기준선. 바닥 무늬와 밝기가 같은 색종이는 놓친다.
- delta_e: Lab 전체 거리 (밝기 + 색). 회색 얼룩과 색종이를 모두 잡는다.
- chroma_diff: 색도(a, b)만 비교. 밝기 변화에 강하지만 회색 먼지·검은 얼룩처럼 색이 없는 오염은 놓친다.

조명 보정 (params compare.offset_correction, sad_gray·delta_e에 적용)
- 창 안에서 (현재 L - 기준 L)의 중앙값을 조명 변화로 보고 뺀다.
- 중앙값이라 오염이 창의 절반 미만이면 거의 흔들리지 않는다.
  (처음 시도한 '주변 평균 밝기 빼기'는 어두운 얼룩이 주변 평균을 끌어내려 얼룩 둘레가 오탐됐다 → docs/실험_기록.md)

어긋남 흡수: 기준을 주변 (2r+1)x(2r+1) 칸으로 밀어 보며 가장 작은 차이를 쓴다
(위치 추정 오차로 1~2칸 어긋나도 타일 이음새 같은 무늬를 오염으로 보지 않게).
임계값: max(k * sqrt(기준 분산), min_thr). 분산은 가장 잘 맞은 칸의 것을 쓴다.
"""

from typing import NamedTuple, Optional

import numpy as np

from floor_dirt.floor_grid import FloorGrid, Reference, to_lab
from floor_dirt.projection import WarpResult

METHODS = ("sad_gray", "delta_e", "chroma_diff")
MIN_CELLS_FOR_OFFSET = 50  # 조명 보정에 필요한 최소 관측 칸 수


class CompareResult(NamedTuple):
    diff: np.ndarray       # (h, w) float32, 관측 안 된 칸은 0
    thr: np.ndarray        # (h, w) float32
    observed: np.ndarray   # (h, w) bool, 이번 프레임에서 보았고 기준도 있는 칸
    hit: np.ndarray        # (h, w) bool, observed & diff > thr
    offset: float          # 조명 보정으로 뺀 L 값 (보정 안 하면 0). 조명 변화 기록용


def features(lab, method):
    # type: (np.ndarray, str) -> np.ndarray
    """비교할 특징 (h, w, C)."""
    if method == "sad_gray":
        return lab[..., 0:1].astype(np.float32)
    if method == "delta_e":
        return lab.astype(np.float32)
    if method == "chroma_diff":
        return lab[..., 1:3].astype(np.float32)
    raise ValueError("알 수 없는 비교 방식: {} (가능: {})".format(method, ", ".join(METHODS)))


def feature_var(var, method):
    # type: (np.ndarray, str) -> np.ndarray
    """특징 거리에 맞는 기준 분산 (h, w). 여러 채널 거리는 채널 분산의 합."""
    if method == "sad_gray":
        return var[..., 0].astype(np.float32)
    if method == "chroma_diff":
        return (var[..., 1] + var[..., 2]).astype(np.float32)
    return var.sum(axis=-1).astype(np.float32)


def compare_window(obs_lab, obs_mask, ref, method="delta_e", neighborhood=1, k=3.0, min_thr=8.0,
                   offset_correction=False):
    # type: (np.ndarray, np.ndarray, Reference, str, int, float, float, bool) -> CompareResult
    """창 하나 비교. ref는 창보다 사방으로 neighborhood 칸 큰 기준 (reference_window로 얻는다)."""
    r = int(neighborhood)
    h, w = obs_mask.shape
    if ref.mean.shape[:2] != (h + 2 * r, w + 2 * r):
        raise ValueError("기준 창 크기 {}가 관측 {} + 여백 {}와 맞지 않습니다".format(ref.mean.shape[:2], (h, w), r))

    center_valid = ref.valid[r:r + h, r:r + w]
    observed = obs_mask.astype(bool) & center_valid
    f_obs = features(obs_lab, method)
    f_ref = features(ref.mean, method)
    v_ref = feature_var(ref.var, method)

    offset = 0.0
    if offset_correction and method != "chroma_diff" and observed.sum() >= MIN_CELLS_FOR_OFFSET:
        dL = obs_lab[..., 0][observed] - ref.mean[r:r + h, r:r + w, 0][observed]
        offset = float(np.median(dL))
        f_obs = f_obs.copy()
        f_obs[..., 0] -= offset  # L은 두 방식 모두 0번 채널

    best = np.full((h, w), np.inf, np.float32)
    best_var = np.zeros((h, w), np.float32)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            rs, cs = slice(r + dy, r + dy + h), slice(r + dx, r + dx + w)
            d = np.sqrt(np.sum((f_obs - f_ref[rs, cs]) ** 2, axis=-1))
            d[~ref.valid[rs, cs]] = np.inf
            better = d < best
            best[better] = d[better]
            best_var[better] = v_ref[rs, cs][better]

    thr = np.maximum(k * np.sqrt(best_var), min_thr).astype(np.float32)
    diff = np.where(observed, best, 0.0).astype(np.float32)
    hit = observed & (diff > thr)
    return CompareResult(diff=diff, thr=thr, observed=observed, hit=hit, offset=offset)


def reference_window(fg, r0, c0, h, w, margin):
    # type: (FloorGrid, int, int, int, int, int) -> Reference
    """격자 [r0:r0+h, c0:c0+w] 창을 사방 margin 칸 넓혀 자른 기준. 격자 밖은 '기준 없음'."""
    H, W = fg.grid.height, fg.grid.width
    R0, C0, R1, C1 = r0 - margin, c0 - margin, r0 + h + margin, c0 + w + margin
    src = fg.reference(slice(max(R0, 0), min(R1, H)), slice(max(C0, 0), min(C1, W)))
    if (R0, C0, R1, C1) == (max(R0, 0), max(C0, 0), min(R1, H), min(C1, W)):
        return src
    hh, ww = R1 - R0, C1 - C0
    pad = (max(0, -R0), max(0, -C0))
    rs = slice(pad[0], pad[0] + src.count.shape[0])
    cs = slice(pad[1], pad[1] + src.count.shape[1])
    mean = np.zeros((hh, ww, 3), np.float32)
    var = np.zeros((hh, ww, 3), np.float32)
    count = np.zeros((hh, ww), np.float32)
    valid = np.zeros((hh, ww), bool)
    mean[rs, cs], var[rs, cs], count[rs, cs], valid[rs, cs] = src.mean, src.var, src.count, src.valid
    return Reference(mean, var, count, valid)


def compare_warp(result, fg, cfg, mask=None):
    # type: (WarpResult, FloorGrid, dict, Optional[np.ndarray]) -> CompareResult
    """warp_to_grid 결과를 기준 격자와 비교. cfg = params["compare"]. mask = 추가로 남길 칸 (제외 마스크 반영)."""
    r = int(cfg.get("neighborhood", 1))
    h, w = result.valid.shape
    ref = reference_window(fg, result.r0, result.c0, h, w, r)
    obs_mask = result.valid if mask is None else (result.valid & mask)
    return compare_window(to_lab(result.patch), obs_mask, ref, method=cfg["method"], neighborhood=r,
                          k=float(cfg["k"]), min_thr=float(cfg["min_thr"]),
                          offset_correction=bool(cfg.get("offset_correction", False)))
