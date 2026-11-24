"""[4] 재방문 · 처리 여부 검증 (CLAUDE.md §5 [4]). ROS와 무관.

1) 방문 순서: 구역이 적으면(≤ 8) 모든 순서를 비교해 최단, 많으면 가까운 곳부터 + 2-opt로 다듬기.
   거리는 기본 유클리드. dist_fn을 바꿔 내비게이션 경로 길이를 넣을 수 있다.
2) 설 자리: 구역 중심이 로봇 앞 standoff(근거리 ROI 가운데)에 오도록, 다가오는 방향에서 바라보는 자세.
   그 자리가 지도상 벽·장애물에 너무 가까우면 바라보는 방향을 30°씩 돌려 다른 자리를 찾는다.
3) 복원율: 구역 칸 중 기준과 같다고 판정된 칸 / 이번에 본 구역 칸. ≥ restore_ratio면 "처리 완료".
   한 칸을 여러 장에서 보면 달라짐(hit) 비율이 hit_frac 미만일 때 "같음"으로 본다 (한 장의 노이즈에 덜 흔들림).
   구역을 min_coverage 미만으로 봤으면 "확인 불가".
4) 기준 갱신: 처리 완료된 구역 칸만 이번에 본 바닥으로 바꾸고, 기존 기준 폴더는 통째로 백업한다.
"""

import datetime
import itertools
import json
import math
import os
import shutil
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from floor_dirt.compare import CompareResult
from floor_dirt.floor_grid import FloorGrid
from floor_dirt.projection import WarpResult, window_slices

Point = Tuple[float, float]
Pose = Tuple[float, float, float]

CLEANED, NEEDS_CLEANING, UNVERIFIED = "cleaned", "needs_cleaning", "unverified"
STATUS_KO = {CLEANED: "처리 완료", NEEDS_CLEANING: "재청소 필요", UNVERIFIED: "확인 불가"}


def euclid(a, b):
    # type: (Point, Point) -> float
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ---------------------------------------------------------------- 1) 방문 순서

def path_length(start, points, order, dist_fn=euclid):
    total, cur = 0.0, start
    for i in order:
        total += dist_fn(cur, points[i])
        cur = points[i]
    return total


def _two_opt(start, points, order, dist_fn):
    best = list(order)
    best_len = path_length(start, points, best, dist_fn)
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                L = path_length(start, points, cand, dist_fn)
                if L < best_len - 1e-12:
                    best, best_len, improved = cand, L, True
    return best, best_len


def visit_order(start, points, dist_fn=euclid, max_exhaustive=8):
    # type: (Point, Sequence[Point], Callable, int) -> Tuple[List[int], float]
    """start에서 출발해 모든 점을 한 번씩 방문하는 최단(근사) 순서와 길이. 돌아오지 않는다."""
    n = len(points)
    if n == 0:
        return [], 0.0
    if n <= max_exhaustive:
        best = min(itertools.permutations(range(n)), key=lambda o: path_length(start, points, o, dist_fn))
        return list(best), path_length(start, points, best, dist_fn)
    left, order, cur = set(range(n)), [], start
    while left:
        nxt = min(left, key=lambda i: dist_fn(cur, points[i]))
        order.append(nxt)
        left.remove(nxt)
        cur = points[nxt]
    return _two_opt(start, points, order, dist_fn)


# ---------------------------------------------------------------- 2) 설 자리

def free_space_checker(map_yaml_path, clearance):
    # type: (str, float) -> Callable[[Point], bool]
    """LiDAR 지도에서 '벽·장애물·모르는 곳에서 clearance 이상 떨어진 빈 곳'인지 판정하는 함수."""
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
    res = float(info["resolution"])
    ox, oy = float(info["origin"][0]), float(info["origin"][1])
    free = (img >= 254).astype(np.uint8)[::-1]                 # row = +y (격자 규칙)
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5) * res   # 가장 가까운 빈 곳 아닌 칸까지 거리 [m]
    h, w = free.shape

    def is_free(p):
        c, r = int(math.floor((p[0] - ox) / res)), int(math.floor((p[1] - oy) / res))
        return 0 <= r < h and 0 <= c < w and dist[r, c] >= clearance
    return is_free


def goal_pose_for(center, approach_from, standoff, is_free=None, step_deg=30.0):
    # type: (Point, Point, float, Optional[Callable[[Point], bool]], float) -> Optional[Pose]
    """구역 중심이 로봇 전방 standoff에 오는 자세. 다가오는 방향 우선, 막혀 있으면 좌우로 돌려 찾는다."""
    base = math.atan2(center[1] - approach_from[1], center[0] - approach_from[0])
    if euclid(center, approach_from) < 1e-9:
        base = 0.0
    n = int(round(360.0 / step_deg))
    for k in range(n):
        off = math.radians(step_deg) * ((k + 1) // 2) * (1 if k % 2 else -1)
        th = math.atan2(math.sin(base + off), math.cos(base + off))
        pos = (center[0] - standoff * math.cos(th), center[1] - standoff * math.sin(th))
        if is_free is None or is_free(pos):
            return (pos[0], pos[1], th)
    return None


def plan_revisit(regions, start_pose, standoff, is_free=None, dist_fn=euclid):
    # type: (Sequence, Pose, float, Optional[Callable], Callable) -> List[Dict]
    """구역 목록 → 방문 순서대로 [{id, center, goal(x, y, yaw) 또는 None}]."""
    centers = [tuple(r.center) for r in regions]
    order, length = visit_order(start_pose[:2], centers, dist_fn)
    plan, cur = [], start_pose[:2]
    for k, i in enumerate(order):
        r = regions[i]
        goal = goal_pose_for(tuple(r.center), cur, standoff, is_free)
        plan.append({"step": k + 1, "id": r.id, "center": [round(float(v), 3) for v in r.center],
                     "goal": None if goal is None else [round(goal[0], 3), round(goal[1], 3), round(goal[2], 4)],
                     "area_m2": r.area_m2})
        if goal is not None:
            cur = goal[:2]
    return plan


# ---------------------------------------------------------------- 3) 복원율

class RegionVerifier:
    """재방문 사진들의 비교 결과를 구역 칸별로 누적해 복원율을 계산한다."""

    def __init__(self, labels):
        # type: (np.ndarray) -> None
        self.labels = labels.astype(np.int32)          # 격자 전체, 0 = 구역 아님, k = 구역 k
        self.obs = np.zeros(labels.shape, np.int32)
        self.hits = np.zeros(labels.shape, np.int32)
        self.frames_per_region = {}                    # type: Dict[int, int]

    def add(self, result, cmp):
        # type: (WarpResult, CompareResult) -> List[int]
        """한 장의 비교 결과 반영. 이 장에 보인 구역 번호들을 돌려준다."""
        rs, cs = window_slices(result)
        lab = self.labels[rs, cs]
        seen = cmp.observed & (lab > 0)
        self.obs[rs, cs] += seen
        self.hits[rs, cs] += cmp.hit & seen
        ids = sorted(int(v) for v in np.unique(lab[seen]))
        for i in ids:
            self.frames_per_region[i] = self.frames_per_region.get(i, 0) + 1
        return ids

    def region_result(self, region_id, restore_ratio=0.9, min_coverage=0.5, hit_frac=0.5):
        # type: (int, float, float, float) -> Dict
        cells = self.labels == region_id
        n = int(cells.sum())
        seen = cells & (self.obs > 0)
        n_seen = int(seen.sum())
        frac = np.zeros(self.obs.shape, np.float32)
        np.divide(self.hits, self.obs, out=frac, where=self.obs > 0)
        n_same = int((seen & (frac < hit_frac)).sum())
        coverage = n_seen / n if n else 0.0
        ratio = (n_same / n_seen) if n_seen else None
        if n == 0 or coverage < min_coverage:
            status = UNVERIFIED
        elif ratio >= restore_ratio:
            status = CLEANED
        else:
            status = NEEDS_CLEANING
        return {"id": int(region_id), "status": status, "status_ko": STATUS_KO[status],
                "restoration_ratio": None if ratio is None else round(ratio, 3),
                "coverage": round(coverage, 3), "cells": n, "cells_seen": n_seen, "cells_same": n_same,
                "frames": self.frames_per_region.get(int(region_id), 0)}

    def results(self, region_ids, **kw):
        return [self.region_result(i, **kw) for i in region_ids]


# ---------------------------------------------------------------- 4) 기준 갱신

def backup_reference(ref_dir):
    # type: (str) -> str
    """기준 폴더를 통째로 <ref_dir>_backup_<시각>으로 복사."""
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = os.path.normpath(ref_dir) + "_backup_" + stamp
    shutil.copytree(ref_dir, dst)
    return dst


def promote_reference(ref_dir, revisit_grid, labels, cleaned_ids, min_count=1, note=""):
    # type: (str, FloorGrid, np.ndarray, Sequence[int], int, str) -> Dict
    """처리 완료된 구역 칸만 재방문 때 본 바닥으로 바꿔 새 기준으로 저장 (기존 기준은 백업).

    처리 완료 = 기준과 거의 같아졌다는 뜻이지만, 청소 뒤 바닥 상태(물기 마름, 자국)를 반영해
    다음 정찰에서 같은 자리를 다시 오염으로 잡지 않게 한다. 재청소 필요·확인 불가 구역은 건드리지 않는다.
    """
    ref = FloorGrid.load(ref_dir)
    if ref.grid != revisit_grid.grid:
        raise ValueError("기준과 재방문 격자가 다릅니다")
    backup = backup_reference(ref_dir)
    target = np.isin(labels, list(cleaned_ids)) & (revisit_grid.count >= min_count)
    ref.mean[target] = revisit_grid.mean[target]
    ref.m2[target] = revisit_grid.m2[target]
    ref.count[target] = revisit_grid.count[target]
    ref.save(ref_dir, extra={"promoted_from_revisit": {
        "time": datetime.datetime.now().isoformat(timespec="seconds"), "regions": [int(i) for i in cleaned_ids],
        "cells": int(target.sum()), "backup": backup, "note": note}})
    return {"backup": backup, "cells_updated": int(target.sum())}


def write_report(path, plan, results, extra=None):
    data = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
            "summary": {s: sum(1 for r in results if r["status"] == s) for s in (CLEANED, NEEDS_CLEANING, UNVERIFIED)},
            "plan": plan, "regions": results}
    if extra:
        data.update(extra)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return data


# ---------------------------------------------------------------- 재방문 진행 (재방문 노드가 사용)

class Mission:
    """계획의 구역을 차례로: 이동 → 멈춰서 기다림(흔들림 가라앉기) → n장 촬영 → 다음 구역.

    시각은 모두 ROS 시각(초)으로 받는다. 이동 중에 찍힌 사진이 섞이지 않도록 '촬영 시작 시각 이후' 사진만 쓴다.
    이동 실패·설 자리 없음·시간 초과는 기록하고 다음 구역으로 넘어간다 (판정은 '확인 불가').
    """

    IDLE, NAVIGATING, SETTLING, SHOOTING, DONE = "idle", "navigating", "settling", "shooting", "done"

    def __init__(self, plan, n_shots=5, settle_s=1.0, shoot_timeout_s=15.0):
        self.plan = [p for p in plan]
        self.n_shots = int(n_shots)
        self.settle_s = float(settle_s)
        self.shoot_timeout_s = float(shoot_timeout_s)
        self.i = 0
        self.state = self.IDLE if self.plan else self.DONE
        self.t = None
        self.log = []      # type: List[Dict]
        self._cur = None

    def current(self):
        return self.plan[self.i] if self.i < len(self.plan) else None

    def _finish_step(self, **info):
        p = self.current()
        rec = {"step": p["step"], "id": p["id"]}
        rec.update(self._cur or {})
        rec.update(info)
        self.log.append(rec)
        self.i += 1
        self._cur = None
        self.state = self.IDLE if self.i < len(self.plan) else self.DONE

    def next_goal(self, now):
        """IDLE이면 다음 목표 자세를 돌려주고 NAVIGATING으로. 설 자리가 없는 구역은 건너뛴다."""
        while self.state == self.IDLE:
            p = self.current()
            if p.get("goal") is None:
                self._cur = {"nav": "no_goal"}
                self._finish_step(shots=0, missed=0)
                continue
            self.state, self.t = self.NAVIGATING, now
            self._cur = {"nav": "pending", "shots": 0, "missed": 0}
            return tuple(p["goal"])
        return None

    def nav_done(self, success, now, detail=""):
        if self.state != self.NAVIGATING:
            return
        if success:
            self._cur["nav"] = "ok"
            self._cur["nav_s"] = round(now - self.t, 1)
            self.state, self.t = self.SETTLING, now
        else:
            self._cur["nav"] = "failed" + (": " + detail if detail else "")
            self._finish_step()

    def tick(self, now):
        if self.state == self.SETTLING and now - self.t >= self.settle_s:
            self.state, self.t = self.SHOOTING, now
        elif self.state == self.SHOOTING and now - self.t >= self.shoot_timeout_s:
            self._finish_step(timeout=True)

    def want_frame(self, stamp):
        """이 시각에 찍힌 사진을 쓸지 (촬영 중이고, 멈춘 뒤에 찍힌 사진만)."""
        return self.state == self.SHOOTING and stamp >= self.t

    def frame_done(self, region_ids):
        """want_frame으로 받은 사진을 처리한 뒤 호출. 지금 구역이 사진에 보였으면 1장으로 센다."""
        if self.state != self.SHOOTING:
            return
        if self.current()["id"] in region_ids:
            self._cur["shots"] += 1
        else:
            self._cur["missed"] += 1
        if self._cur["shots"] >= self.n_shots:
            self._finish_step()
