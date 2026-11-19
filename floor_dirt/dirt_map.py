"""[3] 오염 확률 지도: 칸별 log-odds 누적, 상태 판정, 오염 구역 추출·저장.

점유 격자 지도(Probabilistic Robotics 9장) 방식
- 관측된 칸마다 hit이면 l += log(p_hit / (1 - p_hit)), 아니면 l += log(p_miss / (1 - p_miss)).
- l은 [l_min, l_max]로 자른다 (한쪽으로 너무 굳어서 나중 관측이 안 먹히는 것 방지).
- 초기값 0 = 확률 0.5 (모름).

상태: 확인 불가(UNKNOWN) / 깨끗함(CLEAN) / 오염 의심(SUSPECT, p ≥ suspect_p).
min_observations 미만으로 본 칸은 확인 불가. 기준이 없는 칸은 비교에서 observed가 아니므로 자동으로 확인 불가.
"""

import datetime
import json
import math
import os
from typing import List, NamedTuple, Optional

import cv2
import numpy as np

from floor_dirt.compare import CompareResult
from floor_dirt.projection import GridMeta, WarpResult, cell_to_map

UNKNOWN, CLEAN, SUSPECT = 0, 1, 2
STATE_NAMES = {UNKNOWN: "unknown", CLEAN: "clean", SUSPECT: "suspect"}
NPZ_NAME = "dirt_map.npz"
REGIONS_NAME = "dirt_regions.json"


def logit(p):
    # type: (float) -> float
    return math.log(p / (1.0 - p))


class Region(NamedTuple):
    id: int
    center: tuple        # (x, y) 지도 좌표 [m], 칸 중심들의 평균
    size: tuple          # (x 폭, y 폭) [m], 칸 바운딩 박스 기준
    area_m2: float
    bbox_cells: tuple    # (r0, c0, r1, c1), r1·c1은 끝 다음 칸
    n_cells: int
    mean_p: float
    max_p: float
    best_frame: int      # 차이가 가장 컸던 프레임 (대표 사진용), 없으면 -1

    def to_json(self):
        d = self._asdict()
        d["center"] = [round(float(v), 4) for v in self.center]
        d["size"] = [round(float(v), 4) for v in self.size]
        d["area_m2"] = round(float(self.area_m2), 6)
        d["bbox_cells"] = [int(v) for v in self.bbox_cells]
        d["mean_p"] = round(float(self.mean_p), 4)
        d["max_p"] = round(float(self.max_p), 4)
        return d


class DirtMap:
    """칸별 오염 확률. 배열 인덱스 규칙은 projection.GridMeta와 같다 ([row, col], row = +y)."""

    def __init__(self, grid, cfg):
        # type: (GridMeta, dict) -> None
        self.grid = grid
        self.cfg = dict(cfg)
        self.l_hit = logit(float(cfg["p_hit"]))
        self.l_miss = logit(float(cfg["p_miss"]))
        self.l_min = float(cfg["l_min"])
        self.l_max = float(cfg["l_max"])
        self.suspect_p = float(cfg["suspect_p"])
        self.min_obs = int(cfg.get("min_observations", 1))
        if not (self.l_hit > 0 > self.l_miss):
            raise ValueError("p_miss < 0.5 < p_hit 이어야 합니다")
        shape = (grid.height, grid.width)
        self.logodds = np.zeros(shape, np.float32)
        self.n_obs = np.zeros(shape, np.int32)
        self.n_hit = np.zeros(shape, np.int32)
        self.n_seen = np.zeros(shape, np.int32)        # 화면에 찍힌 횟수 (기준 유무 무관, 제외 영역 빼고)
        self.best_diff = np.zeros(shape, np.float32)   # hit이었던 관측 중 (차이 / 임계값) 최대
        self.best_frame = np.full(shape, -1, np.int32)

    # ------------------------------------------------------------ 갱신

    def update(self, r0, c0, observed, hit, score=None, frame_id=-1, seen=None):
        # type: (int, int, np.ndarray, np.ndarray, Optional[np.ndarray], int, Optional[np.ndarray]) -> None
        """창 [r0:, c0:]의 관측 결과 반영. score = 대표 사진 선택용 점수 (클수록 뚜렷).

        seen = 화면에 찍혔고 제외되지 않은 칸 (기준이 없어도 True). 주지 않으면 observed와 같다.
        '봤는데 기준이 없는 칸'을 시각화에서 '안 간 곳'과 구분하는 데 쓴다.
        """
        h, w = observed.shape
        if r0 < 0 or c0 < 0 or r0 + h > self.grid.height or c0 + w > self.grid.width:
            raise ValueError("창이 격자 밖으로 나갑니다: r0={} c0={} size={}".format(r0, c0, (h, w)))
        rs, cs = slice(r0, r0 + h), slice(c0, c0 + w)
        obs = observed.astype(bool)
        hit = hit.astype(bool) & obs
        l = self.logodds[rs, cs]
        l += np.where(hit, self.l_hit, np.where(obs, self.l_miss, 0.0)).astype(np.float32)
        np.clip(l, self.l_min, self.l_max, out=l)
        self.n_obs[rs, cs] += obs
        self.n_hit[rs, cs] += hit
        self.n_seen[rs, cs] += obs if seen is None else (seen.astype(bool) | obs)
        if score is not None and frame_id >= 0:
            bd = self.best_diff[rs, cs]
            better = hit & (score > bd)
            bd[better] = score[better]
            self.best_frame[rs, cs][better] = frame_id

    def update_compare(self, result, cmp, frame_id=-1, mask=None):
        # type: (WarpResult, CompareResult, int, Optional[np.ndarray]) -> None
        """compare_warp 결과를 그대로 반영. mask = compare_warp에 넘긴 것과 같은 제외 마스크."""
        with np.errstate(divide="ignore", invalid="ignore"):
            score = np.where(cmp.thr > 0, cmp.diff / cmp.thr, 0.0).astype(np.float32)
        seen = result.valid if mask is None else (result.valid & mask)
        self.update(result.r0, result.c0, cmp.observed, cmp.hit, score, frame_id, seen=seen)

    # ------------------------------------------------------------ 조회

    def probability(self):
        # type: () -> np.ndarray
        return (1.0 / (1.0 + np.exp(-self.logodds))).astype(np.float32)

    def states(self):
        # type: () -> np.ndarray
        st = np.full(self.logodds.shape, CLEAN, np.uint8)
        # p ≥ suspect_p  ⇔  l ≥ logit(suspect_p). 부동소수 경계 문제를 피하려고 약간의 여유를 둔다
        st[self.logodds >= logit(self.suspect_p) - 1e-4] = SUSPECT
        st[self.n_obs < max(self.min_obs, 1)] = UNKNOWN
        return st

    def summary(self):
        # type: () -> dict
        st = self.states()
        a = self.grid.resolution ** 2
        return {STATE_NAMES[k] + "_m2": round(float((st == k).sum() * a), 4) for k in (UNKNOWN, CLEAN, SUSPECT)}

    def extract_regions(self, min_area_m2=None):
        # type: (Optional[float]) -> tuple
        """오염 의심 칸을 8방향 연결로 묶어 구역 목록과 칸별 구역 번호(0 = 없음)를 돌려준다.

        구역은 넓이 큰 순으로 1번부터 번호를 매긴다. min_area_m2 미만은 버린다.
        """
        if min_area_m2 is None:
            min_area_m2 = float(self.cfg.get("min_region_area", 0.0))
        suspect = (self.states() == SUSPECT).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(suspect, connectivity=8)
        res = self.grid.resolution
        prob = self.probability()
        found = []
        for lab in range(1, n):
            n_cells = int(stats[lab, cv2.CC_STAT_AREA])
            if n_cells * res * res < min_area_m2 - 1e-12:
                continue
            x, y, w, h = (int(stats[lab, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                                        cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
            m = labels[y:y + h, x:x + w] == lab
            rr, cc = np.nonzero(m)
            center = cell_to_map(self.grid, [[rr.mean() + y, cc.mean() + x]])[0]
            p = prob[y:y + h, x:x + w][m]
            bd = self.best_diff[y:y + h, x:x + w][m]
            bf = self.best_frame[y:y + h, x:x + w][m]
            best = int(bf[int(np.argmax(bd))]) if len(bd) else -1
            found.append((n_cells, lab, Region(0, (float(center[0]), float(center[1])), (w * res, h * res),
                                                n_cells * res * res, (y, x, y + h, x + w), n_cells,
                                                float(p.mean()), float(p.max()), best)))
        found.sort(key=lambda t: -t[0])
        out_labels = np.zeros(labels.shape, np.int32)
        regions = []  # type: List[Region]
        for new_id, (_, lab, reg) in enumerate(found, start=1):
            out_labels[labels == lab] = new_id
            regions.append(reg._replace(id=new_id))
        return regions, out_labels

    # ------------------------------------------------------------ 저장

    def save(self, directory):
        # type: (str) -> None
        os.makedirs(directory, exist_ok=True)
        tmp = os.path.join(directory, NPZ_NAME + ".tmp.npz")
        np.savez_compressed(tmp, logodds=self.logodds, n_obs=self.n_obs, n_hit=self.n_hit, n_seen=self.n_seen,
                            best_diff=self.best_diff, best_frame=self.best_frame,
                            grid=np.array(self.grid, dtype=np.float64))
        os.replace(tmp, os.path.join(directory, NPZ_NAME))

    @classmethod
    def load(cls, directory, cfg):
        # type: (str, dict) -> DirtMap
        with np.load(os.path.join(directory, NPZ_NAME)) as d:
            g = d["grid"]
            grid = GridMeta(float(g[0]), float(g[1]), float(g[2]), int(g[3]), int(g[4]))
            dm = cls(grid, cfg)
            for k in ("logodds", "n_obs", "n_hit", "n_seen", "best_diff", "best_frame"):
                if k in d:
                    setattr(dm, k, d[k].copy())
        return dm


def export_regions(path, regions, grid, run="", extra=None):
    # type: (str, List[Region], GridMeta, str, Optional[dict]) -> None
    """results/<run>/dirt_regions.json. 웹 화면과 재방문 단계가 읽는다. 원자적으로 교체."""
    data = {
        "run": run,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "grid": dict(grid._asdict()),
        "n_regions": len(regions),
        "regions": [r.to_json() for r in regions],
    }
    if extra:
        data.update(extra)
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_regions(path):
    # type: (str) -> List[Region]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Region(int(r["id"]), tuple(r["center"]), tuple(r["size"]), float(r["area_m2"]), tuple(r["bbox_cells"]),
                   int(r["n_cells"]), float(r["mean_p"]), float(r["max_p"]), int(r["best_frame"]))
            for r in data["regions"]]
