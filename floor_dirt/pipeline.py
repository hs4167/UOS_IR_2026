"""프레임 처리 묶음: 기준 주행(ReferenceBuilder)과 정찰 주행(Patroller).

오프라인 스크립트(scripts/run_reference.py, run_patrol.py)와 ROS 노드가 **같은 클래스**를 쓴다.
입력은 dataset.Frame (원본 이미지 + 촬영 시점 자세 + LiDAR 점), ROS와 무관하다.

프레임 하나의 처리 순서 (CLAUDE.md §5 [2], [3])
  왜곡 보정 → 이미지 제외(몸체·segNet·detectNet) → 격자로 투영 → 격자 제외(LiDAR)
  → 기준: 누적 / 정찰: 비교 → 확률 갱신
단계별 시간은 StageTimer로 기록한다 (CLAUDE.md §7).
"""

import os
from typing import Callable, List, NamedTuple, Optional

import cv2
import numpy as np

from floor_dirt.compare import CompareResult, compare_warp
from floor_dirt.dataset import Frame
from floor_dirt.debug_stream import DebugOutput
from floor_dirt.dirt_map import DirtMap, Region, export_regions
from floor_dirt.floor_grid import FloorGrid
from floor_dirt.masks import Detection, ImageExcluder, window_keep_mask
from floor_dirt.projection import GridMeta, WarpResult, roi_corners_pixel, undistort, warp_to_grid, window_slices
from floor_dirt.timing import StageTimer
from floor_dirt.viz import debug_overlay, region_crop, render_dirt_map, save_png_atomic, window_to_image

REFERENCE_STAGES = ["undistort", "exclude_image", "warp", "exclude_grid", "accumulate", "debug"]
PATROL_STAGES = ["undistort", "exclude_image", "warp", "exclude_grid", "compare", "update", "debug"]


class Prepared(NamedTuple):
    image: np.ndarray                 # 왜곡 보정된 BGR
    warp: Optional[WarpResult]        # ROI가 격자 밖이면 None
    keep: Optional[np.ndarray]        # 창 칸 중 비교·누적에 쓸 칸 (valid 미포함)
    image_exclude: np.ndarray         # 이미지 제외 마스크 (디버그용)
    detections: List[Detection]


class PatrolStep(NamedTuple):
    frame_id: int
    warp: Optional[WarpResult]
    cmp: Optional[CompareResult]
    detections: List[Detection]
    timing: dict


class _FrameProcessor:
    def __init__(self, params, grid, H, stages, undistort_maps=None, excluder=None, timer=None, debug=None):
        # type: (dict, GridMeta, np.ndarray, List[str], Optional[tuple], Optional[Callable], Optional[StageTimer], Optional[DebugOutput]) -> None
        self.params = params
        self.grid = grid
        self.H = np.asarray(H, dtype=np.float64)
        self.roi = params["roi"]
        self.supersample = int(params["grid"].get("supersample", 1))
        self.lidar_radius = float(params["exclusion"]["lidar_radius"])
        self.lidar_shadow = float(params["exclusion"].get("lidar_shadow", 0.0))
        self.lidar_origin = tuple(params["exclusion"].get("lidar_origin", (0.0, 0.0)))
        self.maps = undistort_maps
        self.excluder = excluder if excluder is not None else ImageExcluder(params["exclusion"])
        self.timer = timer if timer is not None else StageTimer(stages)
        self.debug = debug if debug is not None else DebugOutput({"output": "none"})
        self.n_frames = 0

    def _prepare(self, frame):
        # type: (Frame) -> Prepared
        t = self.timer
        with t.stage("undistort"):
            img = undistort(frame.image, self.maps) if self.maps is not None else frame.image
        with t.stage("exclude_image"):
            ex, dets = self.excluder(img)
        with t.stage("warp"):
            w = warp_to_grid(img, self.H, frame.pose, self.grid, self.roi, supersample=self.supersample)
        keep = None
        if w is not None:
            with t.stage("exclude_grid"):
                keep = window_keep_mask(w, image_exclude=ex, lidar_points=frame.scan, pose=frame.pose,
                                        grid=self.grid, lidar_radius=self.lidar_radius,
                                        lidar_shadow=self.lidar_shadow, lidar_origin=self.lidar_origin)
        return Prepared(img, w, keep, ex, dets)

    def _debug(self, frame_id, prep, hit=None, lines=()):
        if not self.debug.wants():
            self.debug.skip()
            return
        with self.timer.stage("debug"):
            hit_px = window_to_image(hit, prep.warp, prep.image.shape) if (hit is not None and prep.warp is not None) else None
            ov = debug_overlay(prep.image, exclude_mask=prep.image_exclude,
                               roi_poly_px=roi_corners_pixel(self.H, self.roi), detections=prep.detections,
                               hit_mask=hit_px, text_lines=["frame {}".format(frame_id)] + list(lines))
            self.debug.push(frame_id, ov)

    def close(self):
        self.timer.close()
        self.debug.close()


class ReferenceBuilder(_FrameProcessor):
    """[2] 기준 주행: 프레임마다 바닥 색을 기준 격자에 누적."""

    def __init__(self, params, grid, H, **kw):
        super().__init__(params, grid, H, REFERENCE_STAGES, **kw)
        self.floor_grid = FloorGrid(grid, min_count=int(params["reference"]["min_count"]))

    def process(self, frame):
        # type: (Frame) -> dict
        prep = self._prepare(frame)
        n = 0
        if prep.warp is not None:
            with self.timer.stage("accumulate"):
                n = self.floor_grid.add_warp(prep.warp, prep.keep)
        self._debug(frame.frame_id, prep, lines=["reference +{} cells".format(n)])
        self.n_frames += 1
        timing = self.timer.end_frame(frame.frame_id)
        return {"frame_id": frame.frame_id, "cells": n, "timing": timing}

    def save(self, directory, run="", floor_mask=None):
        # type: (str, str, Optional[np.ndarray]) -> dict
        if run and run not in self.floor_grid.runs:
            self.floor_grid.runs.append(run)
        cov = self.floor_grid.coverage(floor_mask)
        self.floor_grid.save(directory, extra={"frames": self.n_frames, "coverage_floor": cov})
        save_png_atomic(os.path.join(directory, "coverage.png"), self.floor_grid.coverage_image(floor_mask))
        return cov


class Patroller(_FrameProcessor):
    """[3] 정찰 주행: 프레임마다 기준과 비교해 오염 확률을 갱신."""

    def __init__(self, params, reference, H, **kw):
        # type: (dict, FloorGrid, np.ndarray, ...) -> None
        super().__init__(params, reference.grid, H, PATROL_STAGES, **kw)
        self.reference = reference
        self.dirt_map = DirtMap(reference.grid, params["dirt_map"])
        self.path = []  # type: List[tuple]
        self.offsets = []  # type: List[float]

    def process(self, frame):
        # type: (Frame) -> PatrolStep
        prep = self._prepare(frame)
        cmp = None
        if prep.warp is not None:
            with self.timer.stage("compare"):
                cmp = compare_warp(prep.warp, self.reference, self.params["compare"], mask=prep.keep)
            with self.timer.stage("update"):
                self.dirt_map.update_compare(prep.warp, cmp, frame.frame_id, mask=prep.keep)
            self.offsets.append(cmp.offset)
        self._debug(frame.frame_id, prep, hit=None if cmp is None else cmp.hit,
                    lines=[] if cmp is None else ["light offset {:+.1f}".format(cmp.offset),
                                                  "hit {}".format(int(cmp.hit.sum()))])
        self.path.append(tuple(frame.pose))
        self.n_frames += 1
        timing = self.timer.end_frame(frame.frame_id)
        return PatrolStep(frame.frame_id, prep.warp, cmp, prep.detections, timing)

    def regions(self):
        # type: () -> tuple
        return self.dirt_map.extract_regions()

    def export(self, results_dir, run="", background=None, get_frame=None, scale=2):
        # type: (str, str, Optional[np.ndarray], Optional[Callable[[int], Frame]], int) -> List[Region]
        """dirt_regions.json, dirt_map.png, dirt_map.npz, 구역 사진(regions/region_XX.jpg)을 쓴다.

        get_frame(frame_id): 대표 사진용 원본 프레임 (데이터셋의 __getitem__ 등). 없으면 사진 생략.
        """
        regions, _ = self.regions()
        photos = {}
        if get_frame is not None:
            for r in regions:
                if r.best_frame < 0:
                    continue
                f = get_frame(r.best_frame)
                img = undistort(f.image, self.maps) if self.maps is not None else f.image
                w = warp_to_grid(img, self.H, f.pose, self.grid, self.roi)
                crop = region_crop(img, w, r) if w is not None else None
                if crop is not None:
                    rel = os.path.join("regions", "region_{:02d}.jpg".format(r.id))
                    os.makedirs(os.path.join(results_dir, "regions"), exist_ok=True)
                    cv2.imwrite(os.path.join(results_dir, rel), crop)
                    photos[r.id] = rel
        self.dirt_map.save(results_dir)
        img = render_dirt_map(self.dirt_map, regions, background=background,
                              robot_pose=self.path[-1] if self.path else None,
                              path_xy=np.array([p[:2] for p in self.path]) if len(self.path) > 1 else None,
                              scale=scale)
        save_png_atomic(os.path.join(results_dir, "dirt_map.png"), img)
        export_regions(os.path.join(results_dir, "dirt_regions.json"), regions, self.grid, run=run, extra={
            "frames": self.n_frames,
            "summary": self.dirt_map.summary(),
            "photos": {str(k): v for k, v in photos.items()},
            "light_offset_median": float(np.median(self.offsets)) if self.offsets else 0.0,
            "map_scale": scale,
        })
        return regions


REVISIT_STAGES = ["undistort", "exclude_image", "warp", "exclude_grid", "compare", "verify", "debug"]


class RevisitProcessor(_FrameProcessor):
    """[4] 재방문: 오염 구역 칸만 기준과 비교해 복원율을 쌓고, 기준 갱신용으로 새 바닥도 누적한다."""

    def __init__(self, params, reference, H, labels, **kw):
        # type: (dict, FloorGrid, np.ndarray, np.ndarray, ...) -> None
        super().__init__(params, reference.grid, H, REVISIT_STAGES, **kw)
        from floor_dirt.revisit import RegionVerifier
        self.reference = reference
        self.labels = labels
        self.verifier = RegionVerifier(labels)
        self.new_floor = FloorGrid(reference.grid, min_count=reference.min_count)

    def process(self, frame):
        # type: (Frame) -> List[int]
        """한 장 처리. 이 장에 보인 구역 번호들을 돌려준다."""
        prep = self._prepare(frame)
        ids = []
        cmp = None
        if prep.warp is not None:
            with self.timer.stage("compare"):
                cmp = compare_warp(prep.warp, self.reference, self.params["compare"], mask=prep.keep)
            with self.timer.stage("verify"):
                ids = self.verifier.add(prep.warp, cmp)
                if ids:
                    rs, cs = window_slices(prep.warp)
                    self.new_floor.add_warp(prep.warp, prep.keep & (self.labels[rs, cs] > 0))
        self._debug(frame.frame_id, prep, hit=None if cmp is None else cmp.hit,
                    lines=["regions in view: {}".format(ids or "-")])
        self.n_frames += 1
        self.timer.end_frame(frame.frame_id)
        return ids
