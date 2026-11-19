"""합성 E2E: 기준 주행 → 정찰 주행 → 오염 구역. 실기로 넘어가기 전 통과 기준 (계획 5단계).

장면: 30 cm 타일(2 mm 이음새, 타일별 색 차이), 오래된 얼룩, 미세 무늬,
정찰 때 조명 10% 밝아짐, 위치 오차(기준 3 mm / 정찰 4 mm, 방향 0.3~0.4도), 센서 노이즈,
LiDAR에 걸리는 가방, detectNet이 찾는 신발(보라색), 찾아야 할 오염 3개.
통과 기준: 오염 3개 모두 검출, 그 외 구역 0개, 위치 오차 ≤ 2 cm.
"""

import csv
import json
import math
import os

import cv2
import numpy as np
import pytest

from floor_dirt.config import load_params
from floor_dirt.dataset import Frame, FrameDataset, FrameDatasetWriter
from floor_dirt.floor_grid import FloorGrid
from floor_dirt.masks import ImageExcluder
from floor_dirt.pipeline import PATROL_STAGES, Patroller, ReferenceBuilder
from floor_dirt.projection import GridMeta, map_to_robot
from floor_dirt.timing import StageTimer
from synthetic import (
    ColorBlobDetector, add_sensor_noise, classroom_scene, item_outline, noisy, render, wide_camera,
)

GRID = GridMeta(0.01, -0.5, 0.0, 400, 300)  # x -0.5~3.5, y 0~3 m


def _params():
    p = load_params()
    p["exclusion"] = dict(p["exclusion"], use_segnet=False, use_detectnet=False)  # WSL: jetson-inference 없음
    return p


def _record(run_dir, scene, texture, rng, xy_sigma, yaw_sigma, with_objects):
    """주행을 프레임 데이터셋으로 저장 (실제 rosbag 추출 결과와 같은 형식)."""
    cam = wide_camera()
    outline = item_outline(scene.bag) if with_objects else None
    with FrameDatasetWriter(run_dir, {"source": "synthetic"}) as wr:
        for i, true_pose in enumerate(scene.poses):
            img = add_sensor_noise(render(cam, texture, true_pose), rng)
            scan = map_to_robot(true_pose, outline).astype(np.float32) if outline is not None else None
            wr.write(Frame(i, i * 0.2, noisy(true_pose, rng, xy_sigma, yaw_sigma), img, scan))


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    base = tmp_path_factory.mktemp("e2e")
    params = _params()
    scene = classroom_scene(seed=0)
    rng = np.random.RandomState(1)
    H = wide_camera().floor_homography()

    _record(str(base / "data" / "ref"), scene, scene.clean, rng, 0.003, 0.3, with_objects=False)
    _record(str(base / "data" / "patrol"), scene, scene.dirty, rng, 0.004, 0.4, with_objects=True)

    rb = ReferenceBuilder(params, GRID, H)
    for f in FrameDataset(str(base / "data" / "ref")):
        rb.process(f)
    rb.save(str(base / "maps" / "reference"), run="ref")
    rb.close()

    reference = FloorGrid.load(str(base / "maps" / "reference"))
    ds = FrameDataset(str(base / "data" / "patrol"))
    timing_csv = str(base / "results" / "patrol" / "timing.csv")
    os.makedirs(os.path.dirname(timing_csv))
    excluder = ImageExcluder(params["exclusion"], detectnet=ColorBlobDetector(margin_px=10))
    pt = Patroller(params, reference, H, excluder=excluder, timer=StageTimer(PATROL_STAGES, timing_csv))
    detections = 0
    for f in ds:
        step = pt.process(f)
        detections += len(step.detections)
    regions = pt.export(str(base / "results" / "patrol"), run="patrol", get_frame=ds.__getitem__)
    pt.close()
    return dict(base=base, scene=scene, regions=regions, patroller=pt, detections=detections, n=len(ds))


def _nearest(regions, item):
    return min(math.hypot(r.center[0] - item.x, r.center[1] - item.y) for r in regions)


def test_all_dirt_found_within_2cm(e2e):
    for item in e2e["scene"].dirt:
        assert _nearest(e2e["regions"], item) <= 0.02, item.name


def test_no_false_regions(e2e):
    # 이음새·타일 색 차이·오래된 얼룩·조명 변화·가방(LiDAR)·신발(detectNet)은 구역이 되면 안 된다
    assert len(e2e["regions"]) == len(e2e["scene"].dirt), [(r.id, r.center) for r in e2e["regions"]]


def test_objects_were_actually_excluded(e2e):
    sc, pt = e2e["scene"], e2e["patroller"]
    assert e2e["detections"] > 0                                    # 신발이 검출된 프레임이 있었음
    for item in (sc.bag, sc.shoes):
        c = int((item.x - GRID.origin_x) / GRID.resolution), int((item.y - GRID.origin_y) / GRID.resolution)
        assert pt.dirt_map.n_obs[c[1], c[0]] == 0, item.name         # 물체 칸은 한 번도 비교되지 않음


def test_results_files(e2e):
    out = e2e["base"] / "results" / "patrol"
    data = json.loads((out / "dirt_regions.json").read_text(encoding="utf-8"))
    assert data["n_regions"] == len(e2e["regions"]) and data["frames"] == e2e["n"]
    assert data["light_offset_median"] > 5                          # 조명 10% 밝아진 것이 기록됨
    assert len(data["photos"]) == data["n_regions"]
    for rel in data["photos"].values():
        assert cv2.imread(str(out / rel)) is not None
    img = cv2.imread(str(out / "dirt_map.png"))
    assert img is not None and img.shape[0] == GRID.height * data["map_scale"]
    with open(str(out / "timing.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == e2e["n"] and float(rows[0]["total_ms"]) > 0
    ref = e2e["base"] / "maps" / "reference"
    assert (ref / "reference.npz").exists() and (ref / "coverage.png").exists()
