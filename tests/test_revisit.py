"""[4] 재방문 테스트: 방문 순서, 설 자리, 복원율 판정, 기준 갱신, 합성 재방문 전체 흐름."""

import itertools
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

from floor_dirt.compare import CompareResult
from floor_dirt.config import REPO_ROOT, load_params
from floor_dirt.floor_grid import FloorGrid
from floor_dirt.projection import GridMeta, WarpResult
from floor_dirt.revisit import (
    CLEANED, NEEDS_CLEANING, UNVERIFIED, RegionVerifier, free_space_checker, goal_pose_for, path_length,
    promote_reference, visit_order,
)

PARAMS = load_params()


# ---------------------------------------------------------------- 방문 순서

def test_visit_order_exhaustive_is_optimal():
    rng = np.random.RandomState(0)
    pts = [tuple(p) for p in rng.uniform(0, 5, (7, 2))]
    order, length = visit_order((0.0, 0.0), pts)
    brute = min(path_length((0.0, 0.0), pts, o) for o in itertools.permutations(range(7)))
    assert length == pytest.approx(brute) and sorted(order) == list(range(7))


def test_visit_order_many_points_and_empty():
    pts = [(float(i % 5), float(i // 5)) for i in range(15)]           # 3 x 5 격자
    order, length = visit_order((0.0, 0.0), pts)
    assert sorted(order) == list(range(15)) and length <= 15.0 + 1e-9  # 격자를 한 붓으로 (각 1 m)
    assert visit_order((0, 0), []) == ([], 0.0)


def test_visit_order_uses_custom_distance():
    """내비게이션 경로 길이 같은 거리 함수를 넣을 수 있다: A로 바로 가는 길이 막혀 B를 거쳐야 하는 경우."""
    A, B = (1.0, 0.0), (-1.0, 0.0)
    nav = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1]) + (10.0 if b == A and a != B else 0.0)
    assert visit_order((0.0, 0.0), [A, B])[0] in ([0, 1], [1, 0])      # 유클리드: 둘 다 3 m
    order, length = visit_order((0.0, 0.0), [A, B], dist_fn=nav)
    assert order == [1, 0] and length == pytest.approx(3.0)


# ---------------------------------------------------------------- 설 자리

def test_goal_pose_faces_region_at_standoff():
    g = goal_pose_for((2.0, 1.0), (0.0, 1.0), 0.35)
    assert g == pytest.approx((1.65, 1.0, 0.0))
    g = goal_pose_for((2.0, 1.0), (2.0, 3.0), 0.35)                    # 위에서 다가옴 → 아래를 봄
    assert g[0] == pytest.approx(2.0) and g[1] == pytest.approx(1.35) and g[2] == pytest.approx(-math.pi / 2)


def test_goal_pose_rotates_around_obstacles():
    blocked = lambda p: p[0] > 1.0                                     # x ≤ 1 쪽은 막힘
    g = goal_pose_for((1.2, 0.0), (0.0, 0.0), 0.35, is_free=blocked)
    assert g is not None and g[0] > 1.0
    assert math.hypot(1.2 - g[0], 0.0 - g[1]) == pytest.approx(0.35)
    assert goal_pose_for((1.2, 0.0), (0.0, 0.0), 0.35, is_free=lambda p: False) is None


def test_free_space_checker(tmp_path):
    pgm = np.full((40, 60), 254, np.uint8)                             # 3 x 2 m, 0.05 m
    pgm[:, 0] = pgm[:, -1] = pgm[0, :] = pgm[-1, :] = 0               # 벽
    pgm[20:30, 30:34] = 205                                           # 모르는 곳 (pgm 아래쪽 = y 작은 쪽)
    cv2.imwrite(str(tmp_path / "m.pgm"), pgm)
    (tmp_path / "m.yaml").write_text("image: m.pgm\nresolution: 0.05\norigin: [0, 0, 0]\nnegate: 0\n")
    free = free_space_checker(str(tmp_path / "m.yaml"), 0.22)
    assert free((0.6, 1.5)) and not free((0.1, 1.0)) and not free((3.5, 1.0))
    assert not free((1.6, 0.7))                                       # 모르는 곳 근처
    assert free((2.5, 1.6))


# ---------------------------------------------------------------- 복원율

def _frame(labels_shape, r0, c0, observed, hit):
    h, w = observed.shape
    wr = WarpResult(np.zeros((h, w, 3), np.uint8), np.ones((h, w), bool), r0, c0, np.eye(3))
    return wr, CompareResult(np.zeros((h, w), np.float32), np.ones((h, w), np.float32), observed, hit, 0.0)


def test_region_verifier_statuses():
    labels = np.zeros((20, 20), np.int32)
    labels[2:6, 2:6] = 1      # 16칸
    labels[10:14, 10:14] = 2  # 16칸
    labels[16:18, 0:2] = 3    # 안 보이는 구역
    v = RegionVerifier(labels)
    obs = np.ones((20, 20), bool)
    hit = np.zeros((20, 20), bool)
    hit[10:14, 10:12] = True              # 구역 2의 절반은 여전히 다름
    hit[2, 2] = True                      # 구역 1의 한 칸: 3장 중 1장만 다름 (노이즈)
    obs[16:18, 0:2] = False
    for k in range(3):
        h = hit.copy()
        if k:
            h[2, 2] = False
        ids = v.add(*_frame(labels.shape, 0, 0, obs, h))
    assert ids == [1, 2]
    r1, r2, r3 = v.results([1, 2, 3], restore_ratio=0.9, min_coverage=0.5, hit_frac=0.5)
    assert r1["status"] == CLEANED and r1["restoration_ratio"] == 1.0 and r1["frames"] == 3
    assert r2["status"] == NEEDS_CLEANING and r2["restoration_ratio"] == pytest.approx(0.5)
    assert r3["status"] == UNVERIFIED and r3["restoration_ratio"] is None and r3["coverage"] == 0.0


def test_promote_reference_only_cleaned_cells(tmp_path):
    grid = GridMeta(0.01, 0, 0, 10, 10)
    ref = FloorGrid(grid, 1)
    ref.add_patch(np.full((10, 10, 3), 100, np.float32), np.ones((10, 10), bool), 0, 0)
    ref.save(str(tmp_path / "reference"))
    new = FloorGrid(grid, 1)
    new.add_patch(np.full((10, 10, 3), 140, np.float32), np.ones((10, 10), bool), 0, 0)
    labels = np.zeros((10, 10), np.int32)
    labels[0:3, 0:3] = 1
    labels[5:8, 5:8] = 2
    info = promote_reference(str(tmp_path / "reference"), new, labels, cleaned_ids=[1])
    assert info["cells_updated"] == 9 and os.path.isdir(info["backup"])
    back = FloorGrid.load(info["backup"])
    assert (back.mean == 100).all()                                    # 백업은 그대로
    now = FloorGrid.load(str(tmp_path / "reference"))
    assert (now.mean[0:3, 0:3] == 140).all() and (now.mean[5:8, 5:8] == 100).all() and now.mean[9, 9, 0] == 100


# ---------------------------------------------------------------- 합성 재방문 전체 흐름

def test_synthetic_revisit_end_to_end(tmp_path):
    """정찰 → 계획 → 청소 후 재방문(2개 치움, 1개 남김) → 판정 → 기준 갱신."""
    from floor_dirt.offline import load_camera, plan_from_results, prepare_params, run_patrol, run_reference, run_revisit
    from floor_dirt.masks import ImageExcluder
    from floor_dirt.synthetic import ColorBlobDetector
    data = str(tmp_path / "data")
    r = subprocess.run([sys.executable, "scripts/make_synthetic_run.py", "--out", data], cwd=REPO_ROOT,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    params = prepare_params(load_params(), True)
    H, maps = load_camera(os.path.join(data, "synthetic_calib", "camera_intrinsics.yaml"),
                          os.path.join(data, "synthetic_calib", "floor_homography.yaml"))
    mp = os.path.join(data, "synthetic_map", "classroom.yaml")
    ref = str(tmp_path / "reference")
    run_reference(os.path.join(data, "synthetic_ref"), mp, ref, params, H, maps, results_dir=str(tmp_path / "rr"),
                  quiet=True)
    res = str(tmp_path / "patrol")
    exc = ImageExcluder(params["exclusion"], detectnet=ColorBlobDetector(margin_px=10))
    regions = run_patrol(os.path.join(data, "synthetic_patrol"), ref, res, params, H, maps, map_yaml=mp,
                         update_interval=0, quiet=True, excluder=exc)
    assert len(regions) == 3

    plan = plan_from_results(res, mp, (0.35, 1.32, 0.0), params)
    assert [p["step"] for p in plan] == [1, 2, 3] and all(p["goal"] is not None for p in plan)
    assert os.path.exists(os.path.join(res, "revisit_plan.png"))

    rep = run_revisit(os.path.join(data, "synthetic_revisit"), res, ref, params, H, maps, promote=True, quiet=True)
    truth = json.load(open(os.path.join(data, "synthetic_revisit", "truth_revisit.json"), encoding="utf-8"))["items"]
    for item in truth:
        reg = min(rep["regions"], key=lambda g: math.hypot(g["center"][0] - item["x"], g["center"][1] - item["y"]))
        assert reg["status"] == (CLEANED if item["cleaned"] else NEEDS_CLEANING), (item["name"], reg)
    assert rep["summary"] == {"cleaned": 2, "needs_cleaning": 1, "unverified": 0}
    assert rep["promote"]["cells_updated"] > 0 and os.path.isdir(rep["promote"]["backup"])
    assert len(rep["plan"]) == 3

    from floor_dirt.web import run_summary
    assert run_summary(str(tmp_path), "patrol")["revisit"]["summary"]["cleaned"] == 2


# ---------------------------------------------------------------- 재방문 진행 (노드 로직)

from floor_dirt.revisit import Mission  # noqa: E402


def _plan():
    return [{"step": 1, "id": 4, "goal": [1.0, 0.0, 0.0]},
            {"step": 2, "id": 7, "goal": None},                      # 설 자리 없음
            {"step": 3, "id": 2, "goal": [2.0, 1.0, 1.57]}]


def test_mission_sequence_and_failures():
    m = Mission(_plan(), n_shots=2, settle_s=1.0, shoot_timeout_s=5.0)
    assert m.next_goal(0.0) == (1.0, 0.0, 0.0) and m.state == Mission.NAVIGATING
    assert m.next_goal(0.1) is None                                  # 이동 중엔 새 목표 없음
    assert not m.want_frame(0.5)
    m.nav_done(True, 10.0)
    m.tick(10.5)
    assert m.state == Mission.SETTLING and not m.want_frame(10.6)    # 멈춘 직후는 흔들림 대기
    m.tick(11.0)
    assert m.state == Mission.SHOOTING
    assert not m.want_frame(10.9) and m.want_frame(11.1)             # 이동 중 찍힌 사진은 안 씀
    m.frame_done([9])                                                # 구역이 안 보인 사진
    m.frame_done([4])
    m.frame_done([4, 9])
    assert m.state == Mission.IDLE and m.log[0] == {"step": 1, "id": 4, "nav": "ok", "nav_s": 10.0,
                                                    "shots": 2, "missed": 1}
    # 설 자리 없는 구역은 건너뛰고 다음 구역 목표
    assert m.next_goal(12.0) == (2.0, 1.0, 1.57)
    assert m.log[1]["nav"] == "no_goal"
    m.nav_done(False, 20.0, "status 6")
    assert m.state == Mission.DONE and m.log[2]["nav"] == "failed: status 6"


def test_mission_shoot_timeout_and_empty():
    m = Mission([{"step": 1, "id": 1, "goal": [0, 0, 0]}], n_shots=5, settle_s=0.0, shoot_timeout_s=3.0)
    m.next_goal(0.0)
    m.nav_done(True, 1.0)
    m.tick(1.0)
    m.frame_done([1])
    m.tick(4.5)
    assert m.state == Mission.DONE and m.log[0]["timeout"] is True and m.log[0]["shots"] == 1
    assert Mission([]).state == Mission.DONE


def test_node_logic_with_synthetic_frames(tmp_path):
    """재방문 노드가 하는 일을 ROS 없이: 계획의 설 자리에서 찍은 합성 사진 → 진행 로직 → 판정."""
    from floor_dirt.dataset import Frame
    from floor_dirt.dirt_map import DirtMap
    from floor_dirt.pipeline import RevisitProcessor
    from floor_dirt.revisit import plan_revisit
    from floor_dirt.synthetic import FloorTexture, render, wide_camera

    grid = GridMeta(0.01, 0.0, 0.0, 300, 300)
    tex = np.full((1500, 1500, 3), (150, 160, 170), np.uint8)
    ref = FloorGrid(grid, 1)
    cam = wide_camera()
    H = cam.floor_homography()
    params = load_params()
    params["exclusion"] = dict(params["exclusion"], use_segnet=False, use_detectnet=False)
    from floor_dirt.projection import warp_to_grid
    for x in np.arange(0.4, 2.4, 0.05):
        for y in (1.0, 1.5, 2.0):
            p = (float(x), y, 0.0)
            ref.add_warp(warp_to_grid(render(cam, FloorTexture(tex, 0, 0, 0.002), p), H, p, grid, params["roi"]))
    # 정찰에서 찾은 두 구역 (한 곳은 청소됨, 한 곳은 아님)
    dm = DirtMap(grid, params["dirt_map"])
    for (cx, cy) in ((1.2, 1.0), (2.0, 2.0)):
        r0, c0 = int(cy / 0.01) - 3, int(cx / 0.01) - 3
        for _ in range(3):
            dm.update(r0, c0, np.ones((6, 6), bool), np.ones((6, 6), bool))
    regions, labels = dm.extract_regions()
    dirty = tex.copy()
    dirty[1990 // 2 - 15:1990 // 2 + 15, 2000 // 2 - 15:2000 // 2 + 15] = (40, 40, 200)   # (2.0, 2.0) 그대로
    plan = plan_revisit(regions, (0.4, 1.5, 0.0), float(params["revisit"]["standoff"]))
    rp = RevisitProcessor(params, ref, H, labels)
    m = Mission(plan, n_shots=3, settle_s=0.5, shoot_timeout_s=5.0)
    t, fid = 0.0, 0
    while m.state != Mission.DONE:
        goal = m.next_goal(t)
        m.nav_done(True, t + 5.0)                                   # 이동 성공 가정
        t += 5.0
        while m.state in (Mission.SETTLING, Mission.SHOOTING):
            t += 0.2
            m.tick(t)
            if m.want_frame(t):
                img = render(cam, FloorTexture(dirty, 0, 0, 0.002), goal)
                ids = rp.process(Frame(fid, t, goal, img, None))
                fid += 1
                m.frame_done(ids)
    res = {r["id"]: r for r in rp.verifier.results([g.id for g in regions], **{
        "restore_ratio": 0.9, "min_coverage": 0.5, "hit_frac": 0.5})}
    by_pos = {tuple(np.round(g.center, 1)): g.id for g in regions}
    assert res[by_pos[(1.2, 1.0)]]["status"] == CLEANED
    assert res[by_pos[(2.0, 2.0)]]["status"] == NEEDS_CLEANING
    assert all(e["shots"] == 3 and e["missed"] == 0 for e in m.log)  # 설 자리에서 구역이 매번 화면에 들어옴
