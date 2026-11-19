import math

import numpy as np
import pytest

from floor_dirt.compare import compare_warp
from floor_dirt.config import load_params
from floor_dirt.dirt_map import CLEAN, SUSPECT, UNKNOWN, DirtMap, export_regions, load_regions, logit
from floor_dirt.floor_grid import FloorGrid
from floor_dirt.projection import GridMeta, map_to_cell, warp_to_grid
from synthetic import FloorTexture, render, wide_camera

PARAMS = load_params()
CFG = PARAMS["dirt_map"]
GRID = GridMeta(0.01, 0.0, 0.0, 50, 40)


def _ones(h, w):
    return np.ones((h, w), bool)


# ---------------------------------------------------------------- log-odds

def test_logodds_accumulation_and_clamp():
    dm = DirtMap(GRID, CFG)
    obs, hit, miss = _ones(1, 1), _ones(1, 1), np.zeros((1, 1), bool)
    dm.update(0, 0, obs, hit)
    assert dm.logodds[0, 0] == pytest.approx(logit(0.7), abs=1e-6)
    dm.update(0, 0, obs, miss)
    assert dm.logodds[0, 0] == pytest.approx(logit(0.7) + logit(0.4), abs=1e-6)
    for _ in range(20):
        dm.update(0, 0, obs, hit)
    assert dm.logodds[0, 0] == pytest.approx(CFG["l_max"])
    for _ in range(40):
        dm.update(0, 0, obs, miss)
    assert dm.logodds[0, 0] == pytest.approx(CFG["l_min"])
    # 상한에서도 미스 7번이면 의심이 풀린다 (굳지 않음): (3.5 - logit(0.7)) / -logit(0.4) ≈ 6.5
    for _ in range(20):
        dm.update(0, 0, obs, hit)
    for _ in range(6):
        dm.update(0, 0, obs, miss)
    assert dm.states()[0, 0] == SUSPECT
    dm.update(0, 0, obs, miss)
    assert dm.states()[0, 0] == CLEAN


def test_unobserved_cells_untouched_and_hit_requires_observed():
    dm = DirtMap(GRID, CFG)
    obs = np.zeros((3, 3), bool)
    obs[1, 1] = True
    hit = _ones(3, 3)  # 관측 안 된 칸의 hit은 무시
    dm.update(5, 5, obs, hit)
    assert dm.n_obs.sum() == 1 and dm.n_hit.sum() == 1
    assert dm.logodds[6, 6] > 0 and (dm.logodds[5:8, 5:8] != 0).sum() == 1
    with pytest.raises(ValueError):
        dm.update(39, 0, obs, hit)


def test_states_unknown_clean_suspect():
    dm = DirtMap(GRID, CFG)
    o = _ones(1, 1)
    dm.update(0, 0, o, o)                       # hit 1번: p = 0.7 이지만 관측 1번 → 확인 불가
    dm.update(0, 1, o, o)
    dm.update(0, 1, o, o)                       # hit 2번 → 의심
    dm.update(0, 2, o, ~o)
    dm.update(0, 2, o, ~o)                      # miss 2번 → 깨끗
    dm.update(0, 3, o, o)
    dm.update(0, 3, o, ~o)                      # hit 1 + miss 1 → p ≈ 0.61 → 깨끗
    st = dm.states()
    assert st[0, 0] == UNKNOWN and st[0, 1] == SUSPECT and st[0, 2] == CLEAN and st[0, 3] == CLEAN
    assert st[10, 10] == UNKNOWN
    s = dm.summary()
    assert s["suspect_m2"] == pytest.approx(0.0001) and s["clean_m2"] == pytest.approx(0.0002)


def test_invalid_probabilities_rejected():
    with pytest.raises(ValueError):
        DirtMap(GRID, dict(CFG, p_hit=0.4))


# ---------------------------------------------------------------- 구역

def _paint(dm, r0, c0, h, w, times=3, frame=0, score=2.0):
    for _ in range(times):
        dm.update(r0, c0, _ones(h, w), _ones(h, w), np.full((h, w), score, np.float32), frame)


def test_extract_regions_sizes_centers_and_min_area():
    dm = DirtMap(GRID, CFG)
    dm.update(0, 0, _ones(40, 50), np.zeros((40, 50), bool))   # 전부 한 번 관측(깨끗)
    dm.update(0, 0, _ones(40, 50), np.zeros((40, 50), bool))
    _paint(dm, 5, 5, 6, 8, frame=3)            # 6 x 8칸 = 0.0048 m²
    _paint(dm, 20, 30, 4, 4, frame=7)          # 4 x 4칸 = 0.0016 m²
    _paint(dm, 35, 45, 1, 1, frame=9)          # 1칸 = 최소 면적(0.0004) 미만 → 버림
    _paint(dm, 21, 30, 1, 1, frame=8, score=9.0)  # 두 번째 구역 안의 더 뚜렷한 관측
    regions, labels = dm.extract_regions()
    assert [r.id for r in regions] == [1, 2]
    r1, r2 = regions
    assert r1.n_cells == 48 and r1.area_m2 == pytest.approx(0.0048)
    assert r1.size == pytest.approx((0.08, 0.06))
    assert r1.center == pytest.approx((0.05 + 0.04, 0.05 + 0.03))   # 칸 5~12 중심 평균
    assert r1.bbox_cells == (5, 5, 11, 13) and r1.best_frame == 3
    assert r2.n_cells == 16 and r2.best_frame == 8
    assert (labels == 1).sum() == 48 and (labels == 2).sum() == 16 and labels[35, 45] == 0
    assert r1.mean_p >= CFG["suspect_p"]


def test_diagonal_cells_join_one_region():
    dm = DirtMap(GRID, CFG)
    for i in range(5):
        _paint(dm, 10 + i, 10 + i, 1, 1)
    regions, _ = dm.extract_regions(min_area_m2=0.0)
    assert len(regions) == 1 and regions[0].n_cells == 5


def test_save_load_and_json(tmp_path):
    dm = DirtMap(GRID, CFG)
    _paint(dm, 5, 5, 6, 8, frame=3)
    dm.save(str(tmp_path / "run"))
    back = DirtMap.load(str(tmp_path / "run"), CFG)
    assert back.grid == GRID
    np.testing.assert_array_equal(back.logodds, dm.logodds)
    np.testing.assert_array_equal(back.best_frame, dm.best_frame)

    regions, _ = dm.extract_regions()
    path = str(tmp_path / "run" / "dirt_regions.json")
    export_regions(path, regions, GRID, run="test", extra={"summary": dm.summary()})
    loaded = load_regions(path)
    assert len(loaded) == 1 and loaded[0].n_cells == 48 and loaded[0].best_frame == 3
    assert loaded[0].center == pytest.approx(regions[0].center, abs=1e-4)


# ---------------------------------------------------------------- 합성 정찰

def test_synthetic_patrol_produces_one_region_at_paper():
    """기준(깨끗) → 정찰(빨간 종이 6 cm, 조명 15% 밝음, 위치 오차): 구역 1개, 위치 오차 ≤ 2 cm."""
    grid = GridMeta(0.01, 0.0, 0.0, 300, 300)
    res_t = 0.002
    tex = np.full((1500, 1500, 3), (150, 160, 170), np.uint8)
    tex[::50, :] = (110, 115, 120)
    tex[:, ::50] = (110, 115, 120)
    dirty = np.clip(tex.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)
    paper = (1.43, 1.47)
    c = (int(paper[0] / res_t), int(paper[1] / res_t))
    dirty[c[1] - 15:c[1] + 15, c[0] - 15:c[0] + 15] = (50, 50, 215)

    cam = wide_camera()
    H = cam.floor_homography()
    roi = PARAMS["roi"]
    rng = np.random.RandomState(7)
    fg = FloorGrid(grid, min_count=PARAMS["reference"]["min_count"])
    poses = [(0.5 + 0.03 * i, 1.5 + 0.01 * math.sin(i), 0.0) for i in range(40)]
    for p in poses:
        est = (p[0] + rng.normal(0, 0.003), p[1] + rng.normal(0, 0.003), p[2])
        fg.add_warp(warp_to_grid(render(cam, FloorTexture(tex, 0, 0, res_t), p), H, est, grid, roi))

    dm = DirtMap(grid, CFG)
    for fid, p in enumerate(poses):
        est = (p[0] + rng.normal(0, 0.004), p[1] + rng.normal(0, 0.004), p[2])
        w = warp_to_grid(render(cam, FloorTexture(dirty, 0, 0, res_t), p), H, est, grid, roi)
        dm.update_compare(w, compare_warp(w, fg, PARAMS["compare"]), fid)

    regions, _ = dm.extract_regions()
    assert len(regions) == 1
    r = regions[0]
    assert math.hypot(r.center[0] - paper[0], r.center[1] - paper[1]) <= 0.02
    assert 0.04 <= r.size[0] <= 0.09 and 0.04 <= r.size[1] <= 0.09
    assert 0 <= r.best_frame < len(poses)
    st = dm.states()
    pr, pc = map_to_cell(grid, [paper])[0]
    assert st[pr, pc] == SUSPECT
    assert (st == CLEAN).sum() > 20 * (st == SUSPECT).sum()
