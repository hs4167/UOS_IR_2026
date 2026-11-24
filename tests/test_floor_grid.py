import math

import cv2
import numpy as np
import pytest

from floor_dirt.config import load_params
from floor_dirt.floor_grid import FloorGrid, lidar_free_mask, to_lab
from floor_dirt.projection import GridMeta, cell_to_map, map_to_cell, warp_to_grid
from floor_dirt.synthetic import FloorTexture, render, wide_camera

GRID = GridMeta(resolution=0.01, origin_x=0.0, origin_y=0.0, width=60, height=40)
ROI = load_params()["roi"]


def test_welford_matches_numpy_with_partial_masks():
    rng = np.random.RandomState(1)
    fg = FloorGrid(GRID)
    samples = [[] for _ in range(GRID.width * GRID.height)]
    for _ in range(30):
        h, w = 10, 15
        r0, c0 = rng.randint(0, GRID.height - h), rng.randint(0, GRID.width - w)
        lab = rng.uniform(0, 255, (h, w, 3)).astype(np.float32)
        mask = rng.rand(h, w) > 0.3
        fg.add_patch(lab, mask, r0, c0)
        for (r, c) in zip(*np.nonzero(mask)):
            samples[(r0 + r) * GRID.width + (c0 + c)].append(lab[r, c])
    ref = fg.reference()
    for idx in rng.choice(len(samples), 200, replace=False):
        r, c = divmod(idx, GRID.width)
        s = np.array(samples[idx])
        assert ref.count[r, c] == len(s)
        if len(s):
            np.testing.assert_allclose(ref.mean[r, c], s.mean(0), atol=1e-3)
            np.testing.assert_allclose(ref.var[r, c], s.var(0), rtol=1e-3, atol=1e-2)


def test_variance_stays_accurate_with_many_bright_samples():
    """밝은 값(≈200) 수천 번 + 작은 흔들림: 합·제곱합 float32라면 자릿수 손실이 나는 상황."""
    rng = np.random.RandomState(2)
    fg = FloorGrid(GridMeta(0.01, 0, 0, 1, 1))
    xs = 200.0 + rng.normal(0, 1.5, (5000, 3))
    for x in xs:
        fg.add_patch(x.reshape(1, 1, 3).astype(np.float32), np.ones((1, 1), bool), 0, 0)
    ref = fg.reference()
    np.testing.assert_allclose(ref.mean[0, 0], xs.mean(0), atol=1e-3)
    np.testing.assert_allclose(ref.var[0, 0], xs.var(0), rtol=0.01)


def test_min_count_and_window_slice():
    fg = FloorGrid(GRID, min_count=3)
    lab = np.full((5, 5, 3), 100, np.float32)
    for _ in range(2):
        fg.add_patch(lab, np.ones((5, 5), bool), 10, 20)
    fg.add_patch(lab[:2, :2], np.ones((2, 2), bool), 10, 20)
    with pytest.raises(ValueError):
        fg.add_patch(lab, np.ones((2, 2), bool), 10, 20)
    ref = fg.reference(slice(10, 15), slice(20, 25))
    assert ref.mean.shape == (5, 5, 3)
    assert ref.valid.sum() == 4 and ref.valid[:2, :2].all()
    np.testing.assert_allclose(ref.var, 0.0)


def test_patch_outside_grid_rejected():
    fg = FloorGrid(GRID)
    with pytest.raises(ValueError):
        fg.add_patch(np.zeros((5, 5, 3), np.float32), np.ones((5, 5), bool), 38, 0)


def test_merge_equals_single_accumulation():
    rng = np.random.RandomState(3)
    a, b, both = FloorGrid(GRID), FloorGrid(GRID), FloorGrid(GRID)
    a.runs, b.runs = ["run1"], ["run2"]
    for i in range(20):
        lab = rng.uniform(50, 200, (8, 8, 3)).astype(np.float32)
        mask = rng.rand(8, 8) > 0.2
        r0, c0 = rng.randint(0, 30), rng.randint(0, 50)
        (a if i % 2 else b).add_patch(lab, mask, r0, c0)
        both.add_patch(lab, mask, r0, c0)
    a.merge(b)
    np.testing.assert_array_equal(a.count, both.count)
    np.testing.assert_allclose(a.mean, both.mean, atol=1e-3)
    np.testing.assert_allclose(a.var(), both.var(), rtol=1e-3, atol=1e-2)
    assert a.runs == ["run1", "run2"]
    with pytest.raises(ValueError):
        a.merge(FloorGrid(GridMeta(0.01, 0, 0, 10, 10)))


def test_save_load_roundtrip(tmp_path):
    fg = FloorGrid(GRID, min_count=2)
    fg.runs = ["2026-11-19_ref"]
    fg.add_patch(np.full((4, 4, 3), 120, np.float32), np.ones((4, 4), bool), 1, 2)
    fg.add_patch(np.full((4, 4, 3), 130, np.float32), np.ones((4, 4), bool), 1, 2)
    fg.save(str(tmp_path / "ref"))
    back = FloorGrid.load(str(tmp_path / "ref"))
    assert back.grid == GRID and back.min_count == 2 and back.runs == ["2026-11-19_ref"]
    np.testing.assert_array_equal(back.count, fg.count)
    np.testing.assert_allclose(back.reference().var[1, 2], [25, 25, 25])
    assert FloorGrid.load(str(tmp_path / "ref"), min_count=5).min_count == 5


def test_lidar_free_mask_orientation(tmp_path):
    # 지도 4 x 3 칸(0.05 m). pgm 맨 위 왼쪽 칸만 빈 곳 = 지도에서 x 최소, y 최대
    pgm = np.full((3, 4), 205, np.uint8)
    pgm[0, 0] = 254
    cv2.imwrite(str(tmp_path / "m.pgm"), pgm)
    (tmp_path / "m.yaml").write_text("image: m.pgm\nresolution: 0.05\norigin: [0.0, 0.0, 0.0]\nnegate: 0\n")
    grid = GridMeta.from_lidar_map(str(tmp_path / "m.yaml"), 0.01)
    free = lidar_free_mask(str(tmp_path / "m.yaml"), grid)
    assert free.shape == (15, 20) and free.sum() == 25
    rows, cols = np.nonzero(free)
    center = cell_to_map(grid, [[rows.mean(), cols.mean()]])[0]
    np.testing.assert_allclose(center, [0.025, 0.125], atol=1e-9)


def test_coverage_report_and_image():
    fg = FloorGrid(GRID, min_count=1)
    fg.add_patch(np.full((10, 10, 3), 128, np.float32), np.ones((10, 10), bool), 0, 0)
    floor = np.zeros((GRID.height, GRID.width), bool)
    floor[:20, :20] = True
    cov = fg.coverage(floor)
    assert cov["floor_cells"] == 400 and cov["covered_cells"] == 100
    assert cov["covered_ratio"] == pytest.approx(0.25)
    assert cov["missing_area_m2"] == pytest.approx(0.03)
    img = fg.coverage_image(floor)
    assert img.shape == (GRID.height, GRID.width, 3)
    # 격자 row 0(= y 최소)은 그림 맨 아래
    assert tuple(img[-1, 15]) == (0, 0, 220)
    assert tuple(img[-1, 25]) == (40, 40, 40)


def test_reference_from_synthetic_drive():
    """합성 카메라로 바닥을 여러 자세에서 찍어 누적: 칸 평균이 실제 바닥 색과 같아야 한다."""
    grid = GridMeta(0.01, 0.0, 0.0, 300, 300)
    res = 0.002
    tex = np.zeros((1500, 1500, 3), np.uint8)
    tex[:, :750] = (60, 120, 180)    # x < 1.5 m: 주황빛 바닥
    tex[:, 750:] = (170, 160, 150)   # x ≥ 1.5 m: 회색 바닥
    texture = FloorTexture(tex, 0.0, 0.0, res)
    cam = wide_camera()
    H = cam.floor_homography()
    rng = np.random.RandomState(4)
    fg = FloorGrid(grid, min_count=3)
    for i in range(40):
        pose = (0.6 + 0.04 * i, 1.5 + 0.02 * math.sin(i), math.radians(rng.uniform(-5, 5)))
        img = render(cam, texture, pose)
        noisy = np.clip(img.astype(np.int16) + rng.randint(-3, 4, img.shape), 0, 255).astype(np.uint8)
        # 위치 추정 오차(수 mm)가 있는 자세로 투영 → 색 경계 칸이 매번 조금씩 다른 색을 받는다
        est = (pose[0] + rng.normal(0, 0.004), pose[1] + rng.normal(0, 0.004), pose[2])
        fg.add_warp(warp_to_grid(noisy, H, est, grid, ROI))

    ref = fg.reference()
    expect_a = to_lab(np.array([[[60, 120, 180]]], np.uint8))[0, 0]
    expect_b = to_lab(np.array([[[170, 160, 150]]], np.uint8))[0, 0]
    ra, ca = map_to_cell(grid, [[1.2, 1.5]])[0]
    rb, cb = map_to_cell(grid, [[1.8, 1.5]])[0]
    assert ref.valid[ra, ca] and ref.valid[rb, cb]
    np.testing.assert_allclose(ref.mean[ra, ca], expect_a, atol=1.5)
    np.testing.assert_allclose(ref.mean[rb, cb], expect_b, atol=1.5)
    assert ref.var[ra, ca].max() < 10  # 노이즈 수준
    # 색 경계(x = 1.5 m) 칸은 칸이 흔들려 섞이므로 분산이 크다 → 이후 임계값이 자동으로 높아짐
    re, ce = map_to_cell(grid, [[1.5, 1.5]])[0]
    assert ref.var[re, ce - 1:ce + 1].max() > 10 * ref.var[ra, ca].max()
