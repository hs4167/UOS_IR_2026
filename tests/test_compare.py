import numpy as np
import pytest

from floor_dirt.compare import compare_warp, compare_window, reference_window
from floor_dirt.config import load_params
from floor_dirt.floor_grid import FloorGrid, Reference, to_lab
from floor_dirt.projection import GridMeta, map_to_cell, warp_to_grid, window_slices
from floor_dirt.synthetic import FloorTexture, render, wide_camera

PARAMS = load_params()
SS = PARAMS["grid"]["supersample"]
CFG = PARAMS["compare"]
ROI = PARAMS["roi"]
SEAM = (110, 115, 120)
RED = (50, 50, 215)  # 이음새와 밝기(L)가 거의 같은 빨간 종이


def _ref_from(lab, var=4.0, margin=1):
    """관측 크기 lab에 대해 여백까지 포함한 기준 (가장자리는 복제)."""
    m = np.pad(lab, ((margin, margin), (margin, margin), (0, 0)), mode="edge").astype(np.float32)
    h, w = m.shape[:2]
    return Reference(mean=m, var=np.full((h, w, 3), var, np.float32), count=np.full((h, w), 5, np.float32),
                     valid=np.ones((h, w), bool))


def _floor_bgr(h=40, w=50, seed=0):
    rng = np.random.RandomState(seed)
    bgr = np.full((h, w, 3), (150, 160, 170), np.uint8)
    bgr[::10, :] = SEAM  # 타일 이음새
    bgr[:, ::10] = SEAM
    return np.clip(bgr.astype(int) + rng.randint(-2, 3, bgr.shape), 0, 255).astype(np.uint8)


def _brighter(bgr, gain=1.15):
    return np.clip(bgr.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def _cmp(obs_bgr, ref_bgr, method, r=1, offset=False, mask=None):
    obs = to_lab(obs_bgr)
    ref = _ref_from(to_lab(ref_bgr), margin=r)
    mask = np.ones(obs.shape[:2], bool) if mask is None else mask
    return compare_window(obs, mask, ref, method=method, neighborhood=r, k=CFG["k"], min_thr=CFG["min_thr"],
                          offset_correction=offset)


ALL = [("sad_gray", False), ("sad_gray", True), ("delta_e", False), ("delta_e", True), ("chroma_diff", False)]


# ---------------------------------------------------------------- 기본 동작

@pytest.mark.parametrize("method,offset", ALL)
def test_same_floor_no_hits(method, offset):
    floor = _floor_bgr()
    noisy = np.clip(floor.astype(int) + np.random.RandomState(9).randint(-2, 3, floor.shape), 0, 255).astype(np.uint8)
    res = _cmp(noisy, floor, method, offset=offset)
    assert res.observed.all() and not res.hit.any()


@pytest.mark.parametrize("method,offset", [m for m in ALL if m[0] != "chroma_diff"])
def test_dark_stain_detected_without_halo(method, offset):
    floor = _floor_bgr()
    obs = floor.copy()
    obs[15:22, 20:28] = (60, 60, 60)  # 회색 얼룩 7 x 8 칸
    hit = _cmp(obs, floor, method, offset=offset).hit
    assert hit[16:21, 21:27].all()
    assert hit.sum() <= 9 * 10        # 얼룩 + 테두리 1칸 이내 (둘레 오탐 없음)


def test_chroma_misses_gray_stain():
    floor = _floor_bgr()
    obs = floor.copy()
    obs[15:22, 20:28] = (60, 60, 60)
    assert not _cmp(obs, floor, "chroma_diff").hit.any()  # 색도 비교의 한계


def test_paper_with_seam_brightness_needs_color():
    """이음새와 밝기가 같은 빨간 종이가 이음새 위에 놓이면 밝기만 보는 sad_gray는 놓친다."""
    floor = _floor_bgr()
    assert abs(to_lab(np.array([[RED]], np.uint8))[0, 0, 0] - to_lab(np.array([[SEAM]], np.uint8))[0, 0, 0]) < 3
    obs = floor.copy()
    obs[17:24, 17:24] = RED  # (20, 20) 이음새 교차점을 덮음
    core = (slice(18, 23), slice(18, 23))
    assert not _cmp(obs, floor, "sad_gray").hit[core].all()
    assert _cmp(obs, floor, "delta_e").hit[core].all()
    assert _cmp(obs, floor, "chroma_diff").hit[core].all()


# ---------------------------------------------------------------- 어긋남 흡수

def test_neighborhood_absorbs_one_cell_shift():
    floor = _floor_bgr(seed=1)
    shifted = np.roll(floor, shift=(1, -1), axis=(0, 1))  # 위치 추정 오차로 1칸 어긋남
    inner = (slice(2, -2), slice(2, -2))
    assert _cmp(shifted, floor, "delta_e", r=0).hit[inner].sum() > 50  # 이음새가 오염으로 보임
    assert not _cmp(shifted, floor, "delta_e", r=1).hit[inner].any()


# ---------------------------------------------------------------- 조명 변화

def test_brightness_change_needs_offset_correction():
    floor = _floor_bgr(seed=2)
    obs = _brighter(floor)  # 형광등 하나 더 켬
    n = floor.shape[0] * floor.shape[1]
    assert _cmp(obs, floor, "sad_gray").hit.sum() > 0.8 * n
    assert _cmp(obs, floor, "delta_e").hit.sum() > 0.8 * n
    for method in ("sad_gray", "delta_e"):
        res = _cmp(obs, floor, method, offset=True)
        assert res.hit.sum() < 0.02 * n, method
        assert res.offset > 5.0  # 조명 변화량이 기록됨
    assert _cmp(obs, floor, "chroma_diff").hit.sum() < 0.02 * n


def test_offset_correction_still_finds_stain_under_new_lighting():
    floor = _floor_bgr(seed=3)
    obs = _brighter(floor)
    obs[10:16, 30:36] = (50, 50, 50)
    hit = _cmp(obs, floor, "delta_e", offset=True).hit
    assert hit[11:15, 31:35].all()
    assert hit.sum() <= 8 * 8


def test_offset_ignores_large_but_minor_stain_share():
    """창의 30%를 덮는 큰 얼룩이 있어도 조명 보정값(중앙값)은 거의 그대로."""
    floor = np.full((40, 50, 3), (150, 160, 170), np.uint8)
    obs = floor.copy()
    obs[:, :15] = (40, 40, 40)
    res = _cmp(obs, floor, "delta_e", offset=True)
    assert abs(res.offset) < 1.0
    assert res.hit[:, :14].all() and not res.hit[:, 16:].any()


# ---------------------------------------------------------------- 임계값·유효 칸

def test_threshold_uses_reference_variance():
    lab = np.full((5, 5, 3), 120, np.float32)
    ref = _ref_from(lab, var=0.0, margin=0)
    ref.var[2, 2] = (100.0, 0.0, 0.0)   # 반사가 심한 칸
    obs = lab.copy()
    obs[2, 2, 0] += 20.0                # 일반 칸이면 hit, 반사 칸이면 기준(3*10=30) 미만
    obs[0, 0, 0] += 20.0
    res = compare_window(obs, np.ones((5, 5), bool), ref, method="delta_e", neighborhood=0, k=3.0, min_thr=8.0)
    assert res.thr[2, 2] == pytest.approx(30.0) and res.thr[0, 0] == pytest.approx(8.0)
    assert res.hit[0, 0] and not res.hit[2, 2]


def test_cells_without_reference_or_masked_are_not_observed():
    floor = _floor_bgr()
    obs = np.full_like(floor, 30)
    ref = _ref_from(to_lab(floor))
    ref.valid[:, :20] = False
    res = compare_window(to_lab(obs), np.ones(floor.shape[:2], bool), ref, neighborhood=1)
    assert not res.observed[:, :19].any() and not res.hit[:, :19].any()
    assert res.hit[:, 20:].all()
    mask = np.ones(floor.shape[:2], bool)
    mask[:, 40:] = False  # 제외 마스크 (의자 등)
    res = compare_window(to_lab(obs), mask, ref, neighborhood=1)
    assert not res.hit[:, 40:].any() and (res.diff[:, 40:] == 0).all()


def test_unknown_method_rejected():
    lab = np.zeros((3, 3, 3), np.float32)
    with pytest.raises(ValueError):
        compare_window(lab, np.ones((3, 3), bool), _ref_from(lab), method="foo")


def test_reference_window_pads_outside_grid():
    fg = FloorGrid(GridMeta(0.01, 0, 0, 20, 10), min_count=1)
    fg.add_patch(np.full((10, 20, 3), 77, np.float32), np.ones((10, 20), bool), 0, 0)
    ref = reference_window(fg, 0, 15, 10, 5, 2)
    assert ref.mean.shape == (14, 9, 3)
    assert not ref.valid[:2].any() and not ref.valid[:, -2:].any()
    assert ref.valid[2:12, 2:7].all() and (ref.mean[2:12, 2:7] == 77).all()
    inner = reference_window(fg, 3, 3, 4, 4, 1)
    assert inner.mean.shape == (6, 6, 3) and inner.valid.all()


# ---------------------------------------------------------------- 합성 주행

def test_synthetic_patrol_finds_paper_ignores_tiles():
    """기준 주행(깨끗) → 정찰(빨간 종이 1장, 조명 15% 밝아짐, 위치 오차): 종이 칸만 달라짐 (기본 설정)."""
    grid = GridMeta(0.01, 0.0, 0.0, 300, 300)
    res_t = 0.002
    tex = np.full((1500, 1500, 3), (150, 160, 170), np.uint8)
    tex[::50, :] = SEAM  # 10 cm 타일 이음새
    tex[:, ::50] = SEAM
    clean = FloorTexture(tex, 0.0, 0.0, res_t)
    dirty_img = _brighter(tex)
    paper_xy = (1.40, 1.52)  # 이음새 교차점 근처
    c = (int(paper_xy[0] / res_t), int(paper_xy[1] / res_t))
    dirty_img[c[1] - 15:c[1] + 15, c[0] - 15:c[0] + 15] = RED  # 6 cm
    dirty = FloorTexture(dirty_img, 0.0, 0.0, res_t)

    cam = wide_camera()
    H = cam.floor_homography()
    rng = np.random.RandomState(5)
    fg = FloorGrid(grid, min_count=3)
    poses = [(0.6 + 0.03 * i, 1.5, 0.0) for i in range(30)]
    for p in poses:
        est = (p[0] + rng.normal(0, 0.003), p[1] + rng.normal(0, 0.003), p[2])
        fg.add_warp(warp_to_grid(render(cam, clean, p), H, est, grid, ROI, supersample=SS))

    hits = np.zeros((grid.height, grid.width), np.int32)
    seen = np.zeros_like(hits)
    for p in poses[5:25]:
        est = (p[0] + rng.normal(0, 0.004), p[1] + rng.normal(0, 0.004), p[2])
        w = warp_to_grid(render(cam, dirty, p), H, est, grid, ROI, supersample=SS)
        cr = compare_warp(w, fg, CFG)
        rs, cs = window_slices(w)
        hits[rs, cs] += cr.hit
        seen[rs, cs] += cr.observed

    frac = np.zeros(hits.shape)
    np.divide(hits, seen, out=frac, where=seen > 0)
    pr, pc = map_to_cell(grid, [paper_xy])[0]
    core = (slice(pr - 2, pr + 2), slice(pc - 2, pc + 2))           # 종이 6 x 6칸 중 안쪽 4 x 4
    assert seen[core].min() >= 10
    assert frac[core].min() > 0.8                                    # 종이 칸은 거의 매번 hit
    far = np.ones_like(frac, bool)
    far[pr - 6:pr + 7, pc - 6:pc + 7] = False
    assert (frac[far & (seen >= 3)] > 0.5).sum() == 0                # 타일 이음새·조명 변화는 오탐 없음
