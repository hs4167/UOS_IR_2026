import math

import numpy as np
import pytest

from floor_dirt.calibration import (
    Intrinsics, calibrate_intrinsics, corner_coverage, detect_floor_board, find_chessboard, fit_floor_homography,
    floor_board_world, in_roi, load_floor_calibration, load_intrinsics, order_floor_corners,
    projection_error, save_floor_calibration, save_intrinsics,
)
from floor_dirt.config import load_params
from floor_dirt.projection import apply_homography, pixel_to_robot, roi_polygon
from synthetic import board_view, c920_like, distort_image, floor_board_texture, render

PATTERN = (9, 6)
SQUARE = 0.025
ROI = load_params()["roi"]
D_TRUE = np.array([0.10, -0.20, 0.0005, -0.0008, 0.05])


def _views(cam, D, n=20, seed=0):
    """실제 촬영 요령과 같게 보드를 화면 구석구석에 두고 기울여 찍는다 (구석을 안 찍으면 왜곡이 안 잡힘)."""
    rng = np.random.RandomState(seed)
    imgs = []
    board_c = np.array([(PATTERN[0] - 1) * SQUARE / 2, (PATTERN[1] - 1) * SQUARE / 2])
    while len(imgs) < n:
        rvec = np.radians(rng.uniform([-35, -35, -20], [35, 35, 20]))
        z = rng.uniform(0.35, 0.55)
        center = np.array([rng.uniform(-0.55, 0.55) * z, rng.uniform(-0.4, 0.4) * z])
        tvec = np.array([center[0] - board_c[0], center[1] - board_c[1], z])
        img = board_view(cam.K, D, (cam.width, cam.height), rvec, tvec, PATTERN, SQUARE)
        if find_chessboard(img, PATTERN) is not None:
            imgs.append(img)
    return imgs


@pytest.fixture(scope="module")
def intrinsics_result():
    cam = c920_like()
    return cam, calibrate_intrinsics(_views(cam, D_TRUE), PATTERN, SQUARE)


# ---------------------------------------------------------------- 내부 보정

def test_intrinsics_recovers_camera(intrinsics_result):
    cam, (intr, used) = intrinsics_result
    assert len(used) == 20
    assert intr.rms < 0.5
    assert (intr.width, intr.height) == (cam.width, cam.height)
    np.testing.assert_allclose(intr.K[[0, 1], [0, 1]], cam.K[[0, 1], [0, 1]], rtol=0.02)
    np.testing.assert_allclose(intr.K[:2, 2], cam.K[:2, 2], atol=5)
    # 계수 하나하나보다 실제 왜곡 효과가 맞는지 본다: 화면 구석 점의 보정 결과 비교
    from floor_dirt.projection import undistort_points
    u, v = np.meshgrid(np.linspace(30, 610, 9), np.linspace(30, 450, 7))
    pts = np.column_stack([u.ravel(), v.ravel()])
    np.testing.assert_allclose(undistort_points(intr.K, intr.D, pts),
                               undistort_points(cam.K, D_TRUE, pts), atol=1.0)


def test_corner_coverage_flags_empty_corner():
    center_only = [np.array([[300.0, 220.0], [340.0, 260.0]])]
    cov = corner_coverage(center_only, (640, 480))
    assert cov[1, 1] == 2 and cov.sum() == 2 and (cov == 0).sum() == 8
    full = [np.array([[10.0, 10.0], [630.0, 470.0], [320.0, 240.0]])]
    cov = corner_coverage(full, (640, 480))
    assert cov[0, 0] == 1 and cov[2, 2] == 1 and cov[1, 1] == 1


def test_intrinsics_requires_enough_images():
    blank = [np.full((480, 640), 128, np.uint8)] * 5
    with pytest.raises(ValueError):
        calibrate_intrinsics(blank, PATTERN, SQUARE)


def test_intrinsics_yaml_roundtrip(tmp_path):
    intr = Intrinsics(K=np.array([[500.0, 0, 320], [0, 501.0, 240], [0, 0, 1]]),
                      D=np.array([0.1, -0.2, 0.0, 0.0, 0.05]), width=640, height=480, rms=0.21)
    path = str(tmp_path / "cfg" / "camera_intrinsics.yaml")
    save_intrinsics(path, intr, note="test")
    back = load_intrinsics(path)
    np.testing.assert_allclose(back.K, intr.K)
    np.testing.assert_allclose(back.D, intr.D)
    assert (back.width, back.height, back.rms) == (640, 480, 0.21)


# ---------------------------------------------------------------- 바닥 보드 정렬

def _grid_like(pattern, flip_r=False, flip_c=False):
    """이미지에서 [0,0]이 오른쪽 아래, 가로가 왼쪽, 세로가 위쪽인 가짜 코너 격자."""
    cols, rows = pattern
    r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    g = np.stack([500 - 30.0 * c - 3 * r, 400 - 25.0 * r + 2 * c], axis=-1)
    true = g.copy()
    if flip_r:
        g = g[::-1]
    if flip_c:
        g = g[:, ::-1]
    return g.reshape(-1, 2), true


@pytest.mark.parametrize("flip_r,flip_c", [(False, False), (True, True), (True, False), (False, True)])
def test_order_floor_corners_any_detection_order(flip_r, flip_c):
    corners, true = _grid_like(PATTERN, flip_r, flip_c)
    np.testing.assert_array_equal(order_floor_corners(corners, PATTERN), true)


def test_order_floor_corners_rejects_rotated_board():
    corners, _ = _grid_like(PATTERN)
    rotated = corners[:, ::-1]  # 가로·세로 방향이 바뀐 보드
    with pytest.raises(ValueError):
        order_floor_corners(rotated, PATTERN)


def test_floor_board_world_layout():
    w = floor_board_world((3, 2), 0.1, (0.3, -0.1), yaw=0.0)
    np.testing.assert_allclose(w[0, 0], [0.3, -0.1])
    np.testing.assert_allclose(w[0, 2], [0.3, 0.1])     # 가로 = +y
    np.testing.assert_allclose(w[1, 0], [0.4, -0.1])    # 세로 = +x
    w = floor_board_world((3, 2), 0.1, (0.0, 0.0), yaw=math.pi / 2)
    np.testing.assert_allclose(w[1, 0], [0.0, 0.1], atol=1e-12)


# ---------------------------------------------------------------- 바닥 호모그래피 (합성 이미지)

FLOOR_PATTERN = (7, 5)
FLOOR_SQUARE = 0.03


def _floor_shot(cam, D, origin, yaw):
    tex = floor_board_texture(FLOOR_PATTERN, FLOOR_SQUARE, origin, yaw)
    ideal = render(cam, tex, (0.0, 0.0, 0.0))
    return distort_image(ideal, cam.K, D) if D is not None else ideal


@pytest.mark.parametrize("yaw_deg", [0.0, 8.0, -12.0])
def test_floor_homography_matches_true_camera(yaw_deg):
    cam = c920_like()
    intr = Intrinsics(cam.K, D_TRUE, cam.width, cam.height, 0.0)
    origin = (0.25, -0.09)
    img = _floor_shot(cam, D_TRUE, origin, math.radians(yaw_deg))
    uv, xy = detect_floor_board(img, intr, FLOOR_PATTERN, FLOOR_SQUARE, origin, math.radians(yaw_deg))
    calib = fit_floor_homography(uv, xy, cam.width, cam.height)
    assert calib.fit_mean < 0.001

    # 정답 H와 ROI 안 바닥점(보드 바깥 포함)에서 비교
    H_true = cam.floor_homography()
    xs, ys = np.meshgrid(np.linspace(0.2, 0.55, 8), np.linspace(-0.12, 0.12, 7))
    xy_chk = np.column_stack([xs.ravel(), ys.ravel()])
    uv_chk = apply_homography(np.linalg.inv(H_true), xy_chk)
    err = np.linalg.norm(pixel_to_robot(calib.H, uv_chk) - xy_chk, axis=1)
    assert err.mean() < 0.003 and err.max() < 0.008


def test_wrong_origin_measurement_is_caught_by_check():
    """보정 때 기준 코너를 2 cm 잘못 재면, 다른 위치의 확인 보드로 검사했을 때 실패해야 한다."""
    cam = c920_like()
    intr = Intrinsics(cam.K, np.zeros(5), cam.width, cam.height, 0.0)
    true_origin = (0.25, -0.09)
    img = _floor_shot(cam, None, true_origin, 0.0)
    uv, xy = detect_floor_board(img, intr, FLOOR_PATTERN, FLOOR_SQUARE, (0.27, -0.09))
    calib = fit_floor_homography(uv, xy, cam.width, cam.height)

    chk_origin = (0.35, -0.05)
    img2 = _floor_shot(cam, None, chk_origin, math.radians(5))
    uv2, xy2 = detect_floor_board(img2, intr, FLOOR_PATTERN, FLOOR_SQUARE, chk_origin, math.radians(5))
    stats = projection_error(calib.H, uv2, xy2, pass_mean=0.01, min_points=5)
    assert not stats.passed and stats.mean > 0.015

    # 제대로 잰 경우는 통과
    uv, xy = detect_floor_board(img, intr, FLOOR_PATTERN, FLOOR_SQUARE, true_origin)
    good = fit_floor_homography(uv, xy, cam.width, cam.height)
    stats = projection_error(good.H, uv2, xy2)
    assert stats.passed and stats.mean < 0.003


def test_projection_error_needs_enough_points():
    H = np.eye(3)
    pts = np.array([[0.3, 0.0], [0.4, 0.1]])
    s = projection_error(H, pts, pts, min_points=5)
    assert s.mean == 0.0 and not s.passed
    assert not projection_error(H, np.zeros((0, 2)), np.zeros((0, 2))).passed


def test_floor_calibration_yaml_roundtrip(tmp_path):
    cam = c920_like()
    calib = fit_floor_homography(apply_homography(np.linalg.inv(cam.floor_homography()), roi_polygon(ROI)),
                                 roi_polygon(ROI), cam.width, cam.height)
    path = str(tmp_path / "floor_homography.yaml")
    save_floor_calibration(path, calib, meta={"pattern": [7, 5], "square": 0.03})
    back = load_floor_calibration(path)
    np.testing.assert_allclose(back.H, calib.H)
    assert (back.width, back.height) == (640, 480)


def test_in_roi():
    pts = np.array([[0.3, 0.0], [0.1, 0.0], [0.5, 0.3]])
    assert in_roi(pts, ROI).tolist() == [True, False, False]
