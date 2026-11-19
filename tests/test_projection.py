import math

import cv2
import numpy as np
import pytest

from floor_dirt.config import load_params
from floor_dirt.projection import (
    GridMeta, cell_to_map, map_to_cell, map_to_robot, pixel_to_robot, robot_to_map,
    robot_to_pixel, undistort_maps, undistort, undistort_points, warp_mask, warp_to_grid,
    window_slices, grid_matrix, apply_homography,
)
from synthetic import FloorTexture, c920_like, render, wide_camera

ROI = load_params()["roi"]
GRID = GridMeta(resolution=0.01, origin_x=0.0, origin_y=0.0, width=400, height=400)


# ---------------------------------------------------------------- 점 변환

def test_pixel_robot_roundtrip_and_matches_camera_model():
    cam = c920_like()
    H = cam.floor_homography()
    xy = np.array([[0.30, 0.0], [0.45, 0.10], [0.25, -0.12]])
    uv, in_front = cam.project_robot_points(xy)
    assert in_front.all()
    np.testing.assert_allclose(robot_to_pixel(H, xy), uv, atol=1e-6)
    np.testing.assert_allclose(pixel_to_robot(H, uv), xy, atol=1e-9)


def test_axis_convention_forward_and_left():
    cam = c920_like()
    H = cam.floor_homography()
    bottom, top = pixel_to_robot(H, [[320, 470], [320, 250]])
    assert top[0] > bottom[0] > 0          # 이미지 위쪽 = 로봇에서 더 먼 앞쪽
    assert abs(bottom[1]) < 1e-9           # 가운데 열 = y 0
    left, right = pixel_to_robot(H, [[100, 400], [540, 400]])
    assert left[1] > 0 > right[1]          # 이미지 왼쪽 = 로봇 왼쪽(+y)


def test_robot_map_transform():
    pose = (1.0, 2.0, math.pi / 2)
    np.testing.assert_allclose(robot_to_map(pose, [[1.0, 0.0]]), [[1.0, 3.0]], atol=1e-12)
    np.testing.assert_allclose(robot_to_map(pose, [[0.0, 1.0]]), [[0.0, 2.0]], atol=1e-12)
    pts = np.array([[0.3, -0.2], [5.0, 1.0]])
    np.testing.assert_allclose(map_to_robot(pose, robot_to_map(pose, pts)), pts, atol=1e-12)


def test_grid_cell_convention():
    g = GridMeta(resolution=0.01, origin_x=-1.0, origin_y=-2.0, width=300, height=500)
    # 칸 (row 0, col 0) 중심 = 원점 + 반 칸
    np.testing.assert_allclose(cell_to_map(g, [[0, 0]]), [[-0.995, -1.995]])
    # 칸 중심은 연속 인덱스로 정수
    np.testing.assert_allclose(apply_homography(grid_matrix(g), [[-0.995, -1.995]]), [[0.0, 0.0]], atol=1e-9)
    # 칸 내부 어디든 같은 칸으로
    assert map_to_cell(g, [[0.0049, 0.0001]]).tolist() == [[200, 100]]
    assert map_to_cell(g, [[0.0051, 0.0099]]).tolist() == [[200, 100]]
    rc = np.array([[12, 34], [499, 299]])
    np.testing.assert_array_equal(map_to_cell(g, cell_to_map(g, rc)), rc)


def test_grid_from_lidar_map(tmp_path):
    cv2.imwrite(str(tmp_path / "classroom.pgm"), np.full((160, 200), 254, np.uint8))
    (tmp_path / "classroom.yaml").write_text(
        "image: classroom.pgm\nresolution: 0.05\norigin: [-3.0, -4.5, 0.0]\n"
        "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n", encoding="utf-8")
    g = GridMeta.from_lidar_map(str(tmp_path / "classroom.yaml"), 0.01)
    assert g == GridMeta(0.01, -3.0, -4.5, 1000, 800)

    (tmp_path / "rot.yaml").write_text("image: classroom.pgm\nresolution: 0.05\norigin: [0, 0, 0.3]\n")
    with pytest.raises(ValueError):
        GridMeta.from_lidar_map(str(tmp_path / "rot.yaml"), 0.01)


# ---------------------------------------------------------------- 왜곡 보정

def test_undistort_points_and_image():
    cam = c920_like()
    K = cam.K
    D = np.array([0.12, -0.25, 0.001, -0.002, 0.1])
    ideal = np.array([[100.0, 80.0], [320.0, 240.0], [600.0, 450.0]])
    # 이상적인 픽셀 → 렌즈 왜곡 적용 (OpenCV 왜곡 모델)
    norm = np.column_stack([(ideal - K[:2, 2]) / [K[0, 0], K[1, 1]], np.ones(3)])
    distorted, _ = cv2.projectPoints(norm, np.zeros(3), np.zeros(3), K, D)
    np.testing.assert_allclose(undistort_points(K, D, distorted.reshape(-1, 2)), ideal, atol=0.05)

    # 왜곡이 0이면 이미지 보정은 항등
    img = (np.random.RandomState(0).rand(cam.height, cam.width) * 255).astype(np.uint8)
    out = undistort(img, undistort_maps(K, np.zeros(5), (cam.width, cam.height)))
    assert np.abs(out.astype(int) - img.astype(int))[2:-2, 2:-2].max() <= 1


# ---------------------------------------------------------------- 격자로 옮기기

def _texture_with_square(center_xy, size=0.04):
    res = 0.002
    tex = np.full((2000, 2000), 100, np.uint8)  # 4 m x 4 m, 원점 (0, 0)
    c = (np.asarray(center_xy) / res).astype(int)
    h = int(size / res / 2)
    tex[c[1] - h:c[1] + h, c[0] - h:c[0] + h] = 230
    return FloorTexture(tex, 0.0, 0.0, res)


@pytest.mark.parametrize("pose", [(1.0, 2.0, 0.0), (2.0, 1.5, math.radians(30)), (2.5, 2.5, math.radians(-135))])
@pytest.mark.parametrize("camera", [c920_like, wide_camera])
def test_warp_places_floor_mark_within_one_cell(pose, camera):
    cam = camera()
    target_robot = np.array([[0.40, 0.05]])
    target_map = robot_to_map(pose, target_robot)[0]
    img = render(cam, _texture_with_square(target_map), pose)

    res = warp_to_grid(img, cam.floor_homography(), pose, GRID, ROI)
    assert res is not None
    mark = (res.patch > 180) & res.valid
    assert mark.sum() > 8
    rows, cols = np.nonzero(mark)
    found = cell_to_map(GRID, [[rows.mean() + res.r0, cols.mean() + res.c0]])[0]
    assert np.linalg.norm(found - target_map) <= 0.01  # 1칸 이내


def test_valid_mask_covers_roi_when_fully_visible():
    cam = wide_camera()
    H = cam.floor_homography()
    corners = np.array([[ROI["x_min"], ROI["y_min"]], [ROI["x_max"], ROI["y_max"]],
                        [ROI["x_min"], ROI["y_max"]], [ROI["x_max"], ROI["y_min"]]])
    uv, front = cam.project_robot_points(corners)
    assert front.all() and (uv >= 1).all() and (uv[:, 0] <= cam.width - 2).all() and (uv[:, 1] <= cam.height - 2).all()

    img = np.full((cam.height, cam.width), 50, np.uint8)
    # 축 정렬 자세: ROI 경계가 칸 경계와 일치 → 정확히 45 x 50 칸
    res = warp_to_grid(img, H, (1.0, 2.0, 0.0), GRID, ROI)
    assert res.valid.sum() == 45 * 50
    assert (res.patch[res.valid] == 50).all()

    # 회전된 자세: 넓이는 거의 같아야 함
    res = warp_to_grid(img, H, (2.0, 2.0, math.radians(37)), GRID, ROI)
    assert abs(res.valid.sum() - 2250) / 2250 < 0.03


def test_valid_mask_matches_independent_projection():
    """C920처럼 ROI 일부만 보이는 경우: 유효 칸 = ROI 안 + 실제로 화면에 찍히는 칸."""
    cam = c920_like()
    pose = (2.0, 2.0, math.radians(20))
    img = np.full((cam.height, cam.width), 50, np.uint8)
    res = warp_to_grid(img, cam.floor_homography(), pose, GRID, ROI)

    h, w = res.valid.shape
    rr, cc = np.meshgrid(np.arange(h) + res.r0, np.arange(w) + res.c0, indexing="ij")
    robot_xy = map_to_robot(pose, cell_to_map(GRID, np.column_stack([rr.ravel(), cc.ravel()])))
    uv, front = cam.project_robot_points(robot_xy)
    expected = (front
                & (robot_xy[:, 0] >= ROI["x_min"]) & (robot_xy[:, 0] <= ROI["x_max"])
                & (robot_xy[:, 1] >= ROI["y_min"]) & (robot_xy[:, 1] <= ROI["y_max"])
                & (uv[:, 0] >= 1) & (uv[:, 0] <= cam.width - 2)
                & (uv[:, 1] >= 1) & (uv[:, 1] <= cam.height - 2)).reshape(h, w)
    assert 0 < expected.sum() < 2250          # 일부만 보이는 상황이 맞는지
    assert (res.valid != expected).sum() <= 0.005 * expected.sum()


def test_points_behind_camera_are_invalid():
    """지평선이 화면에 들어오면 카메라 뒤쪽 바닥점이 호모그래피상 화면 위쪽에 '거울상'으로 찍힌다."""
    cam = c920_like(pitch_deg=10.0)
    H = cam.floor_homography()
    behind = dict(x_min=-1.50, x_max=-0.50, y_min=-0.50, y_max=0.50)
    xs, ys = np.meshgrid(np.linspace(-1.5, -0.5, 11), np.linspace(-0.5, 0.5, 11))
    uv = robot_to_pixel(H, np.column_stack([xs.ravel(), ys.ravel()]))
    mirrored = (uv[:, 0] >= 1) & (uv[:, 0] <= cam.width - 2) & (uv[:, 1] >= 1) & (uv[:, 1] <= cam.height - 2)
    assert mirrored.any()  # 앞뒤 판정이 없으면 화면 안으로 잘못 들어오는 상황인지

    img = np.full((cam.height, cam.width), 50, np.uint8)
    res = warp_to_grid(img, H, (2.0, 2.0, 0.0), GRID, behind)
    assert res is not None and not res.valid.any()


def test_roi_outside_or_at_edge_of_grid():
    cam = wide_camera()
    img = np.full((cam.height, cam.width), 50, np.uint8)
    H = cam.floor_homography()
    assert warp_to_grid(img, H, (10.0, 10.0, 0.0), GRID, ROI) is None

    # 격자 오른쪽 끝(x = 4 m)에 걸친 경우: 창이 격자 안으로 잘린다
    res = warp_to_grid(img, H, (3.7, 2.0, 0.0), GRID, ROI)
    rs, cs = window_slices(res)
    assert cs.stop == GRID.width and rs.stop <= GRID.height
    assert res.valid.sum() == (400 - 385) * 50


def test_warp_mask_follows_same_mapping():
    cam = wide_camera()
    H = cam.floor_homography()
    pose = (1.0, 2.0, 0.0)
    img = np.zeros((cam.height, cam.width), np.uint8)
    mask = np.zeros_like(img, bool)
    # 로봇 왼쪽 절반(y > 0)에 해당하는 이미지 영역 = 이미지 왼쪽 절반
    mask[:, :cam.width // 2] = True
    res = warp_to_grid(img, H, pose, GRID, ROI)
    m = warp_mask(mask, res)
    h, w = m.shape
    rr, cc = np.meshgrid(np.arange(h) + res.r0, np.arange(w) + res.c0, indexing="ij")
    robot_y = map_to_robot(pose, cell_to_map(GRID, np.column_stack([rr.ravel(), cc.ravel()])))[:, 1].reshape(h, w)
    v = res.valid
    assert m[v & (robot_y > 0.01)].all()
    assert not m[v & (robot_y < -0.01)].any()
