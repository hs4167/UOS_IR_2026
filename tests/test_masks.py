import math
import sys
import types

import numpy as np
import pytest

from floor_dirt import masks
from floor_dirt.compare import compare_warp
from floor_dirt.config import load_params
from floor_dirt.floor_grid import FloorGrid
from floor_dirt.masks import (
    Detection, ImageExcluder, boxes_to_mask, class_map_to_mask, find_class_index, lidar_exclusion, polygon_mask,
    scan_to_points, window_keep_mask,
)
from floor_dirt.projection import (
    GridMeta, cell_to_map, map_to_cell, map_to_robot, robot_to_map, robot_to_pixel, warp_to_grid,
)
from floor_dirt.synthetic import FloorTexture, render, wide_camera

PARAMS = load_params()
SS = PARAMS["grid"]["supersample"]
ROI = PARAMS["roi"]
GRID = GridMeta(0.01, 0.0, 0.0, 300, 300)


# ---------------------------------------------------------------- 이미지 단계

def test_polygon_mask():
    m = polygon_mask((100, 200), [[0, 80], [199, 80], [199, 99], [0, 99]])  # 화면 아래 띠 (로봇 앞부분)
    assert m[85:, :].all() and not m[:79].any()
    assert not polygon_mask((10, 10), []).any()


def test_boxes_to_mask_filters_classes_and_clips():
    dets = [Detection("person", 0.9, 10.2, 20.0, 30.0, 60.7), Detection("tv", 0.9, 0, 0, 50, 50),
            Detection("Chair", 0.8, 90, 90, 130, 130)]
    m = boxes_to_mask((100, 100), dets, ["person", "chair"], margin_px=2)
    assert m[18:63, 8:32].all()
    assert not m[0:5, 40:50].any()       # tv는 제외 대상 아님
    assert m[88:, 88:].all()             # 화면 밖은 잘림, 대소문자 무시
    assert m.sum() == 45 * 24 + 12 * 12


def test_find_class_index():
    assert find_class_index(["wall", "floor", "cabinet"]) == 1
    assert find_class_index(["void", "Floor_Mat", "chair"]) == 1
    assert find_class_index(["floor", "floor_mat"]) == 0           # 정확히 같은 이름 우선
    with pytest.raises(ValueError):
        find_class_index(["wall", "chair"])
    with pytest.raises(ValueError):
        find_class_index(["floor_a", "floor_b"])


def test_class_map_to_mask_upscales():
    cm = np.array([[0, 2], [2, 1]], np.uint8).reshape(2, 2, 1)       # cudaToNumpy 모양 (h, w, 1)
    m = class_map_to_mask(cm, 2, (4, 6))
    assert m.shape == (4, 6)
    assert m[:2, 3:].all() and m[2:, :3].all() and not m[:2, :3].any()


# ---------------------------------------------------------------- LiDAR

def test_scan_to_points():
    ranges = [1.0, float("inf"), 0.5, 0.0, float("nan"), 10.0]
    pts = scan_to_points(ranges, angle_min=0.0, angle_increment=math.pi / 2, range_min=0.12, range_max=3.5)
    np.testing.assert_allclose(pts, [[1.0, 0.0], [-0.5, 0.0]], atol=1e-12)
    # LiDAR가 로봇 중심보다 6.4 cm 뒤에 달린 경우 (Waffle Pi)
    pts = scan_to_points([1.0], 0.0, 0.1, sensor_pose=(-0.064, 0.0, 0.0))
    np.testing.assert_allclose(pts, [[0.936, 0.0]], atol=1e-12)


def test_lidar_exclusion_disk_and_window_edge():
    pose = (1.0, 1.0, 0.0)
    # 의자 다리 하나: 로봇 앞 0.30 m
    ex = lidar_exclusion(np.array([[0.30, 0.0]]), pose, GRID, 90, 120, 30, 40, radius=0.05)
    rc = map_to_cell(GRID, robot_to_map(pose, [[0.30, 0.0]]))[0] - (90, 120)
    assert ex[rc[0], rc[1]]
    assert ex.sum() == np.count_nonzero(masks.disk_kernel(5.0))   # 반경 5칸 원
    rr, cc = np.nonzero(ex)
    d = np.hypot(rr - rc[0], cc - rc[1])
    assert d.max() <= 5.0 + 1e-9

    # 창 바로 바깥(3칸 밖)의 점도 반경이 창 안으로 들어온다
    pt_map = cell_to_map(GRID, [[100, 117]])  # 창 왼쪽 c0=120에서 3칸 밖
    ex = lidar_exclusion(map_to_robot(pose, pt_map), pose, GRID, 90, 120, 30, 40, radius=0.05)
    assert ex[10, 0] and ex[10, 1] and not ex[10, 3]
    assert not lidar_exclusion(np.zeros((0, 2)), pose, GRID, 0, 0, 5, 5, 0.05).any()


# ---------------------------------------------------------------- 가짜 jetson-inference

def _install_fake_jetson(monkeypatch, seg_names, dets):
    ji = types.ModuleType("jetson_inference")
    ju = types.ModuleType("jetson_utils")

    class Buf:
        def __init__(self, width, height, format):
            self.width, self.height, self.data = width, height, np.zeros((height, width, 1), np.uint8)

    class SegNet:
        def __init__(self, model):
            self.model = model

        def GetNumClasses(self):
            return len(seg_names)

        def GetClassDesc(self, i):
            return seg_names[i]

        def GetGridSize(self):
            return (8, 6)

        def Process(self, img):
            self.last = img

        def Mask(self, buf, w, h, filter_mode="linear"):
            assert filter_mode == "point"
            buf.data[:] = 0
            buf.data[3:, :] = seg_names.index("floor")  # 아래 절반 = 바닥

    class DetNet:
        def __init__(self, model, threshold=0.5):
            self.threshold = threshold

        def Detect(self, img, overlay="box"):
            assert overlay == "none"
            return [types.SimpleNamespace(ClassID=c, Confidence=0.9, Left=l, Top=t, Right=r, Bottom=b)
                    for c, l, t, r, b in dets]

        def GetClassDesc(self, i):
            return ["unlabeled", "person", "chair"][i]

    ji.segNet, ji.detectNet = SegNet, DetNet
    ju.cudaFromNumpy = lambda a: ("cuda", a.shape)
    ju.cudaAllocMapped = lambda width, height, format: Buf(width, height, format)
    ju.cudaToNumpy = lambda b: b.data
    ju.cudaDeviceSynchronize = lambda: None
    monkeypatch.setitem(sys.modules, "jetson_inference", ji)
    monkeypatch.setitem(sys.modules, "jetson_utils", ju)


def test_image_excluder_with_fake_jetson(monkeypatch):
    _install_fake_jetson(monkeypatch, ["wall", "floor", "chair"], dets=[(1, 100, 300, 140, 400), (2, 500, 250, 560, 470)])
    cfg = dict(PARAMS["exclusion"], body_polygon_px=[[0, 460], [639, 460], [639, 479], [0, 479]],
               detectnet_classes=["person"], detectnet_margin_px=0)
    exc = ImageExcluder(cfg)
    assert exc.segnet.floor_index == 1 and exc.segnet.floor_name == "floor"
    img = np.zeros((480, 640, 3), np.uint8)
    ex, dets = exc(img)
    assert [d.name for d in dets] == ["person", "chair"]
    assert ex[:240].all()                      # segNet: 위 절반은 바닥 아님
    assert ex[300:400, 100:140].all()          # person 박스
    assert not ex[300:400, 500:560].any()      # chair는 이 설정에서 제외 대상 아님 (아래쪽은 바닥)
    assert ex[460:].all()                      # 로봇 몸체
    assert not ex[250:450, 200:400].any()      # 나머지 바닥


def test_image_excluder_without_jetson_needs_flags_off():
    cfg = dict(PARAMS["exclusion"], use_segnet=False, use_detectnet=False)
    ex, dets = ImageExcluder(cfg)(np.zeros((10, 10, 3), np.uint8))
    assert not ex.any() and dets == []
    # jetson-inference가 없는 PC(WSL)에서만: segNet을 켜면 원인을 알려주는 ImportError
    # (젯슨에서는 실제로 모델을 올리므로 확인하지 않는다. sys.modules만 보면 지연 import라 판별이 안 됨)
    from floor_dirt.jetson import import_jetson
    try:
        import_jetson(need_inference=True)
    except ImportError:
        with pytest.raises(ImportError):
            ImageExcluder(dict(PARAMS["exclusion"], use_segnet=True))


# ---------------------------------------------------------------- 합성 정찰

def _textures():
    res = 0.002
    tex = np.full((1500, 1500, 3), (150, 160, 170), np.uint8)
    clean = FloorTexture(tex, 0.0, 0.0, res)
    dirty = tex.copy()
    # 바닥에 놓인 가방 (detectNet이 찾은 물체라고 가정): 바닥 접점 부분이 어둡게 보임
    bag_xy = (1.40, 1.50)
    c = (int(bag_xy[0] / res), int(bag_xy[1] / res))
    dirty[c[1] - 40:c[1] + 40, c[0] - 25:c[0] + 25] = (30, 30, 30)
    return clean, FloorTexture(dirty, 0.0, 0.0, res), bag_xy


def test_exclusion_removes_object_hits_in_patrol():
    clean, dirty, bag_xy = _textures()
    cam = wide_camera()
    H = cam.floor_homography()
    fg = FloorGrid(GRID, min_count=3)
    for i in range(30):
        p = (0.6 + 0.03 * i, 1.5, 0.0)
        fg.add_warp(warp_to_grid(render(cam, clean, p), H, p, GRID, ROI, supersample=SS))

    pose = (1.0, 1.5, 0.0)
    img = render(cam, dirty, pose)
    w = warp_to_grid(img, H, pose, GRID, ROI, supersample=SS)
    no_mask = compare_warp(w, fg, PARAMS["compare"])
    assert no_mask.hit.sum() > 30                         # 마스크 없으면 가방이 '오염'

    # (1) detectNet 박스: 가방이 찍힌 이미지 영역
    corners = map_to_robot(pose, [[bag_xy[0] - 0.05, bag_xy[1] - 0.08], [bag_xy[0] + 0.05, bag_xy[1] + 0.08]])
    uv = robot_to_pixel(H, np.array([[corners[0, 0], corners[0, 1]], [corners[1, 0], corners[1, 1]],
                                     [corners[0, 0], corners[1, 1]], [corners[1, 0], corners[0, 1]]]))
    det = Detection("backpack", 0.8, uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max())
    ex_img = boxes_to_mask(img.shape, [det], PARAMS["exclusion"]["detectnet_classes"], margin_px=5)
    keep = window_keep_mask(w, image_exclude=ex_img)
    assert not compare_warp(w, fg, PARAMS["compare"], mask=keep).hit.any()

    # (2) LiDAR: 가방 둘레의 점 (높이가 LiDAR에 걸리는 물체)
    ring = [(bag_xy[0] + 0.05 * math.cos(a), bag_xy[1] + 0.08 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 40)]
    pts = map_to_robot(pose, ring)
    keep = window_keep_mask(w, lidar_points=pts, pose=pose, grid=GRID, lidar_radius=0.05)
    left = compare_warp(w, fg, PARAMS["compare"], mask=keep).hit
    assert left.sum() <= 2                                # 반경 5 cm면 10 x 16 cm 가방을 덮음

    # 제외해도 나머지 바닥은 계속 관측된다
    assert compare_warp(w, fg, PARAMS["compare"], mask=keep).observed.sum() > 0.5 * no_mask.observed.sum()


def test_window_keep_mask_requires_pose_for_lidar():
    cam = wide_camera()
    w = warp_to_grid(np.zeros((480, 640, 3), np.uint8), cam.floor_homography(), (1, 1.5, 0), GRID, ROI)
    with pytest.raises(ValueError):
        window_keep_mask(w, lidar_points=np.array([[0.3, 0.0]]))
    assert window_keep_mask(w).all()


def test_lidar_shadow_covers_object_interior():
    """LiDAR는 물체 바깥면만 본다: 반경만으로는 12 cm 폭 물체의 가운데가 남고, 그림자를 더하면 덮인다."""
    pose = (1.0, 1.0, 0.0)
    # 로봇 앞 0.30 m에 놓인 물체(앞면 y -0.06~0.06)의 앞면 점들만 보임
    front = np.array([[0.30, y] for y in np.linspace(-0.06, 0.06, 7)])
    win = (90, 120, 40, 60)  # r0, c0, h, w → 창은 로봇 좌표 x 0.195~0.795 근처
    centre = map_to_cell(GRID, robot_to_map(pose, [[0.40, 0.0]]))[0] - win[:2]   # 물체 가운데(앞면에서 10 cm 뒤)
    only_radius = lidar_exclusion(front, pose, GRID, *win, radius=0.05)
    with_shadow = lidar_exclusion(front, pose, GRID, *win, radius=0.05, shadow=0.30, origin=(-0.064, 0.0))
    assert not only_radius[centre[0], centre[1]]
    assert with_shadow[centre[0], centre[1]]
    before = map_to_cell(GRID, robot_to_map(pose, [[0.22, 0.0]]))[0] - win[:2]   # 물체 앞쪽 8 cm (로봇 쪽)
    assert not with_shadow[before[0], before[1]]                                    # 앞쪽 바닥은 남는다
