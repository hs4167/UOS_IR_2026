import sys
import types

import cv2
import numpy as np
import pytest

from floor_dirt.config import load_params
from floor_dirt.debug_stream import DebugOutput
from floor_dirt.dirt_map import DirtMap
from floor_dirt.masks import Detection
from floor_dirt.projection import GridMeta, roi_corners_pixel, warp_to_grid
from floor_dirt.viz import (
    COL_CLEAN, COL_HATCH, COL_OCC, COL_REGION, debug_overlay, lidar_background, region_crop, render_dirt_map,
    save_png_atomic, window_to_image,
)
from floor_dirt.synthetic import FloorTexture, render, wide_camera

PARAMS = load_params()
GRID = GridMeta(0.01, 0.0, 0.0, 200, 100)  # 2 m x 1 m


def _ones(h, w):
    return np.ones((h, w), bool)


def _map_with_states():
    dm = DirtMap(GRID, PARAMS["dirt_map"])
    for _ in range(3):
        dm.update(0, 0, _ones(100, 100), np.zeros((100, 100), bool))   # 왼쪽 절반: 깨끗
    for _ in range(4):
        dm.update(0, 10, _ones(5, 6), _ones(5, 6), np.full((5, 6), 2.0, np.float32), 4)  # 아래쪽(y 작은) 의심 구역
    # 오른쪽 일부: 봤지만 기준 없음
    dm.update(40, 150, np.zeros((20, 30), bool), np.zeros((20, 30), bool), seen=_ones(20, 30))
    return dm


def test_render_colors_orientation_and_regions():
    dm = _map_with_states()
    regions, _ = dm.extract_regions()
    img = render_dirt_map(dm, regions, legend=False)
    assert img.shape == (100, 200, 3)
    # 격자 row 0 (y 최소)은 그림 맨 아래 → 의심 구역은 그림 아래쪽
    sus_px = img[97, 12]
    assert sus_px[2] > 150 and sus_px[0] < 50                  # 빨강·노랑 계열 (B 낮음, R 높음)
    assert tuple(img[30, 50]) == tuple(((255 + np.array(COL_CLEAN)) * 0.5).astype(np.uint8))
    assert tuple(img[50, 190]) == (255, 255, 255)               # 안 간 곳 = 배경
    hatch = img[100 - 60:100 - 40, 150:180].reshape(-1, 3)
    assert (hatch == COL_HATCH).all(axis=1).sum() > 20          # 봤지만 판정 못 함 = 빗금
    region_px = (img.reshape(-1, 3) == COL_REGION).all(axis=1).sum()
    assert region_px > 20                                       # 구역 테두리


def test_render_scale_crop_robot_and_path():
    dm = _map_with_states()
    img = render_dirt_map(dm, robot_pose=(0.5, 0.5, 0.0), path_xy=np.array([[0.1, 0.5], [0.5, 0.5]]), scale=3,
                          crop=(0, 0, 100, 100))
    assert img.shape == (300, 300, 3)
    # 로봇 중심 (0.5, 0.5) → 칸 50 → 그림 (150, 150) 근처, 반경 15칸 = 45 px 원
    ring = img[150, 150 + 45]
    assert ring[0] > ring[2]                                    # 파란 계열


def test_lidar_background(tmp_path):
    pgm = np.full((20, 40), 254, np.uint8)  # 0.05 m 칸 → 2 m x 1 m
    pgm[0, :] = 0                           # 맨 위 줄 = 벽 (y 최대)
    pgm[:, 0] = 205                         # 왼쪽 열 = 모름
    cv2.imwrite(str(tmp_path / "m.pgm"), pgm)
    (tmp_path / "m.yaml").write_text("image: m.pgm\nresolution: 0.05\norigin: [0, 0, 0]\nnegate: 0\n"
                                     "occupied_thresh: 0.65\nfree_thresh: 0.196\n")
    bg = lidar_background(str(tmp_path / "m.yaml"), GRID)
    assert bg.shape == (100, 200)
    assert (bg[-5:, 10:] == COL_OCC).all()                      # 격자 row 끝(y 최대) = 벽
    assert bg[50, 100] == 235 and bg[50, 2] == 190
    img = render_dirt_map(DirtMap(GRID, PARAMS["dirt_map"]), background=bg, legend=False)
    assert tuple(img[2, 100]) == (COL_OCC,) * 3                 # 그림 맨 위 = 벽
    with pytest.raises(ValueError):
        render_dirt_map(DirtMap(GRID, PARAMS["dirt_map"]), background=np.zeros((10, 10), np.uint8))


def test_save_png_atomic(tmp_path):
    path = str(tmp_path / "res" / "dirt_map.png")
    save_png_atomic(path, np.zeros((5, 5, 3), np.uint8))
    save_png_atomic(path, np.full((5, 5, 3), 9, np.uint8))
    assert cv2.imread(path)[0, 0, 0] == 9
    assert sorted(p.name for p in (tmp_path / "res").iterdir()) == ["dirt_map.png"]


# ---------------------------------------------------------------- 카메라 쪽

def _frame_with_paper():
    cam = wide_camera()
    H = cam.floor_homography()
    tex = np.full((1000, 1000, 3), (150, 160, 170), np.uint8)
    tex[480:520, 680:720] = (40, 40, 200)  # 지도 (1.4, 1.0) 근처 8 cm 종이
    pose = (1.0, 1.0, 0.0)
    img = render(cam, FloorTexture(tex, 0.0, 0.0, 0.002), pose)
    grid = GridMeta(0.01, 0.0, 0.0, 200, 200)
    return img, warp_to_grid(img, H, pose, grid, PARAMS["roi"]), H


def test_window_to_image_and_overlay():
    img, w, H = _frame_with_paper()
    hit = (w.patch[..., 2] > 180) & (w.patch[..., 0] < 80) & w.valid
    assert hit.sum() > 30
    back = window_to_image(hit, w, img.shape)
    paper_px = (img[..., 2] > 180) & (img[..., 0] < 80)
    overlap = (back & paper_px).sum() / paper_px.sum()
    assert overlap > 0.8                                        # 격자 hit가 카메라 화면의 종이 위치로 돌아옴

    roi_px = roi_corners_pixel(H, PARAMS["roi"])
    det = Detection("person", 0.91, 20, 30, 120, 200)
    out = debug_overlay(img, floor_mask=np.ones(img.shape[:2], bool), exclude_mask=None, roi_poly_px=roi_px,
                        detections=[det], hit_mask=back, text_lines=["frame 12", "8.3 ms"])
    assert out.shape == img.shape and out.dtype == np.uint8
    assert not np.array_equal(out, img)
    u, v = np.round(roi_px[0]).astype(int)
    assert tuple(out[min(v, 479), min(u, 639)]) != tuple(img[min(v, 479), min(u, 639)])
    assert (out[back][:, 1] > img[back][:, 1]).mean() > 0.9     # hit 부분이 노랗게


def test_region_crop():
    img, w, _ = _frame_with_paper()
    dm = DirtMap(GridMeta(0.01, 0.0, 0.0, 200, 200), PARAMS["dirt_map"])
    hit = (w.patch[..., 2] > 180) & (w.patch[..., 0] < 80) & w.valid
    for _ in range(3):
        dm.update(w.r0, w.c0, w.valid, hit)
    regions, _ = dm.extract_regions()
    crop = region_crop(img, w, regions[0])
    assert crop is not None and 20 < crop.shape[0] < 300 and 20 < crop.shape[1] < 400
    red = (crop[..., 2] > 180) & (crop[..., 0] < 80) & (crop[..., 1] < 80)
    assert red.sum() > 50                                       # 잘라낸 사진 안에 종이가 있음


# ---------------------------------------------------------------- 디버그 출력

def test_debug_output_none_and_file(tmp_path):
    none = DebugOutput({"output": "none"})
    assert not none.wants() and not none.push(0, np.zeros((4, 4, 3), np.uint8))
    out = DebugOutput({"output": "file", "every_n": 3}, str(tmp_path / "debug"))
    for i in range(7):
        if out.wants():
            out.push(i, np.zeros((4, 4, 3), np.uint8))
        else:
            out.skip()
    assert sorted(p.name for p in (tmp_path / "debug").iterdir()) == ["000000.jpg", "000003.jpg", "000006.jpg"]
    with pytest.raises(ValueError):
        DebugOutput({"output": "web"})


def test_debug_output_rtsp_with_fake_jetson(monkeypatch):
    ju = types.ModuleType("jetson_utils")
    rendered = []

    class Video:
        def __init__(self, url):
            self.url = url

        def Render(self, img):
            rendered.append(img)

    ju.videoOutput = Video
    ju.cudaFromNumpy = lambda a: a.shape
    monkeypatch.setitem(sys.modules, "jetson_utils", ju)
    out = DebugOutput(dict(PARAMS["debug"], output="rtsp"))
    assert out._video.url == PARAMS["debug"]["rtsp_url"]
    out.push(0, np.zeros((4, 6, 3), np.uint8))
    assert rendered == [(4, 6, 3)]
