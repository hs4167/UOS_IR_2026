"""[1] 지도 작성·위치 추정 준비 테스트: OccupancyGrid 저장·방향, 지도 뷰어 그림, AMCL 설정."""

import os

import numpy as np
import pytest
import yaml

from floor_dirt.config import REPO_ROOT
from floor_dirt.floor_grid import lidar_free_mask
from floor_dirt.occupancy import FREE, OCC, UNKNOWN, backup_map, map_stats, occupancy_to_image, render_map_view, save_map
from floor_dirt.projection import GridMeta, cell_to_map
from floor_dirt.viz import lidar_background


def _grid(w=40, h=30):
    """OccupancyGrid data: 전부 모름, 아래쪽(작은 y) 10줄은 빈 곳, 맨 위 줄은 벽."""
    d = np.full((h, w), -1, np.int8)
    d[:10, :] = 0
    d[-1, :] = 100
    d[5, 5] = 50          # 애매한 값(25~65%) = 모름
    return d.ravel().tolist()


def test_occupancy_to_image_values_and_flip():
    img = occupancy_to_image(_grid(), 40, 30)
    assert img.shape == (30, 40)
    assert img[0, 0] == OCC                    # OccupancyGrid 맨 위 줄(y 최대)은 그림 맨 위
    assert img[-1, 0] == FREE                  # row 0(y 최소)은 그림 맨 아래
    assert img[-6, 5] == UNKNOWN               # 50%는 모름
    assert img[15, 0] == UNKNOWN


def test_saved_map_is_read_back_in_same_orientation(tmp_path):
    """우리가 저장한 SLAM 지도를 바닥 격자·배경 코드가 같은 방향으로 읽는지."""
    origin = (-1.0, -0.5, 0.0)
    pgm, yml = save_map(str(tmp_path / "classroom"), _grid(), 40, 30, 0.05, origin)
    info = yaml.safe_load(open(yml))
    assert info["image"] == "classroom.pgm" and info["origin"] == [-1.0, -0.5, 0.0] and info["resolution"] == 0.05
    grid = GridMeta.from_lidar_map(yml, 0.01)
    assert grid == GridMeta(0.01, -1.0, -0.5, 200, 150)
    free = lidar_free_mask(yml, grid)
    rows = np.nonzero(free.any(axis=1))[0]
    assert rows.min() == 0 and rows.max() == 49                                    # 빈 곳 = y 최소 쪽 0.5 m
    y_top = cell_to_map(grid, [[rows.max(), 0]])[0, 1]
    assert y_top == pytest.approx(-0.5 + 0.495)
    bg = lidar_background(yml, grid)
    assert (bg[-5:] == 40).all() and (bg[:50] == 235).all()                        # 벽 = y 최대 쪽


def test_map_stats():
    s = map_stats(_grid(), 0.05)
    assert s["free_m2"] == pytest.approx(10 * 40 * 0.0025)
    assert s["occupied_m2"] == pytest.approx(40 * 0.0025)
    assert s["unknown_m2"] == pytest.approx((30 * 40 - 400 - 40) * 0.0025)


def test_render_map_view_robot_and_scan():
    origin = (0.0, 0.0, 0.0)
    img = render_map_view(_grid(), 40, 30, 0.05, origin, robot_pose=(1.0, 0.25, 0.0),
                          scan_points_map=np.array([[1.5, 1.45]]), max_px=400, path=[(0.2, 0.25), (1.0, 0.25)])
    s = 10                                                       # 400 // 40
    assert img.shape == (300, 400, 3)
    H = 300
    u, v = int(1.5 / 0.05 * s), int(H - 1.45 / 0.05 * s)
    assert img[v, u, 2] > 200 and img[v, u, 0] < 80              # LiDAR 점 = 빨강
    ru, rv = int(1.0 / 0.05 * s), int(H - 0.25 / 0.05 * s)
    ring = img[rv, ru + int(0.15 / 0.05 * s)]
    assert ring[0] > ring[2]                                     # 로봇 원 = 파랑


def test_amcl_config_for_turtlebot3():
    cfg = yaml.safe_load(open(os.path.join(REPO_ROOT, "config", "amcl.yaml")))
    p = cfg["amcl"]["ros__parameters"]
    assert p["base_frame_id"] == "base_footprint" and p["odom_frame_id"] == "odom" and p["global_frame_id"] == "map"
    assert p["robot_model_type"] == "differential" and p["scan_topic"] == "scan"
    assert p["use_sim_time"] is False and p["set_initial_pose"] is True
    assert p["update_min_d"] <= 0.1                              # 사진마다 위치가 덜 밀리게
    assert "map_server" in cfg
    # launch가 넘기는 값(지도 경로, 시작 위치)은 파일에 두지 않는다: Foxy에선 파일의 노드별 값이 launch 값보다 우선함
    assert "yaml_filename" not in cfg["map_server"]["ros__parameters"]
    assert "initial_pose" not in p


def test_backup_map_points_yaml_to_backup_image(tmp_path):
    """같은 이름으로 다시 저장할 때 백업 yaml이 새 지도가 아니라 백업 그림을 가리켜야 한다."""
    path = str(tmp_path / "box")
    assert backup_map(path, "x") is None                        # 기존 지도 없음
    old = [0] * 4 + [100] * 2
    save_map(path, old, 3, 2, 0.05, (-0.5, -0.3, 0.0))
    backup = backup_map(path, "20261127_154610")
    assert backup == path + "_backup_20261127_154610"
    assert not (tmp_path / "box.yaml").exists() and (tmp_path / "box_backup_20261127_154610.pgm").exists()
    meta = yaml.safe_load(open(backup + ".yaml"))
    assert meta["image"] == "box_backup_20261127_154610.pgm"
    assert meta["origin"] == [-0.5, -0.3, 0.0]
    save_map(path, [0] * 6, 3, 2, 0.05, (-0.7, -0.3, 0.0))     # 새 지도는 원래 이름 그대로
    assert yaml.safe_load(open(path + ".yaml"))["image"] == "box.pgm"
