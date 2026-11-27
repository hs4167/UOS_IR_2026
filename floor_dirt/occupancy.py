"""LiDAR 점유 격자 지도 (nav_msgs/OccupancyGrid) 변환·저장·그리기. ROS와 무관.

OccupancyGrid: data[row * width + col], 값 -1 = 모름, 0~100 = 점유 확률(%).
  row 0은 origin(가장 작은 y) 쪽 → 바닥 격자(GridMeta)와 같은 규칙.
map_server 형식(pgm + yaml)으로 저장할 때는 사람이 보는 그림처럼 위아래를 뒤집는다 (pgm 첫 줄 = 가장 큰 y).
  흰색 254 = 빈 곳, 검정 0 = 벽·장애물, 회색 205 = 모름. 기준값은 nav2 map_saver 기본값과 같다.
이렇게 저장한 지도는 GridMeta.from_lidar_map / lidar_free_mask / viz.lidar_background가 그대로 읽는다.
"""

import math
import os
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

FREE, OCC, UNKNOWN = 254, 0, 205
FREE_THRESH = 0.25      # nav2 map_saver 기본값
OCC_THRESH = 0.65


def occupancy_to_image(data, width, height, free_thresh=FREE_THRESH, occ_thresh=OCC_THRESH):
    # type: (Sequence[int], int, int, float, float) -> np.ndarray
    """OccupancyGrid data → map_server pgm 그림 (height, width) uint8, 위아래 뒤집힘."""
    d = np.asarray(data, dtype=np.int16).reshape(height, width)
    img = np.full((height, width), UNKNOWN, np.uint8)
    img[(d >= 0) & (d <= free_thresh * 100)] = FREE
    img[d >= occ_thresh * 100] = OCC
    return img[::-1].copy()


def save_map(path_noext, data, width, height, resolution, origin, free_thresh=FREE_THRESH, occ_thresh=OCC_THRESH):
    # type: (str, Sequence[int], int, int, float, Tuple[float, float, float], float, float) -> Tuple[str, str]
    """maps/classroom → maps/classroom.pgm + maps/classroom.yaml."""
    d = os.path.dirname(os.path.abspath(path_noext))
    os.makedirs(d, exist_ok=True)
    pgm = path_noext + ".pgm"
    yml = path_noext + ".yaml"
    if not cv2.imwrite(pgm, occupancy_to_image(data, width, height, free_thresh, occ_thresh)):
        raise IOError("지도 그림 저장 실패: {}".format(pgm))
    with open(yml, "w", encoding="utf-8") as f:
        f.write("image: {}\nmode: trinary\nresolution: {}\norigin: [{}, {}, {}]\nnegate: 0\n"
                "occupied_thresh: {}\nfree_thresh: {}\n".format(
                    os.path.basename(pgm), float(resolution), float(origin[0]), float(origin[1]),
                    float(origin[2]), occ_thresh, free_thresh))
    return pgm, yml


def backup_map(path_noext, stamp):
    # type: (str, str) -> Optional[str]
    """기존 지도(pgm + yaml)를 <이름>_backup_<stamp>.*로 옮긴다. yaml의 image: 줄도 새 pgm 이름으로 고친다.

    고치지 않으면 백업 yaml이 새로 저장한 지도 그림을 가리켜, 백업을 열어도 새 지도가 보인다.
    기존 지도가 없으면 None.
    """
    if not os.path.exists(path_noext + ".yaml"):
        return None
    backup = path_noext + "_backup_" + stamp
    for ext in (".yaml", ".pgm"):
        if os.path.exists(path_noext + ext):
            os.replace(path_noext + ext, backup + ext)
    with open(backup + ".yaml", encoding="utf-8") as f:
        lines = f.read().splitlines()
    lines = ["image: " + os.path.basename(backup) + ".pgm" if ln.startswith("image:") else ln for ln in lines]
    with open(backup + ".yaml", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return backup


def map_stats(data, resolution):
    """빈 곳·벽·모름 넓이 [m²]."""
    d = np.asarray(data, dtype=np.int16)
    a = float(resolution) ** 2
    return {"free_m2": round(float(((d >= 0) & (d <= FREE_THRESH * 100)).sum() * a), 2),
            "occupied_m2": round(float((d >= OCC_THRESH * 100).sum() * a), 2),
            "unknown_m2": round(float((d < 0).sum() * a), 2)}


def render_map_view(data, width, height, resolution, origin, robot_pose=None, scan_points_map=None,
                    max_px=800, path=None):
    # type: (Sequence[int], int, int, float, Tuple[float, float, float], Optional[Tuple[float, float, float]], Optional[np.ndarray], int, Optional[Sequence[Tuple[float, float]]]) -> np.ndarray
    """지도 뷰어 그림 (BGR): 지도 + 로봇(파랑) + 현재 LiDAR 점(빨강, 지도 좌표) + 지나온 길.

    LiDAR 점이 벽 위에 겹치면 위치 추정이 맞는 것이다.
    origin yaw는 0으로 가정한다 (TurtleBot3 SLAM 지도는 0).
    """
    base = occupancy_to_image(data, width, height)
    s = max(1, int(max_px // max(width, height, 1)))
    img = cv2.cvtColor(cv2.resize(base, (width * s, height * s), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    H = img.shape[0]

    def to_px(xy):
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        u = (xy[:, 0] - origin[0]) / resolution * s
        v = H - (xy[:, 1] - origin[1]) / resolution * s
        return np.round(np.stack([u, v], axis=1)).astype(np.int32)

    if path is not None and len(path) >= 2:
        cv2.polylines(img, [to_px(path).reshape(-1, 1, 2)], False, (230, 180, 60), 1, cv2.LINE_AA)
    if scan_points_map is not None and len(scan_points_map):
        for u, v in to_px(scan_points_map):
            if 0 <= u < img.shape[1] and 0 <= v < H:
                cv2.circle(img, (int(u), int(v)), max(1, s // 2), (40, 40, 230), -1)
    if robot_pose is not None:
        x, y, yaw = robot_pose
        c = tuple(int(v) for v in to_px([[x, y]])[0])
        tip = tuple(int(v) for v in to_px([[x + 0.25 * math.cos(yaw), y + 0.25 * math.sin(yaw)]])[0])
        cv2.circle(img, c, max(int(round(0.15 / resolution * s)), 4), (200, 90, 0), 2, cv2.LINE_AA)
        cv2.line(img, c, tip, (200, 90, 0), 2, cv2.LINE_AA)
    return img
