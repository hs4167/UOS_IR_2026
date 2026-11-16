"""프레임 데이터셋: ROS 쪽과 처리 코드(floor_dirt) 사이의 경계 형식.

ROS 쪽(rosbag 추출, 실시간 노드)은 아래 형식으로만 데이터를 넘긴다. 이 형식은 ROS 버전과 무관하다.

    data/<run>/
    ├── meta.yaml          # 조명 상태, 시간대, 메모 등 (CLAUDE.md §5 카메라 설정)
    ├── poses.csv          # frame_id, stamp, x, y, yaw  (이미지 header.stamp 시점의 map→base_footprint)
    ├── frames/000000.png  # 카메라 원본 이미지 (왜곡 보정 전)
    └── scans/000000.npy   # (선택) LiDAR 점, base_footprint 기준 (N, 2) [m]
"""
import csv
import os
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple

import cv2
import numpy as np
import yaml

POSE_HEADER = ["frame_id", "stamp", "x", "y", "yaw"]


class Frame(NamedTuple):
    frame_id: int
    stamp: float                  # 초 (이미지 header.stamp)
    pose: Tuple[float, float, float]  # map 프레임 (x [m], y [m], yaw [rad])
    image: np.ndarray             # BGR uint8
    scan: Optional[np.ndarray]    # (N, 2) float32, base_footprint 기준, 없으면 None


def _frame_name(frame_id: int) -> str:
    return "{:06d}".format(frame_id)


class FrameDatasetWriter:
    """프레임을 하나씩 저장한다. with 문으로 쓰면 poses.csv가 확실히 닫힌다."""

    def __init__(self, run_dir: str, meta: Optional[Dict[str, Any]] = None):
        self.run_dir = run_dir
        os.makedirs(os.path.join(run_dir, "frames"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "scans"), exist_ok=True)
        with open(os.path.join(run_dir, "meta.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(meta or {}, f, allow_unicode=True, sort_keys=False)
        self._pose_file = open(os.path.join(run_dir, "poses.csv"), "w", newline="", encoding="utf-8")
        self._pose_writer = csv.writer(self._pose_file)
        self._pose_writer.writerow(POSE_HEADER)

    def write(self, frame: Frame) -> None:
        name = _frame_name(frame.frame_id)
        if not cv2.imwrite(os.path.join(self.run_dir, "frames", name + ".png"), frame.image):
            raise IOError("이미지 저장 실패: frame {}".format(frame.frame_id))
        if frame.scan is not None:
            np.save(os.path.join(self.run_dir, "scans", name + ".npy"),
                    np.asarray(frame.scan, dtype=np.float32).reshape(-1, 2))
        x, y, yaw = frame.pose
        self._pose_writer.writerow([frame.frame_id, repr(float(frame.stamp)),
                                    repr(float(x)), repr(float(y)), repr(float(yaw))])

    def close(self) -> None:
        if self._pose_file is not None:
            self._pose_file.close()
            self._pose_file = None

    def __enter__(self) -> "FrameDatasetWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class FrameDataset:
    """저장된 프레임 데이터셋을 읽는다. 이미지는 순회할 때 하나씩 읽는다(메모리 절약)."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        meta_path = os.path.join(run_dir, "meta.yaml")
        self.meta = {}  # type: Dict[str, Any]
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = yaml.safe_load(f) or {}
        self._rows = []  # type: List[Tuple[int, float, Tuple[float, float, float]]]
        with open(os.path.join(run_dir, "poses.csv"), "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self._rows.append((int(row["frame_id"]), float(row["stamp"]),
                                   (float(row["x"]), float(row["y"]), float(row["yaw"]))))

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> Frame:
        frame_id, stamp, pose = self._rows[idx]
        name = _frame_name(frame_id)
        image = cv2.imread(os.path.join(self.run_dir, "frames", name + ".png"), cv2.IMREAD_COLOR)
        if image is None:
            raise IOError("이미지 읽기 실패: frame {}".format(frame_id))
        scan_path = os.path.join(self.run_dir, "scans", name + ".npy")
        scan = np.load(scan_path) if os.path.exists(scan_path) else None
        return Frame(frame_id, stamp, pose, image, scan)

    def __iter__(self) -> Iterator[Frame]:
        for i in range(len(self)):
            yield self[i]
