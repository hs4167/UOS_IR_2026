"""주행 녹화 로직 (ROS와 무관, ros_nodes/dataset_recorder.py가 사용).

녹화 노드가 하는 일: 카메라 사진이 올 때마다
  1) 저장 주기(rate)에 맞는지 → 아니면 버림
  2) 대기열에 넣음
  3) 그 사진 시각(header.stamp)의 TF(map 또는 odom → base_footprint)가 준비되면 위치를 붙이고,
     시각이 가장 가까운 LiDAR 스캔(차이 ≤ slop)을 골라 함께 저장
  4) 너무 오래 TF가 안 오면 버림 (위치 없는 사진은 쓸모가 없다)
TF를 기다리느라 콜백을 막지 않으려고 '대기열 + 주기 처리' 방식을 쓴다 (CLAUDE.md §4: 사진 시각의 TF 사용).
"""

import math
from collections import deque
from typing import Deque, List, NamedTuple, Optional, Tuple


def quat_to_yaw(x, y, z, w):
    # type: (float, float, float, float) -> float
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def transform_to_pose(tx, ty, qx, qy, qz, qw):
    # type: (float, float, float, float, float, float) -> Tuple[float, float, float]
    """geometry_msgs/Transform 값 → (x, y, yaw)."""
    return float(tx), float(ty), float(quat_to_yaw(qx, qy, qz, qw))


class RateGate:
    """초당 rate장만 통과 (주행 속도 0.2 m/s, 5 Hz면 4 cm마다 1장)."""

    def __init__(self, rate_hz):
        self.period = 1.0 / float(rate_hz) if rate_hz > 0 else 0.0
        self.last = None

    def accept(self, t):
        # type: (float) -> bool
        if self.last is not None and t - self.last < self.period - 1e-6:
            return False
        self.last = t
        return True


class ScanBuffer:
    """최근 LiDAR 스캔 (시각, 점) 보관. 사진 시각에 가장 가까운 것을 고른다."""

    def __init__(self, keep_s=2.0):
        self.keep_s = keep_s
        self.items = deque()  # type: Deque[Tuple[float, object]]

    def add(self, t, points):
        self.items.append((t, points))
        while self.items and t - self.items[0][0] > self.keep_s:
            self.items.popleft()

    def nearest(self, t, slop):
        # type: (float, float) -> Optional[object]
        best = None
        for ts, pts in self.items:
            d = abs(ts - t)
            if d <= slop and (best is None or d < best[0]):
                best = (d, pts)
        return None if best is None else best[1]


class Pending(NamedTuple):
    stamp: float      # 사진 시각 [s]
    image: object     # 사진 (노드에서는 (BGR 배열, ROS 시각) 묶음)
    received: float   # 받은 시각 (대기 시간 계산용)


class PendingFrames:
    """TF를 기다리는 사진 대기열."""

    def __init__(self, max_wait_s=1.0, max_len=30):
        self.max_wait_s = max_wait_s
        self.items = deque(maxlen=max_len)  # type: Deque[Pending]
        self.dropped = 0

    def add(self, stamp, image, now):
        if len(self.items) == self.items.maxlen:
            self.dropped += 1   # 가장 오래된 것이 밀려남
        self.items.append(Pending(stamp, image, now))

    def ready(self, can_transform, now):
        # type: (callable, float) -> List[Pending]
        """TF가 준비된 사진은 꺼내 돌려주고, 오래 기다린 사진은 버린다. can_transform(Pending) → bool."""
        out, keep = [], deque(maxlen=self.items.maxlen)
        for p in self.items:
            if can_transform(p):
                out.append(p)
            elif now - p.received > self.max_wait_s:
                self.dropped += 1
            else:
                keep.append(p)
        self.items = keep
        return out
