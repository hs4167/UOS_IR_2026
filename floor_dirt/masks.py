"""[3] 제외 마스크: 비교·누적에서 뺄 영역을 만든다.

마스크는 두 단계에서 만든다.
- 이미지 단계 (왜곡 보정된 이미지 픽셀, True = 제외): 로봇 몸체, detectNet 박스, segNet 바닥 아닌 픽셀.
  바닥 평면 위에 있지 않은 물체 픽셀을 투영하면 엉뚱한(더 먼) 칸에 떨어지므로, 투영 전에 이미지에서 지운다.
  detectNet 박스는 바닥 접점만이 아니라 박스 전체를 지운다 (박스 위쪽 물체 픽셀이 먼 칸에 '오염'으로 찍히는 것 방지).
- 격자 단계 (창 칸, True = 제외): LiDAR 점 주변 반경 r.

jetson-inference(segNet, detectNet)는 젯슨에만 있으므로 클래스 생성 시점에 불러온다.
WSL 테스트는 박스·클래스 처리 같은 순수 함수와 가짜 모듈로 확인한다.
"""

import math
from typing import List, NamedTuple, Optional, Sequence, Tuple

import cv2
import numpy as np

from floor_dirt.projection import GridMeta, Pose, WarpResult, grid_matrix, apply_homography, pose_matrix, warp_mask


class Detection(NamedTuple):
    name: str
    confidence: float
    left: float
    top: float
    right: float
    bottom: float


# ---------------------------------------------------------------- 이미지 단계 (순수 함수)

def polygon_mask(shape, polygon):
    # type: (Tuple[int, int], Sequence[Sequence[float]]) -> np.ndarray
    """다각형 내부 True (h, w). polygon이 비어 있으면 전부 False."""
    mask = np.zeros(shape[:2], np.uint8)
    if polygon is not None and len(polygon) >= 3:
        pts = np.round(np.asarray(polygon, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def boxes_to_mask(shape, detections, classes, margin_px=0):
    # type: (Tuple[int, int], Sequence[Detection], Sequence[str], int) -> np.ndarray
    """제외 대상 클래스의 박스(사방 margin_px 확장)를 True로 (h, w)."""
    h, w = shape[:2]
    mask = np.zeros((h, w), bool)
    wanted = set(c.lower() for c in classes)
    for d in detections:
        if d.name.lower() not in wanted:
            continue
        x0 = max(int(math.floor(d.left)) - margin_px, 0)
        y0 = max(int(math.floor(d.top)) - margin_px, 0)
        x1 = min(int(math.ceil(d.right)) + margin_px, w)
        y1 = min(int(math.ceil(d.bottom)) + margin_px, h)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask


def find_class_index(names, keyword="floor"):
    # type: (Sequence[str], str) -> int
    """클래스 이름 목록에서 바닥 클래스 번호. 정확히 같은 이름 우선, 없으면 포함하는 이름."""
    low = [n.lower().strip() for n in names]
    kw = keyword.lower()
    if kw in low:
        return low.index(kw)
    hits = [i for i, n in enumerate(low) if kw in n]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise ValueError("'{}' 클래스가 없습니다. 모델 클래스: {}".format(keyword, ", ".join(names)))
    raise ValueError("'{}'를 포함하는 클래스가 여러 개입니다: {}".format(keyword, [names[i] for i in hits]))


def class_map_to_mask(class_map, class_index, out_shape):
    # type: (np.ndarray, int, Tuple[int, int]) -> np.ndarray
    """segNet 클래스 번호 지도(작은 해상도) → 원본 크기 bool 마스크 (가장 가까운 값으로 확대)."""
    cm = np.asarray(class_map).reshape(class_map.shape[0], class_map.shape[1])
    m = (cm == class_index).astype(np.uint8)
    h, w = out_shape[:2]
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return m.astype(bool)


# ---------------------------------------------------------------- 격자 단계

def scan_to_points(ranges, angle_min, angle_increment, range_min=0.0, range_max=float("inf"),
                   sensor_pose=(0.0, 0.0, 0.0)):
    # type: (Sequence[float], float, float, float, float, Tuple[float, float, float]) -> np.ndarray
    """LaserScan → base_footprint 기준 점 (N, 2). 범위 밖·inf·nan은 버린다.

    sensor_pose: base_footprint → LiDAR 프레임 (x, y, yaw). ROS 쪽에서 TF로 읽어 넘긴다.
    """
    r = np.asarray(ranges, dtype=np.float64)
    ang = angle_min + angle_increment * np.arange(len(r))
    ok = np.isfinite(r) & (r >= range_min) & (r <= range_max) & (r > 0)
    pts = np.stack([r[ok] * np.cos(ang[ok]), r[ok] * np.sin(ang[ok])], axis=1)
    return apply_homography(pose_matrix(sensor_pose), pts) if len(pts) else np.zeros((0, 2))


def disk_kernel(radius_cells):
    # type: (float) -> np.ndarray
    k = int(math.ceil(radius_cells))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    return ((xx * xx + yy * yy) <= radius_cells * radius_cells + 1e-9).astype(np.uint8)


def lidar_exclusion(points_robot, pose, grid, r0, c0, h, w, radius):
    # type: (np.ndarray, Pose, GridMeta, int, int, int, int, float) -> np.ndarray
    """LiDAR 점 주변 반경 radius [m] 안의 창 칸을 True (h, w).

    창 바로 밖의 점도 반경이 창 안으로 들어올 수 있으므로 창을 반경만큼 넓혀 찍고 잘라낸다.
    """
    out = np.zeros((h, w), bool)
    pts = np.asarray(points_robot, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return out
    rad_cells = radius / grid.resolution
    m = int(math.ceil(rad_cells))
    cr = apply_homography(grid_matrix(grid) @ pose_matrix(pose), pts)  # (col, row) 연속, 정수 = 칸 중심
    cols = np.floor(cr[:, 0] + 0.5).astype(np.int64) - (c0 - m)
    rows = np.floor(cr[:, 1] + 0.5).astype(np.int64) - (r0 - m)
    H, W = h + 2 * m, w + 2 * m
    keep = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    if not keep.any():
        return out
    canvas = np.zeros((H, W), np.uint8)
    canvas[rows[keep], cols[keep]] = 1
    if m > 0:
        canvas = cv2.dilate(canvas, disk_kernel(rad_cells))
    return canvas[m:m + h, m:m + w].astype(bool)


def window_keep_mask(result, image_exclude=None, lidar_points=None, pose=None, grid=None, lidar_radius=0.05):
    # type: (WarpResult, Optional[np.ndarray], Optional[np.ndarray], Optional[Pose], Optional[GridMeta], float) -> np.ndarray
    """창에서 비교·누적에 쓸 칸 (True = 사용). valid는 포함하지 않는다 (compare_warp·add_warp가 valid와 AND)."""
    h, w = result.valid.shape
    keep = np.ones((h, w), bool)
    if image_exclude is not None and image_exclude.any():
        # 경계 칸이 반쯤 물체에 걸치는 것을 막으려고 격자로 옮긴 뒤 1칸 넓힌다
        ex = warp_mask(image_exclude, result).astype(np.uint8)
        keep &= ~cv2.dilate(ex, np.ones((3, 3), np.uint8)).astype(bool)
    if lidar_points is not None and len(lidar_points):
        if pose is None or grid is None:
            raise ValueError("LiDAR 제외에는 pose와 grid가 필요합니다")
        keep &= ~lidar_exclusion(lidar_points, pose, grid, result.r0, result.c0, h, w, lidar_radius)
    return keep


# ---------------------------------------------------------------- jetson-inference 래퍼 (젯슨 전용)

def _import_jetson():
    try:
        import jetson_inference  # type: ignore
        import jetson_utils  # type: ignore
    except ImportError:
        try:
            import jetson.inference as jetson_inference  # type: ignore  # 예전 이름
            import jetson.utils as jetson_utils  # type: ignore
        except ImportError as e:
            raise ImportError("jetson-inference를 불러올 수 없습니다 (젯슨에서 실행하거나 use_segnet/use_detectnet을 끄세요): {}".format(e))
    return jetson_inference, jetson_utils


def _to_cuda(ju, bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return ju.cudaFromNumpy(np.ascontiguousarray(rgb))


class SegNetFloor:
    """segNet으로 바닥 픽셀을 찾는다. 바닥 클래스는 이름으로 찾는다."""

    def __init__(self, model="fcn-resnet18-sun-640x512", floor_keyword="floor"):
        self._ji, self._ju = _import_jetson()
        self.net = self._ji.segNet(model)
        names = [self.net.GetClassDesc(i) for i in range(self.net.GetNumClasses())]
        self.floor_index = find_class_index(names, floor_keyword)
        self.floor_name = names[self.floor_index]
        self._mask_buf = None

    def floor_mask(self, bgr):
        # type: (np.ndarray) -> np.ndarray
        """바닥이면 True (h, w)."""
        h, w = bgr.shape[:2]
        self.net.Process(_to_cuda(self._ju, bgr))
        if hasattr(self.net, "GetGridSize"):
            gw, gh = self.net.GetGridSize()
        else:
            gw, gh = self.net.GetGridWidth(), self.net.GetGridHeight()
        if self._mask_buf is None or (self._mask_buf.width, self._mask_buf.height) != (gw, gh):
            self._mask_buf = self._ju.cudaAllocMapped(width=gw, height=gh, format="gray8")
        self.net.Mask(self._mask_buf, gw, gh, filter_mode="point")  # gray8 = 칸별 클래스 번호
        self._ju.cudaDeviceSynchronize()
        return class_map_to_mask(self._ju.cudaToNumpy(self._mask_buf), self.floor_index, (h, w))


class DetectNetExcluder:
    """detectNet 물체 박스 중 제외 클래스만 마스크로."""

    def __init__(self, model="ssd-mobilenet-v2", threshold=0.5, classes=(), margin_px=10):
        self._ji, self._ju = _import_jetson()
        self.net = self._ji.detectNet(model, threshold=threshold)
        self.classes = list(classes)
        self.margin_px = int(margin_px)

    def detect(self, bgr):
        # type: (np.ndarray) -> List[Detection]
        dets = self.net.Detect(_to_cuda(self._ju, bgr), overlay="none")
        return [Detection(self.net.GetClassDesc(d.ClassID), float(d.Confidence),
                          float(d.Left), float(d.Top), float(d.Right), float(d.Bottom)) for d in dets]

    def exclude_mask(self, bgr):
        # type: (np.ndarray) -> Tuple[np.ndarray, List[Detection]]
        dets = self.detect(bgr)
        return boxes_to_mask(bgr.shape, dets, self.classes, self.margin_px), dets


class ImageExcluder:
    """params["exclusion"] 설정대로 이미지 단계 제외 마스크를 만든다 (True = 제외)."""

    def __init__(self, cfg, segnet=None, detectnet=None):
        self.cfg = cfg
        self.body_polygon = cfg.get("body_polygon_px") or []
        if segnet is None and cfg.get("use_segnet", False):
            segnet = SegNetFloor(cfg.get("segnet_model", "fcn-resnet18-sun-640x512"),
                                 cfg.get("segnet_floor_keyword", "floor"))
        if detectnet is None and cfg.get("use_detectnet", False):
            detectnet = DetectNetExcluder(cfg.get("detectnet_model", "ssd-mobilenet-v2"),
                                          cfg.get("detectnet_threshold", 0.5),
                                          cfg.get("detectnet_classes", []), cfg.get("detectnet_margin_px", 10))
        self.segnet = segnet
        self.detectnet = detectnet

    def __call__(self, bgr):
        # type: (np.ndarray) -> Tuple[np.ndarray, List[Detection]]
        ex = polygon_mask(bgr.shape, self.body_polygon)
        dets = []  # type: List[Detection]
        if self.segnet is not None:
            ex |= ~self.segnet.floor_mask(bgr)
        if self.detectnet is not None:
            box_ex, dets = self.detectnet.exclude_mask(bgr)
            ex |= box_ex
        return ex, dets
