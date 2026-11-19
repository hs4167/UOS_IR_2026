#!/usr/bin/env python3
"""바닥 투영 검사: 보정 때와 '다른' 위치의 실측점으로 오차를 재서 통과/실패를 판정한다.

통과 기준 (CLAUDE.md §5 [0]): 근거리 ROI 안 확인점 5개 이상, 오차 평균 ≤ 1 cm.
실패하면 종료 코드 1 (다음 단계로 가지 않는다).

확인점 주는 방법 두 가지 (함께 써도 된다)
  1) --board: 체커보드를 보정 때와 다른 위치에 놓고 찍은 사진 + 그 기준 코너 실측 좌표
       python scripts/check_floor_projection.py --board data/calib_floor/001.png --origin 0.380 0.020
  2) --marks: 바닥 테이프 표시의 CSV. 한 줄에 u,v,x,y
       (u, v = 원본 사진의 픽셀 좌표, 이미지 뷰어로 읽음 / x, y = 줄자로 잰 base_footprint 좌표 [m])
       python scripts/check_floor_projection.py --marks data/calib_floor/marks.csv
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from floor_dirt.calibration import (  # noqa: E402
    DEFAULT_FLOOR_PATH, DEFAULT_INTRINSICS_PATH, detect_floor_board, in_roi, load_floor_calibration,
    load_intrinsics, projection_error,
)
from floor_dirt.config import load_params  # noqa: E402
from floor_dirt.projection import pixel_to_robot, undistort_points  # noqa: E402


def load_marks(path, intr):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line or line.lower().startswith("u"):
                continue
            rows.append([float(v) for v in line.split(",")])
    if not rows:
        return np.zeros((0, 2)), np.zeros((0, 2))
    arr = np.array(rows, dtype=np.float64)
    return undistort_points(intr.K, intr.D, arr[:, :2]), arr[:, 2:4]


def main():
    params = load_params()
    cal = params["calibration"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", help="확인용 보드 사진 (원본)")
    ap.add_argument("--origin", type=float, nargs=2, metavar=("X", "Y"), help="확인용 보드 기준 코너 실측 좌표 [m]")
    ap.add_argument("--yaw", type=float, default=0.0, help="확인용 보드 회전 [deg]")
    ap.add_argument("--marks", help="u,v,x,y CSV")
    ap.add_argument("--pattern", default="{}x{}".format(*cal["pattern"]))
    ap.add_argument("--square", type=float, default=cal["square"])
    ap.add_argument("--intrinsics", default=DEFAULT_INTRINSICS_PATH)
    ap.add_argument("--floor", default=DEFAULT_FLOOR_PATH)
    args = ap.parse_args()
    if not args.board and not args.marks:
        ap.error("--board 또는 --marks 중 하나는 필요합니다")
    if args.board and args.origin is None:
        ap.error("--board에는 --origin이 필요합니다")

    intr = load_intrinsics(args.intrinsics)
    floor = load_floor_calibration(args.floor)
    uv_all, xy_all = [], []
    if args.board:
        img = cv2.imread(args.board)
        if img is None:
            raise SystemExit("이미지를 읽을 수 없습니다: {}".format(args.board))
        pattern = tuple(int(v) for v in args.pattern.lower().split("x"))
        uv, xy = detect_floor_board(img, intr, pattern, args.square, tuple(args.origin), math.radians(args.yaw))
        uv_all.append(uv)
        xy_all.append(xy)
    if args.marks:
        uv, xy = load_marks(args.marks, intr)
        uv_all.append(uv)
        xy_all.append(xy)
    uv = np.vstack(uv_all)
    xy = np.vstack(xy_all)

    inside = in_roi(xy, params["roi"])
    if (~inside).any():
        print("ROI 밖 확인점 {}개는 판정에서 제외".format(int((~inside).sum())))
    stats = projection_error(floor.H, uv[inside], xy[inside],
                             pass_mean=cal["floor_pass_mean"], min_points=cal["floor_min_marks"])

    pred = pixel_to_robot(floor.H, uv[inside])
    print("{:>4} {:>16} {:>16} {:>9}".format("#", "실측 x, y [m]", "투영 x, y [m]", "오차 [mm]"))
    for i, (t, p, e) in enumerate(zip(xy[inside], pred, stats.errors)):
        print("{:>4} {:>7.3f},{:>8.3f} {:>7.3f},{:>8.3f} {:>9.1f}".format(i, t[0], t[1], p[0], p[1], e * 1000))
    print("확인점 {}개 (최소 {}), 평균 {:.1f} mm, 최대 {:.1f} mm (기준 평균 ≤ {:.0f} mm)".format(
        stats.n, cal["floor_min_marks"], stats.mean * 1000, stats.max * 1000, cal["floor_pass_mean"] * 1000))
    print("통과" if stats.passed else "실패")
    sys.exit(0 if stats.passed else 1)


if __name__ == "__main__":
    main()
