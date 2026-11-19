#!/usr/bin/env python3
"""체커보드 이미지들로 카메라 내부 보정(K, D)을 하고 config/camera_intrinsics.yaml에 저장한다.

사용 예
  python scripts/calibrate_intrinsics.py --images "data/calib_intrinsics/*.png"
  (pattern·square 기본값은 config/params.yaml의 calibration 항목)
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from floor_dirt.calibration import (  # noqa: E402
    DEFAULT_INTRINSICS_PATH, calibrate_from_corners, corner_coverage, find_chessboard, save_intrinsics,
)
from floor_dirt.config import load_params  # noqa: E402


def main():
    params = load_params()["calibration"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help='glob 패턴, 예: "data/calib_intrinsics/*.png"')
    ap.add_argument("--pattern", default="{}x{}".format(*params["pattern"]), help="내부 코너 (가로x세로)")
    ap.add_argument("--square", type=float, default=params["square"], help="칸 한 변 [m] (자로 잰 값)")
    ap.add_argument("--out", default=DEFAULT_INTRINSICS_PATH)
    args = ap.parse_args()

    pattern = tuple(int(v) for v in args.pattern.lower().split("x"))
    paths = sorted(glob.glob(args.images))
    if not paths:
        raise SystemExit("이미지가 없습니다: {}".format(args.images))

    corners_list, size = [], None
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print("  읽기 실패: {}".format(p))
            continue
        h, w = img.shape[:2]
        if size is None:
            size = (w, h)
        elif size != (w, h):
            raise SystemExit("해상도가 다른 이미지가 섞여 있습니다: {}".format(p))
        c = find_chessboard(img, pattern)
        print("  {} {}".format("O" if c is not None else "X", os.path.basename(p)))
        if c is not None:
            corners_list.append(c)

    print("체커보드 검출: {}/{}장".format(len(corners_list), len(paths)))
    intr = calibrate_from_corners(corners_list, size, pattern, args.square)

    np.set_printoptions(precision=4, suppress=True)
    print("해상도: {}x{}".format(intr.width, intr.height))
    print("K =\n{}".format(intr.K))
    print("D = {}".format(intr.D))
    print("재투영 오차 RMS: {:.3f} px {}".format(intr.rms, "(양호)" if intr.rms < 0.5 else "(0.5 px 초과: 흐린 사진 제외 후 재시도)"))

    cov = corner_coverage(corners_list, size)
    print("화면 3x3 구역별 코너 수:\n{}".format(cov))
    if (cov == 0).any():
        print("경고: 코너가 하나도 없는 구역이 있습니다. 그 구석에 보드를 두고 더 찍어야 왜곡이 정확해집니다")

    save_intrinsics(args.out, intr, note="{}장, pattern {}, square {} m".format(len(corners_list), args.pattern, args.square))
    print("저장: {}".format(args.out))


if __name__ == "__main__":
    main()
