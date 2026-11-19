#!/usr/bin/env python3
"""바닥에 놓은 체커보드 사진 1장으로 바닥 투영 행렬 H를 구해 config/floor_homography.yaml에 저장한다.

보드 놓는 법
  - 로봇 앞 바닥(근거리 ROI 안)에 평평하게, 가로 코너 방향이 로봇 좌우가 되게 놓는다.
  - 기준 코너 = 로봇에서 가장 가깝고 오른쪽에 있는 내부 코너 (이미지 오른쪽 아래).
  - 기준 코너의 base_footprint 좌표를 줄자로 잰다: x = 바퀴 축 중심에서 앞으로, y = 왼쪽 +.
  - 보드가 로봇 축과 비뚤어졌으면 --yaw (도, 반시계 +).

사용 예
  python scripts/calibrate_floor.py --image data/calib_floor/000.png --origin 0.250 -0.090
  → 결과 확인용 이미지: results/calib/floor_calibration.png (보정된 영상 + 코너 + ROI)
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from floor_dirt.calibration import (  # noqa: E402
    DEFAULT_FLOOR_PATH, DEFAULT_INTRINSICS_PATH, detect_floor_board, fit_floor_homography, load_intrinsics,
    save_floor_calibration,
)
from floor_dirt.config import REPO_ROOT, load_params  # noqa: E402
from floor_dirt.projection import roi_corners_pixel, undistort, undistort_maps  # noqa: E402


def draw_debug(img_undist, uv, H, roi):
    out = img_undist.copy() if img_undist.ndim == 3 else cv2.cvtColor(img_undist, cv2.COLOR_GRAY2BGR)
    for i, (u, v) in enumerate(uv):
        cv2.circle(out, (int(round(u)), int(round(v))), 3, (0, 0, 255) if i == 0 else (0, 200, 0), -1)
    poly = roi_corners_pixel(H, roi)
    cv2.polylines(out, [np.round(poly).astype(np.int32).reshape(-1, 1, 2)], True, (255, 160, 0), 2)
    cv2.putText(out, "ROI", tuple(np.round(poly[2]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 160, 0), 2)
    return out


def main():
    params = load_params()
    cal = params["calibration"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="바닥 보드 원본 사진 (왜곡 보정 전)")
    ap.add_argument("--origin", type=float, nargs=2, required=True, metavar=("X", "Y"),
                    help="기준 코너의 base_footprint 좌표 [m]")
    ap.add_argument("--yaw", type=float, default=0.0, help="보드 회전 [deg], 반시계 +")
    ap.add_argument("--pattern", default="{}x{}".format(*cal["pattern"]))
    ap.add_argument("--square", type=float, default=cal["square"])
    ap.add_argument("--intrinsics", default=DEFAULT_INTRINSICS_PATH)
    ap.add_argument("--out", default=DEFAULT_FLOOR_PATH)
    ap.add_argument("--debug", default=os.path.join(REPO_ROOT, "results", "calib", "floor_calibration.png"))
    args = ap.parse_args()

    pattern = tuple(int(v) for v in args.pattern.lower().split("x"))
    intr = load_intrinsics(args.intrinsics)
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit("이미지를 읽을 수 없습니다: {}".format(args.image))
    if img.shape[1::-1] != (intr.width, intr.height):
        raise SystemExit("해상도 불일치: 사진 {}x{}, 내부 보정 {}x{}".format(img.shape[1], img.shape[0], intr.width, intr.height))

    uv, xy = detect_floor_board(img, intr, pattern, args.square, tuple(args.origin), math.radians(args.yaw))
    calib = fit_floor_homography(uv, xy, intr.width, intr.height)
    print("코너 {}개, 맞춤 오차 평균 {:.2f} mm, 최대 {:.2f} mm".format(len(uv), calib.fit_mean * 1000, calib.fit_max * 1000))
    if calib.fit_max > 0.005:
        print("경고: 맞춤 오차가 5 mm를 넘습니다. 보드가 휘었거나 square 값이 틀렸을 수 있습니다")

    save_floor_calibration(args.out, calib, meta={
        "image": os.path.relpath(os.path.abspath(args.image), REPO_ROOT),
        "pattern": list(pattern), "square": args.square,
        "origin": [float(v) for v in args.origin], "yaw_deg": args.yaw,
    })
    print("저장: {}".format(args.out))

    undist = undistort(img, undistort_maps(intr.K, intr.D, (intr.width, intr.height)))
    os.makedirs(os.path.dirname(args.debug), exist_ok=True)
    cv2.imwrite(args.debug, draw_debug(undist, uv, calib.H, params["roi"]))
    print("확인용 이미지: {} (빨간 점 = 기준 코너, 주황 = 근거리 ROI)".format(args.debug))
    print("다음: 보드를 다른 위치에 놓고 scripts/check_floor_projection.py로 검사 (평균 1 cm 이하여야 통과)")


if __name__ == "__main__":
    main()
