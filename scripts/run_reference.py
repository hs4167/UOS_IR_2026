#!/usr/bin/env python3
"""[2] 기준 주행 데이터셋 → 기준 바닥 지도 (maps/reference/).

사용 예
  python scripts/run_reference.py --run data/2026-11-25_ref --map maps/classroom.yaml
  python scripts/run_reference.py --run data/ref2 --map maps/classroom.yaml --merge     # 기존 기준에 합치기
  (WSL 합성) python scripts/run_reference.py --run data/synthetic_ref --map data/synthetic_map/classroom.yaml \
             --calib-dir data/synthetic_calib --no-dnn --out data/synthetic_reference
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from floor_dirt.config import REPO_ROOT, load_params  # noqa: E402
from floor_dirt.offline import load_camera, prepare_params, run_reference  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="기준 주행 프레임 데이터셋 폴더")
    ap.add_argument("--map", required=True, help="LiDAR 지도 yaml (격자 크기·원점)")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "maps", "reference"))
    ap.add_argument("--calib-dir", default=os.path.join(REPO_ROOT, "config"),
                    help="camera_intrinsics.yaml, floor_homography.yaml이 있는 폴더")
    ap.add_argument("--merge", action="store_true", help="기존 기준에 합치기 (기준 주행을 나눠 했을 때)")
    ap.add_argument("--no-dnn", action="store_true", help="segNet·detectNet 끄기 (WSL 등 젯슨이 아닐 때)")
    ap.add_argument("--results", default=None, help="처리 시간 기록 폴더 (기본 results/<run>)")
    args = ap.parse_args()

    params = prepare_params(load_params(), args.no_dnn)
    H, maps = load_camera(os.path.join(args.calib_dir, "camera_intrinsics.yaml"),
                          os.path.join(args.calib_dir, "floor_homography.yaml"))
    run = os.path.basename(os.path.normpath(args.run))
    cov = run_reference(args.run, args.map, args.out, params, H, maps, merge=args.merge,
                        results_dir=args.results or os.path.join(REPO_ROOT, "results", run))
    if cov.get("resources"):
        from floor_dirt.tegralog import format_summary
        print(format_summary(cov["resources"]))
    warn = float(params["reference"]["coverage_warn"])
    print("기준 저장: {}".format(args.out))
    print("바닥 중 기준이 있는 비율: {:.1%} (빈 곳 {:.2f} m²)".format(cov["covered_ratio"], cov["missing_area_m2"]))
    if cov["covered_ratio"] < warn:
        print("경고: {:.0%} 미만입니다. {}/coverage.png 의 빨간 곳을 다시 주행하고 --merge 로 합치세요".format(warn, args.out))


if __name__ == "__main__":
    main()
