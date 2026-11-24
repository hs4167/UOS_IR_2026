#!/usr/bin/env python3
"""[3] 정찰 주행 데이터셋 → 오염 구역 결과 (results/<run>/).

처리 중에도 몇 초마다 dirt_map.png·dirt_regions.json을 갱신하므로 serve_results.py 웹에서 진행을 볼 수 있다.

사용 예
  python scripts/run_patrol.py --run data/2026-11-25_patrol --map maps/classroom.yaml
  (WSL 합성) python scripts/run_patrol.py --run data/synthetic_patrol --map data/synthetic_map/classroom.yaml \
             --reference data/synthetic_reference --calib-dir data/synthetic_calib --no-dnn --synthetic-detector
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from floor_dirt.config import REPO_ROOT, load_params  # noqa: E402
from floor_dirt.offline import load_camera, prepare_params, run_patrol  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="정찰 주행 프레임 데이터셋 폴더")
    ap.add_argument("--reference", default=os.path.join(REPO_ROOT, "maps", "reference"))
    ap.add_argument("--map", default=None, help="LiDAR 지도 yaml (결과 지도 배경, 선택)")
    ap.add_argument("--out", default=None, help="결과 폴더 (기본 results/<run 이름>)")
    ap.add_argument("--calib-dir", default=os.path.join(REPO_ROOT, "config"))
    ap.add_argument("--no-dnn", action="store_true", help="segNet·detectNet 끄기 (WSL 등 젯슨이 아닐 때)")
    ap.add_argument("--synthetic-detector", action="store_true",
                    help="합성 데이터 전용: 보라색 물체를 person으로 보는 detectNet 대역 사용")
    ap.add_argument("--debug", choices=["none", "file", "http"], default="none",
                    help="디버그 오버레이 (file: <out>/debug/*.jpg, http: 브라우저)")
    ap.add_argument("--update", type=float, default=None, help="중간 저장 간격 [s] (기본 params viz.update_interval_s, 0=끔)")
    args = ap.parse_args()

    params = prepare_params(load_params(), args.no_dnn)
    H, maps = load_camera(os.path.join(args.calib_dir, "camera_intrinsics.yaml"),
                          os.path.join(args.calib_dir, "floor_homography.yaml"))
    out = args.out or os.path.join(REPO_ROOT, "results", os.path.basename(os.path.normpath(args.run)))
    excluder = None
    if args.synthetic_detector:
        from floor_dirt.masks import ImageExcluder
        from floor_dirt.synthetic import ColorBlobDetector
        excluder = ImageExcluder(params["exclusion"], detectnet=ColorBlobDetector(margin_px=10))
    regions = run_patrol(args.run, args.reference, out, params, H, maps, map_yaml=args.map,
                         debug_output=args.debug, update_interval=args.update, excluder=excluder)
    print("결과: {} (오염 구역 {}개)".format(out, len(regions)))
    for r in regions:
        print("  #{:<2} 중심 ({:.2f}, {:.2f}) m  크기 {:.0f}x{:.0f} cm  확률 {:.2f}".format(
            r.id, r.center[0], r.center[1], r.size[0] * 100, r.size[1] * 100, r.max_p))


if __name__ == "__main__":
    main()
