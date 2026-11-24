#!/usr/bin/env python3
"""[4] 재방문 판정 (오프라인): 청소 후 구역들을 다시 찍은 데이터셋 → 구역별 복원율과 처리 여부.

재방문 때 녹화만 해 두면 된다 (재방문 노드 또는 키보드 주행 + ./scripts/record.sh).
결과: results/<정찰>/revisit.json (+ 웹 페이지에 표시). --promote면 처리 완료 구역만 새 기준으로 (기존 기준 백업).

사용 예
  python scripts/run_revisit.py --run data/2026-11-25_revisit --results results/2026-11-25_patrol
  python scripts/run_revisit.py --run data/... --results results/... --promote
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from floor_dirt.config import REPO_ROOT, load_params  # noqa: E402
from floor_dirt.offline import load_camera, prepare_params, run_revisit  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="재방문 때 녹화한 데이터셋")
    ap.add_argument("--results", required=True, help="정찰 결과 폴더 (오염 구역)")
    ap.add_argument("--reference", default=os.path.join(REPO_ROOT, "maps", "reference"))
    ap.add_argument("--calib-dir", default=os.path.join(REPO_ROOT, "config"))
    ap.add_argument("--promote", action="store_true", help="처리 완료 구역을 새 기준으로 저장 (기존 기준 백업)")
    ap.add_argument("--no-dnn", action="store_true")
    ap.add_argument("--synthetic-detector", action="store_true", help="합성 데이터 전용 detectNet 대역")
    args = ap.parse_args()

    params = prepare_params(load_params(), args.no_dnn)
    H, maps = load_camera(os.path.join(args.calib_dir, "camera_intrinsics.yaml"),
                          os.path.join(args.calib_dir, "floor_homography.yaml"))
    excluder = None
    if args.synthetic_detector:
        from floor_dirt.masks import ImageExcluder
        from floor_dirt.synthetic import ColorBlobDetector
        excluder = ImageExcluder(params["exclusion"], detectnet=ColorBlobDetector(margin_px=10))
    rep = run_revisit(args.run, args.results, args.reference, params, H, maps, promote=args.promote,
                      excluder=excluder)
    print("{:>4} {:>10} {:>8} {:>8} {:>6}".format("구역", "판정", "복원율", "본 비율", "사진"))
    for r in rep["regions"]:
        print("{:>4} {:>10} {:>8} {:>8} {:>6}".format(
            r["id"], r["status_ko"], "-" if r["restoration_ratio"] is None else "{:.0%}".format(r["restoration_ratio"]),
            "{:.0%}".format(r["coverage"]), r["frames"]))
    s = rep["summary"]
    print("처리 완료 {}, 재청소 필요 {}, 확인 불가 {} → {}".format(
        s["cleaned"], s["needs_cleaning"], s["unverified"], os.path.join(args.results, "revisit.json")))
    if "promote" in rep:
        print("기준 갱신: {}칸 (백업 {})".format(rep["promote"]["cells_updated"], rep["promote"]["backup"]))


if __name__ == "__main__":
    main()
