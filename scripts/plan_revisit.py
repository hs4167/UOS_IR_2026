#!/usr/bin/env python3
"""[4] 재방문 계획: 정찰 결과의 오염 구역을 최단 순서로 돌 때 각 구역 앞에 설 자리.

결과: results/<정찰>/revisit_plan.json (재방문 노드가 읽음), revisit_plan.png (지도 위 순서·설 자리)

사용 예
  python scripts/plan_revisit.py --results results/2026-11-25_patrol --map maps/classroom.yaml
  python scripts/plan_revisit.py --results results/... --map maps/classroom.yaml --start 0 0 0   # 출발 자세 (yaw 도)
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from floor_dirt.config import load_params  # noqa: E402
from floor_dirt.offline import plan_from_results  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="정찰 결과 폴더")
    ap.add_argument("--map", default=None, help="LiDAR 지도 yaml (벽 피해서 설 자리 찾기, 그림 배경)")
    ap.add_argument("--start", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "YAW_DEG"))
    args = ap.parse_args()
    start = (args.start[0], args.start[1], math.radians(args.start[2]))
    plan = plan_from_results(args.results, args.map, start, load_params())
    print("{:>4} {:>4} {:>18} {:>26}".format("순서", "구역", "구역 중심 [m]", "설 자리 x, y [m], 방향 [deg]"))
    for p in plan:
        g = p["goal"]
        goal = "설 자리 없음 (주변이 막힘)" if g is None else "{:+.2f}, {:+.2f}, {:+.0f}".format(g[0], g[1], math.degrees(g[2]))
        print("{:>4} {:>4} {:>18} {:>26}".format(p["step"], p["id"], "{:.2f}, {:.2f}".format(*p["center"]), goal))
    print("저장: {}/revisit_plan.json, revisit_plan.png".format(args.results))


if __name__ == "__main__":
    main()
