#!/usr/bin/env python3
"""정찰 결과를 정답(줄자로 잰 오염 위치)과 비교: 재현율, 오탐 수, 위치 오차 (CLAUDE.md §8).

truth.json 형식
  {"match_radius": 0.05,
   "items": [{"name": "red_paper", "x": 1.13, "y": 1.42}, ...]}     # x, y = 지도 좌표 [m]

사용 예
  python scripts/evaluate.py --results results/2026-11-25_patrol --truth data/2026-11-25_patrol/truth.json
결과: <results>/evaluation.json + 보고서용 표 출력
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from floor_dirt.offline import evaluate_results, evaluation_markdown  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--radius", type=float, default=None, help="짝짓기 거리 [m] (기본 truth.json의 match_radius 또는 0.05)")
    args = ap.parse_args()
    ev = evaluate_results(args.results, args.truth, args.radius)
    print(evaluation_markdown(ev))
    print("\n저장: {}".format(os.path.join(args.results, "evaluation.json")))


if __name__ == "__main__":
    main()
