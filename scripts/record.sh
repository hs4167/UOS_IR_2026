#!/usr/bin/env bash
# 주행 녹화 → data/<run 이름>/ (사진 + 찍은 순간의 위치 + LiDAR). 끝나면 Ctrl+C.
#
# 사용법 (젯슨, conda activate test1001 상태. bringup·카메라 노드가 켜져 있어야 함):
#   ./scripts/record.sh 2026-11-25_ref --note "형광등 전부, 블라인드 닫음"     지도 위치(map) 기준
#   ./scripts/record.sh test_odom --frame odom                               지도 없이 시험 녹화
#   ./scripts/record.sh <run> --help                                         옵션 전체

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ $# -lt 1 || "$1" == -* ]]; then
    echo "사용법: ./scripts/record.sh <run 이름> [--frame map|odom] [--rate 5] [--note \"메모\"]"
    exit 1
fi
RUN="$1"
shift
# shellcheck source=ros_conda_env.sh
source "$SCRIPT_DIR/ros_conda_env.sh"
[[ $ROS_ENV_OK -eq 1 ]] || exit 1
exec python "$REPO/ros_nodes/dataset_recorder.py" --out "$REPO/data/$RUN" "$@"
