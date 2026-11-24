#!/usr/bin/env bash
# SLAM으로 그린 지도 저장 → maps/<이름>.pgm, maps/<이름>.yaml (기본 classroom, 기존 파일은 백업)
#
# 사용법 (젯슨, conda activate test1001, slam.sh가 켜져 있는 동안):
#   ./scripts/save_map.sh
#   ./scripts/save_map.sh classroom_v2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=ros_conda_env.sh
source "$SCRIPT_DIR/ros_conda_env.sh"
[[ $ROS_ENV_OK -eq 1 ]] || exit 1
exec python "$REPO/ros_nodes/map_saver.py" --out "$REPO/maps/${1:-classroom}"
