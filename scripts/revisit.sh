#!/usr/bin/env bash
# [4] 재방문 노드 실행: 계획의 설 자리로 이동 → 멈춤 → 촬영 → 판정 → results/<정찰>/revisit.json
#
# 사용법 (젯슨, conda activate test1001. robot.sh, nav.sh, camera.sh가 켜져 있어야 함):
#   ./scripts/revisit.sh --results results/2026-11-25_patrol
#   ./scripts/revisit.sh --results results/2026-11-25_patrol --record 2026-11-26_revisit --promote
#   ./scripts/revisit.sh --help
# 중간에 멈추려면 Ctrl+C (이동 중인 목표를 취소하고, 그때까지의 판정을 저장)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ros_conda_env.sh
source "$SCRIPT_DIR/ros_conda_env.sh"
[[ $ROS_ENV_OK -eq 1 ]] || exit 1
exec python "$SCRIPT_DIR/../ros_nodes/revisit_node.py" "$@"
