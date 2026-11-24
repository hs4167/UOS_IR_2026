#!/usr/bin/env bash
# 카메라 노드: 웹캠 → /image_raw (노출·초점 고정 설정 적용). 녹화 전에 켜 둔다.
#
# 사용법 (젯슨, conda activate test1001 상태):
#   ./scripts/camera.sh            종료: Ctrl+C
# 이 노드가 웹캠을 쓰는 동안 모니터는: ./scripts/monitor.sh --topic /image_raw

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ros_conda_env.sh
source "$SCRIPT_DIR/ros_conda_env.sh"
[[ $ROS_ENV_OK -eq 1 ]] || exit 1
exec python "$SCRIPT_DIR/../ros_nodes/camera_node.py" "$@"
