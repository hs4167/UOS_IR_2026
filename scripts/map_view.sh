#!/usr/bin/env bash
# 지도 뷰어 (RViz 대신): 지도 + 로봇 위치 + LiDAR → 브라우저 http://<젯슨IP>:8082
#
# 사용법 (젯슨, conda activate test1001): ./scripts/map_view.sh      종료: Ctrl+C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ros_conda_env.sh
source "$SCRIPT_DIR/ros_conda_env.sh"
[[ $ROS_ENV_OK -eq 1 ]] || exit 1
exec python "$SCRIPT_DIR/../ros_nodes/map_viewer.py" "$@"
