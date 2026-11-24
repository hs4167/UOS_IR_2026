#!/usr/bin/env bash
# 위치 추정(AMCL)에 로봇의 현재 위치를 다시 알려준다 (RViz의 '2D Pose Estimate' 대신).
#
# 사용법 (젯슨, conda 끈 터미널):  ./scripts/set_pose.sh <x> <y> <yaw(도)>
#   예) ./scripts/set_pose.sh 0 0 0       SLAM 시작 자리로 돌려놓은 경우

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -ne 3 ]]; then
    echo "사용법: ./scripts/set_pose.sh <x> <y> <yaw(도)>"
    exit 1
fi
read -r QZ QW < <(python3 -c "import math; a=math.radians(float('$3')); print(math.sin(a/2), math.cos(a/2))") || exit 1
# shellcheck source=robot.sh
source "$SCRIPT_DIR/robot.sh" || exit 1
# 위치 분산 0.25 m², 방향 분산 0.068 rad² (RViz 기본값과 같음)
COV="[0.25,0,0,0,0,0, 0,0.25,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0.068]"
exec ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped \
    "{header: {frame_id: map}, pose: {pose: {position: {x: $1, y: $2, z: 0.0}, orientation: {z: $QZ, w: $QW}}, covariance: $COV}}"
