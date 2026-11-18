#!/usr/bin/env bash
# 키보드 주행 (터미널 2에서 실행. 터미널 1에서 ./scripts/robot.sh 로 bringup이 켜져 있어야 함)
#
# 사용법: ./scripts/teleop.sh
#   w/x: 앞/뒤 속도 증가   a/d: 왼/오른쪽 회전 증가   s 또는 스페이스: 정지
#   종료: 먼저 s로 멈춘 뒤 Ctrl+C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=robot.sh
source "$SCRIPT_DIR/robot.sh" || exit 1

if ! ros2 node list 2>/dev/null | grep -q turtlebot3_node; then
    echo "[teleop.sh] 경고: turtlebot3_node가 보이지 않습니다. 터미널 1에서 ./scripts/robot.sh 가 실행 중인지," \
         "ROS_DOMAIN_ID($ROS_DOMAIN_ID)가 같은지 확인하세요"
fi

echo "[teleop.sh] 키보드 주행 시작. 종료 전에 반드시 s로 정지!"
exec ros2 run turtlebot3_teleop teleop_keyboard
