#!/usr/bin/env bash
# [1] SLAM 지도 작성 (slam_toolbox). bringup이 켜져 있어야 함. 다 그렸으면 ./scripts/save_map.sh
#
# 사용법 (젯슨, conda 끈 터미널):
#   ./scripts/slam.sh                  slam_toolbox (기본, RViz 없음)
#   ./scripts/slam.sh --cartographer   TurtleBot3 공식 cartographer (RViz도 띄우려 함)
# 지도가 그려지는 모습: ./scripts/map_view.sh → 브라우저 http://<젯슨IP>:8082
# 요령: 천천히(0.1 m/s 정도) 벽을 따라 한 바퀴 → 가운데 왕복. 시작한 자리가 지도 원점 근처가 됨

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=robot.sh
source "$SCRIPT_DIR/robot.sh" || exit 1

if [[ "${1:-}" == "--cartographer" ]]; then
    exec ros2 launch turtlebot3_cartographer cartographer.launch.py use_sim_time:=false
fi
if ! ros2 pkg prefix slam_toolbox >/dev/null 2>&1; then
    echo "[slam.sh] slam_toolbox가 없습니다. ./scripts/slam.sh --cartographer 를 쓰세요"
    exit 1
fi
echo "[slam.sh] slam_toolbox 시작 (종료: Ctrl+C, 그 전에 save_map.sh로 저장)"
# Foxy의 online_async_launch.py는 use_sim_time 기본값이 true라 실제 로봇에서는 반드시 false로
exec ros2 launch slam_toolbox online_async_launch.py use_sim_time:=false
