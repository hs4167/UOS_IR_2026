#!/usr/bin/env bash
# Nav2 (지도 + 위치 추정 + 경로 계획·주행). 재방문 노드가 이걸로 구역 앞까지 간다. RViz 없음.
# TurtleBot3 공식 Nav2 설정(turtlebot3_navigation2/param/<모델>.yaml)을 쓴다.
# 위치 추정(AMCL)이 포함되어 있으므로 localize.sh와 동시에 켜지 않는다.
#
# 사용법 (젯슨, conda 끈 터미널, bringup이 켜져 있어야 함):
#   ./scripts/nav.sh                              maps/classroom.yaml, 시작 위치 (0, 0, 0°)
#   ./scripts/nav.sh --x 1.2 --y -0.5 --yaw 90    시작 위치 지정
#   ./scripts/nav.sh --map maps/classroom_v2.yaml
# 확인: ./scripts/map_view.sh 에서 LiDAR 점이 벽에 겹치는지. 종료: Ctrl+C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
MAP="$REPO/maps/classroom.yaml"
X=0.0; Y=0.0; YAW_DEG=0.0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --map) MAP="$2"; shift 2 ;;
        --x) X="$2"; shift 2 ;;
        --y) Y="$2"; shift 2 ;;
        --yaw) YAW_DEG="$2"; shift 2 ;;
        *) echo "알 수 없는 옵션: $1"; exit 1 ;;
    esac
done
[[ -f "$MAP" ]] || { echo "[nav.sh] 지도가 없습니다: $MAP (먼저 slam.sh → save_map.sh)"; exit 1; }
MAP="$(cd "$(dirname "$MAP")" && pwd)/$(basename "$MAP")"

# shellcheck source=robot.sh
source "$SCRIPT_DIR/robot.sh" || exit 1
for p in nav2_bringup turtlebot3_navigation2; do
    ros2 pkg prefix "$p" >/dev/null 2>&1 || { echo "[nav.sh] $p 가 없습니다"; exit 1; }
done
PARAMS="$(ros2 pkg prefix turtlebot3_navigation2)/share/turtlebot3_navigation2/param/${TURTLEBOT3_MODEL}.yaml"
[[ -f "$PARAMS" ]] || { echo "[nav.sh] Nav2 설정 파일이 없습니다: $PARAMS"; exit 1; }

echo "[nav.sh] 지도 $MAP"
echo "[nav.sh] 설정 $PARAMS"
ros2 launch nav2_bringup bringup_launch.py map:="$MAP" params_file:="$PARAMS" use_sim_time:=False autostart:=True &
LAUNCH_PID=$!
trap 'kill -INT $LAUNCH_PID 2>/dev/null; wait $LAUNCH_PID' INT TERM

# Nav2가 뜨면 시작 위치를 알려준다 (공식 설정은 시작 위치를 자동으로 잡지 않음)
for _ in $(seq 1 30); do
    sleep 1
    ros2 node list 2>/dev/null | grep -q "/amcl" && break
done
sleep 2
echo "[nav.sh] 시작 위치 알림: x=$X y=$Y yaw=${YAW_DEG}°"
"$SCRIPT_DIR/set_pose.sh" "$X" "$Y" "$YAW_DEG" >/dev/null
echo "[nav.sh] 준비 완료. 재방문: conda activate test1001 && ./scripts/revisit.sh --results results/<정찰 run>"
wait $LAUNCH_PID
