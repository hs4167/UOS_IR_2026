#!/usr/bin/env bash
# 위치 추정 (지도 서버 + AMCL): 저장된 지도 위에서 로봇 위치를 잡는다. 녹화(record.sh --frame map) 전에 켠다.
#
# 사용법 (젯슨, conda 끈 터미널, bringup이 켜져 있어야 함):
#   ./scripts/localize.sh                              maps/classroom.yaml, 시작 위치 (0, 0, 0°)
#   ./scripts/localize.sh --x 1.2 --y -0.5 --yaw 90    시작 위치 지정 (yaw는 도)
#   ./scripts/localize.sh --map maps/classroom_v2.yaml
# 로봇을 SLAM을 시작한 자리·방향에 두고 켜면 (0, 0, 0)이 맞다.
# 맞는지 확인: ./scripts/map_view.sh 에서 LiDAR 점이 지도 벽 위에 겹치는지. 어긋나면 ./scripts/set_pose.sh

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
[[ -f "$MAP" ]] || { echo "[localize.sh] 지도가 없습니다: $MAP (먼저 slam.sh → save_map.sh)"; exit 1; }
MAP="$(cd "$(dirname "$MAP")" && pwd)/$(basename "$MAP")"
# AMCL의 initial_pose.*는 실수형 파라미터: '--x 0'처럼 정수로 넘기면 launch가 int로 읽어 AMCL이 거부하므로 실수로 바꿔 넘긴다
read -r X Y YAW < <(python3 -c "import math; print(float('$X'), float('$Y'), math.radians(float('$YAW_DEG')))") || exit 1
[[ -n "$YAW" ]] || { echo "[localize.sh] --x --y --yaw 는 숫자여야 합니다"; exit 1; }

# shellcheck source=robot.sh
source "$SCRIPT_DIR/robot.sh" || exit 1
for p in nav2_map_server nav2_amcl nav2_lifecycle_manager; do
    ros2 pkg prefix "$p" >/dev/null 2>&1 || { echo "[localize.sh] $p 가 없습니다"; exit 1; }
done
echo "[localize.sh] 지도 $MAP, 시작 위치 x=$X y=$Y yaw=${YAW_DEG}°"
exec ros2 launch "$REPO/launch/localization.launch.py" map:="$MAP" x:="$X" y:="$Y" yaw:="$YAW"
