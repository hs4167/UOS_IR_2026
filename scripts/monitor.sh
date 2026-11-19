#!/usr/bin/env bash
# 실시간 모니터 (카메라 + LiDAR + odom). 노트북 브라우저(Edge/Chrome)에서 http://<젯슨IP>:8080
#
# 사용법 (젯슨, conda activate test1001 상태에서. bringup은 다른 터미널에서 ./scripts/robot.sh):
#   ./scripts/monitor.sh                       웹캠 직접 + /scan + /odom
#   ./scripts/monitor.sh --no-ros              카메라만 (bringup 없이 장착 각도 확인)
#   ./scripts/monitor.sh --topic /image_raw    카메라 노드가 웹캠을 쓰는 중일 때
#   ./scripts/monitor.sh --help                옵션 전체
# 종료: Ctrl+C
#
# robot.sh와 달리 conda를 끄지 않는다 (우리 코드는 conda test1001에서 실행).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${CONDA_DEFAULT_ENV:-}" != "test1001" ]]; then
    echo "[monitor.sh] 경고: conda 환경이 test1001이 아닙니다 (현재: ${CONDA_DEFAULT_ENV:-없음})." \
         "'conda activate test1001' 후 실행하세요"
fi

if [[ -f /opt/ros/foxy/setup.bash ]]; then
    # shellcheck disable=SC1091
    source /opt/ros/foxy/setup.bash
    export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-7}"
    export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-waffle_pi}"
    export LDS_MODEL="${LDS_MODEL:-LDS-01}"
    echo "[monitor.sh] ROS_DISTRO=$ROS_DISTRO ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
else
    echo "[monitor.sh] /opt/ros/foxy 없음: ROS 없이 실행합니다 (--no-ros 또는 --dataset 필요)"
fi

exec python "$SCRIPT_DIR/live_monitor.py" "$@"
