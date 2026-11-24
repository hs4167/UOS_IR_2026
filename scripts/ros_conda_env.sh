#!/usr/bin/env bash
# 우리 ROS 노드(conda test1001에서 실행)용 환경 설정. monitor.sh, camera.sh, record.sh가 source 한다.
# robot.sh와 달리 conda를 끄지 않는다.

if [[ "${CONDA_DEFAULT_ENV:-}" != "test1001" ]]; then
    echo "[env] 경고: conda 환경이 test1001이 아닙니다 (현재: ${CONDA_DEFAULT_ENV:-없음})." \
         "'conda activate test1001' 후 실행하세요"
fi

if [[ -f /opt/ros/foxy/setup.bash ]]; then
    # shellcheck disable=SC1091
    source /opt/ros/foxy/setup.bash
    export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-7}"
    export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-waffle_pi}"
    export LDS_MODEL="${LDS_MODEL:-LDS-02}"
    echo "[env] ROS_DISTRO=$ROS_DISTRO ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
    ROS_ENV_OK=1
else
    echo "[env] /opt/ros/foxy 없음 (젯슨에서 실행하세요)"
    ROS_ENV_OK=0
fi
