#!/usr/bin/env bash
# TurtleBot3 (ROS 2 Foxy) 환경 설정 + bringup 스크립트
#
# 사용법 (젯슨, conda 끈 상태에서):
#   ./scripts/robot.sh            터미널 1: 환경 설정 후 bringup 실행 (Ctrl+C로 종료)
#   source scripts/robot.sh       터미널 2~: 환경 설정만 (teleop, ros2 topic 등을 쓸 터미널)
#
# 기본값을 바꾸려면 앞에 붙여서 실행:
#   LDS_MODEL=LDS-01 ./scripts/robot.sh       (기본은 LDS-02. 우리 로봇 LiDAR는 LDS-02로 확인됨)
#   LD08_WS=~/other_ws ./scripts/robot.sh    (LDS-02 드라이버 작업공간 위치, 기본 ~/ld08_ws)
#   ROS_DOMAIN_ID=8 source scripts/robot.sh

# source로 불렸는지(환경 설정만) 직접 실행됐는지(bringup까지) 판별
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    _ROBOT_SOURCED=1
else
    _ROBOT_SOURCED=0
fi

_robot_fail() {
    echo "[robot.sh] 오류: $*" >&2
    if [[ $_ROBOT_SOURCED -eq 1 ]]; then return 1; else exit 1; fi
}

_robot_setup_env() {
    # 1. conda와 섞이면 ROS 라이브러리 경로가 꼬이므로 공식 TurtleBot3 실행은 conda 밖에서 한다
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        if [[ $_ROBOT_SOURCED -eq 1 ]] && type conda >/dev/null 2>&1; then
            echo "[robot.sh] conda 환경($(basename "$CONDA_PREFIX"))을 끕니다"
            conda deactivate
        else
            _robot_fail "conda 환경이 켜져 있습니다. 'conda deactivate' 후 다시 실행하세요" || return 1
        fi
    fi

    # 2. ROS 2 Foxy
    if [[ ! -f /opt/ros/foxy/setup.bash ]]; then
        _robot_fail "/opt/ros/foxy/setup.bash 가 없습니다 (젯슨에서 실행하세요)" || return 1
    fi
    # shellcheck disable=SC1091
    source /opt/ros/foxy/setup.bash

    # 3. 환경 변수 (이미 설정된 값이 있으면 그대로 사용)
    export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-7}"          # 7조. 다른 조 로봇과 통신이 섞이지 않게
    export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-waffle_pi}"
    export LDS_MODEL="${LDS_MODEL:-LDS-02}"             # 우리 LiDAR는 LDS-02 (2026-11-19 원시 데이터로 확인)

    # 4. LDS-02 드라이버(ld08_driver) 작업공간. 설치된 bringup에는 들어 있지 않아 따로 빌드한다
    if [[ "$LDS_MODEL" == "LDS-02" ]]; then
        local ws="${LD08_WS:-$HOME/ld08_ws}"
        if [[ -f "$ws/install/setup.bash" ]]; then
            # shellcheck disable=SC1091
            source "$ws/install/setup.bash"
        else
            echo "[robot.sh] 경고: LDS-02 드라이버 작업공간이 없습니다 ($ws/install/setup.bash)." \
                 "README '로봇 주행 → LDS-02 드라이버 설치' 참고"
        fi
    fi

    echo "[robot.sh] ROS_DISTRO=$ROS_DISTRO ROS_DOMAIN_ID=$ROS_DOMAIN_ID" \
         "TURTLEBOT3_MODEL=$TURTLEBOT3_MODEL LDS_MODEL=$LDS_MODEL"
}

_robot_check_devices() {
    if [[ ! -e /dev/ttyACM0 ]]; then
        _robot_fail "/dev/ttyACM0(OpenCR)이 없습니다. OpenCR USB 연결과 전원 스위치를 확인하세요" || return 1
    fi
    echo "[robot.sh] OpenCR: /dev/ttyACM0 확인"
    if [[ ! -e /dev/ttyUSB0 ]]; then
        echo "[robot.sh] 경고: /dev/ttyUSB0(LiDAR)이 없습니다. LiDAR 없이 진행합니다 (/scan 없음, 주행은 가능)"
    else
        echo "[robot.sh] LiDAR: /dev/ttyUSB0 확인"
    fi
}

if [[ $_ROBOT_SOURCED -eq 0 ]]; then
    _robot_setup_env || exit 1
    _robot_check_devices
    if [[ "$LDS_MODEL" == "LDS-02" ]]; then
        # 설치된 bringup은 LDS_MODEL을 무시하고 LDS-01 드라이버만 띄우므로 우리 launch 파일을 쓴다
        if ! ros2 pkg prefix ld08_driver >/dev/null 2>&1; then
            _robot_fail "ld08_driver(LDS-02 드라이버)를 찾을 수 없습니다. README '로봇 주행 → LDS-02 드라이버 설치' 참고"
        fi
        _repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
        # TB3_USE_IMU=0 ./scripts/robot.sh → 회전을 IMU 대신 바퀴 엔코더로 계산 (IMU 값이 멈춰 있을 때)
        _use_imu=true
        [[ "${TB3_USE_IMU:-1}" == "0" ]] && _use_imu=false
        echo "[robot.sh] bringup 시작: LDS-02, use_imu=$_use_imu (종료: Ctrl+C)"
        exec ros2 launch "$_repo/launch/robot_lds02.launch.py" use_imu:=$_use_imu
    fi
    echo "[robot.sh] bringup 시작: LDS-01 공식 launch (종료: Ctrl+C)"
    exec ros2 launch turtlebot3_bringup robot.launch.py
fi

# source 모드: 환경 설정만 하고, 성공 여부를 호출한 쪽에 돌려준다
# (teleop.sh 등이 'source robot.sh || exit 1' 로 실패를 감지할 수 있게)
_robot_setup_env
_robot_rc=$?
[[ $_robot_rc -eq 0 ]] && echo "[robot.sh] 환경 설정 완료"
unset -f _robot_fail _robot_setup_env _robot_check_devices
unset _ROBOT_SOURCED
return $_robot_rc
