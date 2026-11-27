"""TurtleBot3 bringup (LDS-02 LiDAR용). scripts/robot.sh가 LDS_MODEL=LDS-02일 때 실행한다.

젯슨에 설치된 turtlebot3_bringup(2.1.0 계열)의 robot.launch.py는 LDS_MODEL을 보지 않고
항상 LDS-01 드라이버(hlds_laser_publisher)를 띄운다. 우리 LiDAR는 LDS-02(LD08)라서
그 드라이버로는 /scan이 나오지 않는다 (docs/실험_기록.md 2026-11-19).

공식 robot.launch.py와 같은 구성에서 LiDAR 드라이버만 ld08_driver로 바꿨다.
  - turtlebot3_state_publisher.launch.py (로봇 모델·TF)
  - ld08_driver (LDS-02 → /scan, frame_id base_scan, USB 변환 칩 CP2102를 찾아 포트 자동 선택)
  - turtlebot3_node (OpenCR ↔ /cmd_vel, /odom, /battery_state ...)
ld08_driver는 ~/ld08_ws에 따로 빌드한다 (README 참고).

use_imu:=false (robot.sh 기본값. IMU를 쓰려면 TB3_USE_IMU=1)
  공식 설정은 회전 각도를 OpenCR IMU로 계산한다 (diff_drive_controller odometry.use_imu: true).
  IMU 값이 멈춰 있으면 회전해도 /odom 방향이 그대로라 AMCL 위치가 회전을 못 따라간다 (2026-11-27 실기).
  그때는 공식 설정 파일을 복사해 use_imu만 false로 바꾼 파일을 넘겨 바퀴 엔코더로 회전을 계산한다.
  (launch에서 dict로 덮어쓰면 Foxy에선 파일의 노드별 값이 우선해 적용되지 않으므로 파일 자체를 고친다)
"""

import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def tb3_params_without_imu(src):
    """공식 turtlebot3 설정 파일을 읽어 odometry.use_imu만 false로 바꾼 임시 파일 경로를 돌려준다."""
    with open(src) as f:
        cfg = yaml.safe_load(f)
    found = False
    for node in cfg.values():
        odom = (node or {}).get("ros__parameters", {}).get("odometry")
        if isinstance(odom, dict) and "use_imu" in odom:
            odom["use_imu"] = False
            found = True
    if not found:
        raise RuntimeError("{}에 odometry.use_imu가 없습니다".format(src))
    fd, path = tempfile.mkstemp(prefix="tb3_no_imu_", suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


def turtlebot3_node(context, usb_port, tb3_param_dir, use_imu):
    param_file = context.perform_substitution(tb3_param_dir)
    actions = []
    if context.perform_substitution(use_imu).strip().lower() in ("false", "0", "no"):
        param_file = tb3_params_without_imu(param_file)
        actions.append(LogInfo(msg="[robot_lds02] odometry.use_imu = false: 회전을 바퀴 엔코더로 계산 ({})".format(param_file)))
    actions.append(Node(
        package="turtlebot3_node",
        executable="turtlebot3_ros",
        parameters=[param_file],
        arguments=["-i", context.perform_substitution(usb_port)],
        output="screen"))
    return actions


def generate_launch_description():
    model = os.environ.get("TURTLEBOT3_MODEL", "waffle_pi")
    bringup_dir = get_package_share_directory("turtlebot3_bringup")

    usb_port = LaunchConfiguration("usb_port", default="/dev/ttyACM0")
    tb3_param_dir = LaunchConfiguration(
        "tb3_param_dir", default=os.path.join(bringup_dir, "param", model + ".yaml"))
    use_sim_time = LaunchConfiguration("use_sim_time", default="false")
    use_imu = LaunchConfiguration("use_imu", default="true")

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value=use_sim_time,
                              description="Use simulation (Gazebo) clock if true"),
        DeclareLaunchArgument("usb_port", default_value=usb_port,
                              description="Connected USB port with OpenCR"),
        DeclareLaunchArgument("tb3_param_dir", default_value=tb3_param_dir,
                              description="Full path to turtlebot3 parameter file to load"),
        DeclareLaunchArgument("use_imu", default_value=use_imu,
                              description="false = 회전을 IMU 대신 바퀴 엔코더로 계산"),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(bringup_dir, "launch", "turtlebot3_state_publisher.launch.py")),
            launch_arguments={"use_sim_time": use_sim_time}.items(),
        ),

        Node(
            package="ld08_driver",
            executable="ld08_driver",
            name="ld08_driver",
            output="screen"),

        OpaqueFunction(function=turtlebot3_node, args=[usb_port, tb3_param_dir, use_imu]),
    ])
