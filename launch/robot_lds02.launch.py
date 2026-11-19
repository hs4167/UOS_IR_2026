"""TurtleBot3 bringup (LDS-02 LiDAR용). scripts/robot.sh가 LDS_MODEL=LDS-02일 때 실행한다.

젯슨에 설치된 turtlebot3_bringup(2.1.0 계열)의 robot.launch.py는 LDS_MODEL을 보지 않고
항상 LDS-01 드라이버(hlds_laser_publisher)를 띄운다. 우리 LiDAR는 LDS-02(LD08)라서
그 드라이버로는 /scan이 나오지 않는다 (docs/실험_기록.md 2026-11-19).

공식 robot.launch.py와 같은 구성에서 LiDAR 드라이버만 ld08_driver로 바꿨다.
  - turtlebot3_state_publisher.launch.py (로봇 모델·TF)
  - ld08_driver (LDS-02 → /scan, frame_id base_scan, USB 변환 칩 CP2102를 찾아 포트 자동 선택)
  - turtlebot3_node (OpenCR ↔ /cmd_vel, /odom, /battery_state ...)
ld08_driver는 ~/ld08_ws에 따로 빌드한다 (README 참고).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model = os.environ.get("TURTLEBOT3_MODEL", "waffle_pi")
    bringup_dir = get_package_share_directory("turtlebot3_bringup")

    usb_port = LaunchConfiguration("usb_port", default="/dev/ttyACM0")
    tb3_param_dir = LaunchConfiguration(
        "tb3_param_dir", default=os.path.join(bringup_dir, "param", model + ".yaml"))
    use_sim_time = LaunchConfiguration("use_sim_time", default="false")

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value=use_sim_time,
                              description="Use simulation (Gazebo) clock if true"),
        DeclareLaunchArgument("usb_port", default_value=usb_port,
                              description="Connected USB port with OpenCR"),
        DeclareLaunchArgument("tb3_param_dir", default_value=tb3_param_dir,
                              description="Full path to turtlebot3 parameter file to load"),

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

        Node(
            package="turtlebot3_node",
            executable="turtlebot3_ros",
            parameters=[tb3_param_dir],
            arguments=["-i", usb_port],
            output="screen"),
    ])
