"""위치 추정: 저장된 지도(map_server) + AMCL. scripts/localize.sh가 실행한다.

SLAM으로 만든 maps/classroom.yaml 위에서 LiDAR로 로봇 위치를 잡아 TF map → odom을 낸다.
녹화(record.sh --frame map)가 이 TF로 사진마다 지도 좌표 위치를 붙인다.
RViz는 띄우지 않는다 (확인은 ./scripts/map_view.sh 브라우저).
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_launch_description():
    map_yaml = LaunchConfiguration("map")
    params = LaunchConfiguration("params")
    x = LaunchConfiguration("x")
    y = LaunchConfiguration("y")
    yaw = LaunchConfiguration("yaw")
    return LaunchDescription([
        DeclareLaunchArgument("map", default_value=os.path.join(REPO, "maps", "classroom.yaml")),
        DeclareLaunchArgument("params", default_value=os.path.join(REPO, "config", "amcl.yaml")),
        DeclareLaunchArgument("x", default_value="0.0"),
        DeclareLaunchArgument("y", default_value="0.0"),
        DeclareLaunchArgument("yaw", default_value="0.0"),
        Node(package="nav2_map_server", executable="map_server", name="map_server", output="screen",
             parameters=[params, {"yaml_filename": map_yaml}]),
        Node(package="nav2_amcl", executable="amcl", name="amcl", output="screen",
             parameters=[params, {"initial_pose.x": x, "initial_pose.y": y, "initial_pose.yaw": yaw}]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager", name="lifecycle_manager_localization",
             output="screen",
             parameters=[{"use_sim_time": False, "autostart": True, "node_names": ["map_server", "amcl"]}]),
    ])
