#!/usr/bin/env bash
# /odom 모니터링: 수신 주기, 위치, 방향, 속도, 누적 이동 거리를 0.5초마다 한 줄로 표시
# (bringup이 켜져 있어야 함. 오도메트리 정확도 확인용: 로봇을 1 m 밀거나 몰고 거리 비교)
#
# 사용법: ./scripts/odom_monitor.sh          종료: Ctrl+C
#         ./scripts/odom_monitor.sh raw      ros2 topic echo /odom 원본 출력

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=robot.sh
source "$SCRIPT_DIR/robot.sh" || exit 1

if [[ "${1:-}" == "raw" ]]; then
    exec ros2 topic echo /odom
fi

# ROS 2 Foxy의 rclpy를 쓰는 짧은 모니터 (시스템 python3, conda 밖)
exec python3 - <<'PYEOF'
import math
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


class OdomMonitor(Node):
    def __init__(self):
        super().__init__("odom_monitor")
        # sensor_data(best effort) 구독은 reliable / best effort 발행 모두와 호환된다
        self.create_subscription(Odometry, "/odom", self.on_odom, qos_profile_sensor_data)
        self.create_timer(0.5, self.on_timer)
        self.last = None
        self.start_xy = None
        self.prev_xy = None
        self.dist = 0.0
        self.count = 0
        self.window_start = time.monotonic()
        self.rate = 0.0

    def on_odom(self, msg):
        p = msg.pose.pose.position
        if self.start_xy is None:
            self.start_xy = (p.x, p.y)
        if self.prev_xy is not None:
            self.dist += math.hypot(p.x - self.prev_xy[0], p.y - self.prev_xy[1])
        self.prev_xy = (p.x, p.y)
        self.last = msg
        self.count += 1

    def on_timer(self):
        now = time.monotonic()
        elapsed = now - self.window_start
        if elapsed >= 1.0:
            self.rate = self.count / elapsed
            self.count = 0
            self.window_start = now
        if self.last is None:
            print("[odom_monitor] /odom 기다리는 중... (bringup 실행 중인지, ROS_DOMAIN_ID 확인)", flush=True)
            return
        p = self.last.pose.pose.position
        q = self.last.pose.pose.orientation
        yaw = math.degrees(math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)))
        v = self.last.twist.twist.linear.x
        w = math.degrees(self.last.twist.twist.angular.z)
        dx, dy = p.x - self.start_xy[0], p.y - self.start_xy[1]
        print("{:5.1f} Hz | x {:+7.3f} y {:+7.3f} m | yaw {:+7.1f} deg | v {:+5.2f} m/s w {:+6.1f} deg/s"
              " | 시작점부터 {:.3f} m, 누적 {:.3f} m".format(
                  self.rate, p.x, p.y, yaw, v, w, math.hypot(dx, dy), self.dist), flush=True)


def main():
    rclpy.init()
    node = OdomMonitor()
    print("[odom_monitor] 시작 (종료: Ctrl+C). 거리는 이 스크립트 시작 시점 기준", flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


main()
PYEOF
