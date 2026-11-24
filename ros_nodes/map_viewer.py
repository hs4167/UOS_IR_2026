#!/usr/bin/env python3
"""지도 뷰어: /map + 로봇 위치(TF map→base_footprint) + LiDAR를 브라우저로 본다 (RViz 대신).

SLAM 중에는 지도가 그려지는 과정을, 위치 추정(AMCL) 중에는 'LiDAR 점이 지도 벽 위에 겹치는지'를 확인한다.
겹치면 위치 추정이 맞고, 어긋나 있으면 ./scripts/set_pose.sh 로 시작 위치를 다시 알려준다.

실행: ./scripts/map_view.sh   → 브라우저 http://<젯슨IP>:8082
"""

import argparse
import math
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from floor_dirt.masks import scan_to_points  # noqa: E402
from floor_dirt.mjpeg_server import MjpegServer  # noqa: E402
from floor_dirt.occupancy import map_stats, render_map_view  # noqa: E402
from floor_dirt.recording import quat_to_yaw  # noqa: E402


def local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "<젯슨IP>"
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--rate", type=float, default=2.0, help="화면 갱신 [Hz]")
    args = ap.parse_args()

    import rclpy
    import tf2_ros
    from nav_msgs.msg import OccupancyGrid
    from rclpy.duration import Duration
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import LaserScan

    rclpy.init()
    node = rclpy.create_node("map_viewer")
    buf = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
    tf2_ros.TransformListener(buf, node)
    st = {"map": None, "scan": None, "path": []}

    def on_map(msg):
        st["map"] = msg

    def on_scan(msg):
        st["scan"] = msg

    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(OccupancyGrid, "/map", on_map, latched)
    node.create_subscription(LaserScan, "/scan", on_scan, qos_profile_sensor_data)
    srv = MjpegServer(port=args.port, title="UOS_IR_2026 map")
    print("[map_view] 브라우저: http://{}:{}".format(local_ip(), srv.port), flush=True)

    def lookup(target, source):
        try:
            t = buf.lookup_transform(target, source, Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return None
        tr, q = t.transform.translation, t.transform.rotation
        return tr.x, tr.y, quat_to_yaw(q.x, q.y, q.z, q.w)

    def on_timer():
        m = st["map"]
        if m is None:
            img = 40 * np.ones((240, 640, 3), "uint8")
            cv2.putText(img, "waiting for /map (SLAM or localization running?)", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
            srv.push(img)
            return
        info = m.info
        origin = (info.origin.position.x, info.origin.position.y, 0.0)
        robot = lookup("map", "base_footprint")
        scan_map = None
        sc = st["scan"]
        if sc is not None:
            sp = lookup("map", sc.header.frame_id or "base_scan")
            if sp is not None:
                scan_map = scan_to_points(sc.ranges, sc.angle_min, sc.angle_increment, sc.range_min, sc.range_max,
                                          sensor_pose=sp)
        if robot is not None:
            p = st["path"]
            if not p or math.hypot(robot[0] - p[-1][0], robot[1] - p[-1][1]) > 0.02:
                p.append(robot[:2])
        img = render_map_view(m.data, info.width, info.height, info.resolution, origin, robot, scan_map,
                              path=st["path"])
        sm = map_stats(m.data, info.resolution)
        line1 = "map {:.1f} x {:.1f} m | free {:.1f} m2 | unknown {:.1f} m2".format(
            info.width * info.resolution, info.height * info.resolution, sm["free_m2"], sm["unknown_m2"])
        line2 = "robot --" if robot is None else "robot x {:+.2f} y {:+.2f} yaw {:+.0f} deg".format(
            robot[0], robot[1], math.degrees(robot[2]))
        bar = 40 * np.ones((48, img.shape[1], 3), "uint8")
        cv2.putText(bar, line1, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(bar, line2, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        srv.push(np.vstack([bar, img]))

    node.create_timer(1.0 / args.rate, on_timer)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
