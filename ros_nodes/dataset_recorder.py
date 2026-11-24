#!/usr/bin/env python3
"""주행 녹화 → 프레임 데이터셋 (data/<run>/). 젯슨 전용, conda test1001 + Foxy.

사진마다 '그 사진을 찍은 순간'의 로봇 위치(TF: <frame> → base_footprint)와 가장 가까운 LiDAR 스캔을 붙여 저장한다.
저장된 폴더는 그대로 scripts/run_reference.py, run_patrol.py에 넣는다.

필요한 것: bringup(./scripts/robot.sh), 카메라 노드(./scripts/camera.sh).
  --frame map  : 지도 위 위치. SLAM 지도 + 위치 추정(AMCL)이 켜져 있어야 함 → 기준·정찰 주행용
  --frame odom : 바퀴 회전으로 잰 위치. 지도 없이 시험 녹화용 (오래 달리면 위치가 조금씩 틀어짐)

실행: ./scripts/record.sh <run 이름> [--frame odom] [--rate 5] [--note "형광등 전부 켬"]
"""

import argparse
import datetime
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from floor_dirt.camera_settings import load_settings  # noqa: E402
from floor_dirt.config import REPO_ROOT  # noqa: E402
from floor_dirt.dataset import Frame, FrameDatasetWriter  # noqa: E402
from floor_dirt.masks import scan_to_points  # noqa: E402
from floor_dirt.monitor import image_msg_to_bgr  # noqa: E402
from floor_dirt.recording import PendingFrames, RateGate, ScanBuffer, quat_to_yaw, transform_to_pose  # noqa: E402


def stamp_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="저장 폴더 (예: data/2026-11-25_ref)")
    ap.add_argument("--frame", choices=["map", "odom"], default="map")
    ap.add_argument("--rate", type=float, default=5.0, help="초당 저장 장수")
    ap.add_argument("--slop", type=float, default=0.05, help="사진·스캔 시각 차 허용 [s] (CLAUDE.md §4)")
    ap.add_argument("--max-wait", type=float, default=1.0, help="TF를 기다리는 최대 시간 [s]")
    ap.add_argument("--image-topic", default="/image_raw")
    ap.add_argument("--scan-topic", default="/scan")
    ap.add_argument("--note", default="", help="조명·시간대 등 메모 (meta.yaml에 기록)")
    ap.add_argument("--force", action="store_true", help="이미 있는 폴더에 덮어쓰기")
    args = ap.parse_args()

    if os.path.isdir(args.out) and os.listdir(args.out) and not args.force:
        raise SystemExit("이미 있는 폴더입니다: {} (다른 이름을 쓰거나 --force)".format(args.out))

    import rclpy
    import tf2_ros
    from rclpy.duration import Duration
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import Image, LaserScan

    cam_cfg_path = os.path.join(REPO_ROOT, "config", "camera_settings.yaml")
    meta = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "pose_frame": args.frame,
        "rate_hz": args.rate,
        "note": args.note,
        "camera_settings": load_settings(cam_cfg_path) if os.path.exists(cam_cfg_path) else None,
        "source": "dataset_recorder",
    }
    writer = FrameDatasetWriter(args.out, meta)

    rclpy.init()
    node = rclpy.create_node("dataset_recorder")
    buf = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
    tf2_ros.TransformListener(buf, node)
    gate = RateGate(args.rate)
    pending = PendingFrames(max_wait_s=args.max_wait)
    scans = ScanBuffer()
    st = {"saved": 0, "no_scan": 0, "lidar_pose": None, "t_log": time.monotonic()}

    def lidar_pose():
        """base_footprint → LiDAR(scan 프레임) 위치. 고정이라 한 번만 조회."""
        if st["lidar_pose"] is None:
            try:
                t = buf.lookup_transform("base_footprint", st.get("scan_frame", "base_scan"), Time())
                tr, q = t.transform.translation, t.transform.rotation
                st["lidar_pose"] = (tr.x, tr.y, quat_to_yaw(q.x, q.y, q.z, q.w))
                print("[record] LiDAR 위치 (base_footprint 기준): x {:+.3f} y {:+.3f} yaw {:+.1f} deg".format(
                    st["lidar_pose"][0], st["lidar_pose"][1], math.degrees(st["lidar_pose"][2])), flush=True)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                return None
        return st["lidar_pose"]

    def on_scan(msg):
        st["scan_frame"] = msg.header.frame_id or "base_scan"
        pose = lidar_pose()
        if pose is None:
            return
        pts = scan_to_points(msg.ranges, msg.angle_min, msg.angle_increment, msg.range_min, msg.range_max,
                             sensor_pose=pose)
        scans.add(stamp_sec(msg.header.stamp), pts.astype("float32"))

    def on_image(msg):
        t = stamp_sec(msg.header.stamp)
        if not gate.accept(t):
            return
        try:
            img = image_msg_to_bgr(msg.encoding, msg.height, msg.width, msg.step, msg.data)
        except ValueError as e:
            node.get_logger().warn(str(e), throttle_duration_sec=5.0)
            return
        pending.add(t, (img, Time.from_msg(msg.header.stamp)), time.monotonic())

    def can(p):
        return buf.can_transform(args.frame, "base_footprint", p.image[1])

    def on_timer():
        for p in pending.ready(can, time.monotonic()):
            img, ros_t = p.image
            try:
                tf = buf.lookup_transform(args.frame, "base_footprint", ros_t)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                pending.dropped += 1
                continue
            tr, q = tf.transform.translation, tf.transform.rotation
            pose = transform_to_pose(tr.x, tr.y, q.x, q.y, q.z, q.w)
            scan = scans.nearest(p.stamp, args.slop)
            if scan is None:
                st["no_scan"] += 1
            writer.write(Frame(st["saved"], p.stamp, pose, img, scan))
            st["saved"] += 1
        if time.monotonic() - st["t_log"] >= 5.0:
            print("[record] 저장 {}장 | 위치 못 붙여 버림 {} | 스캔 없음 {} | 대기 {}".format(
                st["saved"], pending.dropped, st["no_scan"], len(pending.items)), flush=True)
            st["t_log"] = time.monotonic()

    node.create_subscription(Image, args.image_topic, on_image, qos_profile_sensor_data)
    node.create_subscription(LaserScan, args.scan_topic, on_scan, qos_profile_sensor_data)
    node.create_timer(0.05, on_timer)
    print("[record] 녹화 시작 → {} (위치 기준 {}, {} 장/s). 종료: Ctrl+C".format(args.out, args.frame, args.rate),
          flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        writer.close()
        print("[record] 종료: {}장 저장 → {} (위치 못 붙여 버림 {}, 스캔 없음 {})".format(
            st["saved"], args.out, pending.dropped, st["no_scan"]), flush=True)
        if st["saved"] == 0:
            print("[record] 한 장도 저장하지 못했습니다. 카메라 노드(/image_raw)와 TF({} → base_footprint)를 확인하세요"
                  .format(args.frame), flush=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
