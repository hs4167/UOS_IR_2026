#!/usr/bin/env python3
"""웹캠(C920) → /image_raw (sensor_msgs/Image, bgr8). 젯슨 전용, conda test1001 + Foxy.

시작할 때 config/camera_settings.yaml의 노출·초점·화이트밸런스 고정값을 적용한다.
header.stamp는 프레임을 받은 시각이다. 녹화 노드가 이 시각의 TF로 위치를 붙인다 (CLAUDE.md §4).
웹캠은 한 프로그램만 열 수 있으므로, 이 노드를 켜면 모니터는 --topic /image_raw 로 본다.

실행: ./scripts/camera.sh   (또는 python ros_nodes/camera_node.py --help)
"""

import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from floor_dirt.camera_settings import apply, describe, load_settings, read_controls  # noqa: E402
from floor_dirt.config import REPO_ROOT  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "config", "camera_settings.yaml"))
    ap.add_argument("--topic", default="/image_raw")
    ap.add_argument("--frame-id", default="camera_rgb_optical_frame", help="TurtleBot3 URDF의 카메라 광학 프레임")
    ap.add_argument("--no-settings", action="store_true", help="카메라 고정 설정을 적용하지 않음")
    args = ap.parse_args()

    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image

    cfg = load_settings(args.config)
    device = cfg.get("device", "/dev/video0")
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit("카메라를 열 수 없습니다: {} (모니터가 웹캠을 쓰는 중인지 확인)".format(device))
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.get("fourcc", "YUYV")))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg.get("width", 640)))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg.get("height", 480)))
    cap.set(cv2.CAP_PROP_FPS, int(cfg.get("fps", 30)))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 장치를 연 뒤에 적용한다 (여는 순간 드라이버가 일부 값을 되돌리는 경우가 있음)
    if not args.no_settings:
        _, warnings = apply(device, cfg.get("controls") or {})
        for w in warnings:
            print("[camera] 경고:", w, flush=True)
    print("[camera] {} {}x{} | {}".format(device, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), describe(read_controls(device))),
          flush=True)

    rclpy.init()
    node = rclpy.create_node("camera_node")
    pub = node.create_publisher(Image, args.topic, QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE))
    stop = threading.Event()
    stats = {"n": 0, "t0": time.monotonic()}

    def loop():
        while not stop.is_set() and rclpy.ok():
            ok, frame = cap.read()
            stamp = node.get_clock().now().to_msg()
            if not ok:
                time.sleep(0.05)
                continue
            msg = Image()
            msg.header.stamp = stamp
            msg.header.frame_id = args.frame_id
            msg.height, msg.width = frame.shape[:2]
            msg.encoding = "bgr8"
            msg.is_bigendian = 0
            msg.step = msg.width * 3
            msg.data = frame.tobytes()
            pub.publish(msg)
            stats["n"] += 1
            el = time.monotonic() - stats["t0"]
            if el >= 5.0:
                print("[camera] {:.1f} fps → {}".format(stats["n"] / el, args.topic), flush=True)
                stats["n"], stats["t0"] = 0, time.monotonic()

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    print("[camera] 송출 시작 (종료: Ctrl+C)", flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        th.join(timeout=1.0)
        cap.release()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
