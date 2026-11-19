#!/usr/bin/env python3
"""주행 중 실시간 모니터: 카메라 화면 + LiDAR + odom을 한 화면으로 합쳐 노트북 브라우저로 본다.

보정·기준 지도 없이도 동작한다 (카메라 장착 각도, LiDAR 연결, 주행 상태 확인용).
config/에 보정 파일이 있으면 카메라 화면에 근거리 ROI와 LiDAR 점의 바닥 위치를 겹친다.

젯슨 (터미널 1: ./scripts/robot.sh 로 bringup, 터미널 2: conda activate test1001 후)
  ./scripts/monitor.sh                         # 웹캠 직접 + /scan + /odom → 브라우저
  ./scripts/monitor.sh --topic /image_raw      # 카메라 노드가 웹캠을 잡고 있을 때
  ./scripts/monitor.sh --no-ros                # 카메라만 (bringup 없이 장착 각도 확인)
노트북: Edge/Chrome에서 http://<젯슨IP>:8080 (설치 불필요)
        --output rtsp 이면 VLC에서 rtsp://<젯슨IP>:8554/monitor

WSL (녹화된 프레임 데이터셋 재생, 화면 구성 확인용)
  python scripts/live_monitor.py --dataset data/synthetic_patrol --output file --out-dir results/monitor
"""

import argparse
import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from floor_dirt.calibration import (  # noqa: E402
    DEFAULT_FLOOR_PATH, DEFAULT_INTRINSICS_PATH, load_floor_calibration, load_intrinsics,
)
from floor_dirt.config import REPO_ROOT, load_params  # noqa: E402
from floor_dirt.debug_stream import DebugOutput  # noqa: E402
from floor_dirt.jetson_power import input_rail, read_rails  # noqa: E402
from floor_dirt.masks import scan_to_points  # noqa: E402
from floor_dirt.monitor import (  # noqa: E402
    RateMeter, battery_percent, camera_panel, closest_point, compose_dashboard, image_msg_to_bgr, lidar_topdown,
    power_line,
    quaternion_to_yaw, status_lines,
)
from floor_dirt.projection import undistort, undistort_maps  # noqa: E402


# ---------------------------------------------------------------- 입력

class DirectCamera:
    """웹캠 직접 읽기 (V4L2). 노출·초점 설정 상태도 읽어 화면에 표시한다."""

    def __init__(self, device, width, height, fourcc):
        dev = int(device) if device.isdigit() else device
        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise SystemExit("카메라를 열 수 없습니다: {} (v4l2-ctl --list-devices, 다른 프로그램이 쓰는 중인지 확인)".format(device))
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._ctrl_t = 0.0
        self._ctrl = {}

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def controls(self):
        """1초에 한 번만 조회 (V4L2 조회도 시간이 든다)."""
        now = time.monotonic()
        if now - self._ctrl_t > 1.0:
            g = self.cap.get
            ae = g(cv2.CAP_PROP_AUTO_EXPOSURE)  # V4L2: 1 = 수동, 3 = 자동
            self._ctrl = {
                "AE": "MANUAL" if ae == 1 else "AUTO" if ae == 3 else "{:g}".format(ae),
                "exp": "{:g}".format(g(cv2.CAP_PROP_EXPOSURE)),
                "AF": "AUTO" if g(cv2.CAP_PROP_AUTOFOCUS) == 1 else "OFF",
                "focus": "{:g}".format(g(cv2.CAP_PROP_FOCUS)),
                "AWB": "AUTO" if g(cv2.CAP_PROP_AUTO_WB) == 1 else "OFF",
            }
            self._ctrl_t = now
        return self._ctrl

    def close(self):
        self.cap.release()


class RosInputs:
    """/scan, /odom (+ 선택: 이미지 토픽) 구독. rclpy는 이 클래스를 만들 때 불러온다."""

    def __init__(self, image_topic=None, lidar_pose=(0.0, 0.0, 0.0)):
        try:
            import rclpy
            from nav_msgs.msg import Odometry
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import BatteryState, Image, LaserScan
        except ImportError as e:
            raise SystemExit("rclpy를 불러올 수 없습니다: {}\n"
                             "  source /opt/ros/foxy/setup.bash 후 실행하거나 ./scripts/monitor.sh 를 쓰세요."
                             " ROS 없이 보려면 --no-ros".format(e))
        self.rclpy = rclpy
        rclpy.init()
        self.node = rclpy.create_node("live_monitor")
        self.lock = threading.Lock()
        self.lidar_pose = lidar_pose
        self.scan_pts = None
        self.scan_rate = RateMeter()
        self.odom = None
        self.odom_rate = RateMeter()
        self.image = None
        self.image_rate = RateMeter()
        self.battery = None
        self.node.create_subscription(LaserScan, "/scan", self._on_scan, qos_profile_sensor_data)
        self.node.create_subscription(Odometry, "/odom", self._on_odom, qos_profile_sensor_data)
        self.node.create_subscription(BatteryState, "/battery_state", self._on_battery, qos_profile_sensor_data)
        if image_topic:
            self.node.create_subscription(Image, image_topic, self._on_image, qos_profile_sensor_data)
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def _spin(self):
        try:
            self.rclpy.spin(self.node)
        except Exception as e:  # 종료 시 shutdown 예외는 무시, 그 외는 알림
            if self.rclpy.ok():
                print("[monitor] ROS 수신 스레드 중단: {!r}".format(e), flush=True)

    def _on_scan(self, msg):
        try:
            pts = scan_to_points(msg.ranges, msg.angle_min, msg.angle_increment, msg.range_min, msg.range_max,
                                 sensor_pose=self.lidar_pose)
        except Exception as e:  # 스캔 하나가 이상해도 수신은 계속
            self.node.get_logger().error("스캔 처리 실패: {!r}".format(e), throttle_duration_sec=5.0)
            return
        with self.lock:
            self.scan_pts = pts
            self.scan_rate.tick()

    def _on_odom(self, msg):
        p, q, t = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist
        with self.lock:
            self.odom = {"x": p.x, "y": p.y, "yaw": quaternion_to_yaw(q.x, q.y, q.z, q.w),
                         "v": t.linear.x, "w": t.angular.z}
            self.odom_rate.tick()

    def _on_battery(self, msg):
        # percentage 필드는 노드마다 0~1 / 0~100으로 달라서 쓰지 않고, 전압에서 직접 계산한다
        with self.lock:
            self.battery = {"voltage": float(msg.voltage), "percentage": battery_percent(float(msg.voltage))}

    def _on_image(self, msg):
        try:
            img = image_msg_to_bgr(msg.encoding, msg.height, msg.width, msg.step, msg.data)
        except ValueError as e:
            self.node.get_logger().warn(str(e), throttle_duration_sec=5.0)
            return
        with self.lock:
            self.image = img
            self.image_rate.tick()

    def snapshot(self):
        with self.lock:
            return self.scan_pts, self.scan_rate, self.odom, self.image, self.battery

    def close(self):
        try:
            self.node.destroy_node()
            self.rclpy.shutdown()
        except Exception:
            pass


class DatasetReplay:
    """프레임 데이터셋 재생 (WSL에서 화면 구성 확인, 녹화 데이터 다시 보기)."""

    def __init__(self, run_dir, fps, loop=False):
        from floor_dirt.dataset import FrameDataset
        self.ds = FrameDataset(run_dir)
        self.dt = 1.0 / fps
        self.loop = loop
        self.i = 0
        self.prev = None

    def next(self):
        if self.i >= len(self.ds):
            if not self.loop or len(self.ds) == 0:
                return None
            self.i, self.prev = 0, None
        f = self.ds[self.i]
        self.i += 1
        odom = {"x": f.pose[0], "y": f.pose[1], "yaw": f.pose[2], "v": 0.0, "w": 0.0}
        if self.prev is not None:
            dt = max(f.stamp - self.prev.stamp, 1e-6)
            odom["v"] = float(np.hypot(f.pose[0] - self.prev.pose[0], f.pose[1] - self.prev.pose[1]) / dt)
            odom["w"] = float((f.pose[2] - self.prev.pose[2]) / dt)
        self.prev = f
        return f, odom


# ---------------------------------------------------------------- 출력

class WindowOutput:
    def push(self, _id, img):
        cv2.imshow("live_monitor", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt

    def close(self):
        cv2.destroyAllWindows()


def local_ip():
    """노트북에서 접속할 주소 안내용. UDP 소켓을 '연결'만 하고 패킷은 보내지 않는다."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "<젯슨IP>"
    finally:
        s.close()


def load_calibration(enabled, intr_path, floor_path):
    if not enabled or not (os.path.exists(intr_path) and os.path.exists(floor_path)):
        return None, None, None
    intr = load_intrinsics(intr_path)
    floor = load_floor_calibration(floor_path)
    return intr, floor.H, undistort_maps(intr.K, intr.D, (intr.width, intr.height))


# ---------------------------------------------------------------- 메인

def main():
    params = load_params()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--camera", default="/dev/video0", help="웹캠 장치 (기본 /dev/video0)")
    src.add_argument("--topic", help="카메라 대신 ROS 이미지 토픽 구독 (예: /image_raw)")
    src.add_argument("--dataset", help="프레임 데이터셋 재생 (ROS·카메라 불필요)")
    src.add_argument("--no-camera", action="store_true", help="카메라 없이 LiDAR·odom만")
    ap.add_argument("--no-ros", action="store_true", help="/scan·/odom 구독 안 함 (카메라만 확인)")
    ap.add_argument("--loop", action="store_true", help="--dataset 재생을 끝없이 반복")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fourcc", default="YUYV")
    ap.add_argument("--output", choices=["http", "rtsp", "window", "file"], default="http",
                    help="http: 브라우저(기본), rtsp: VLC, window: 젯슨 화면, file: 이미지 저장")
    ap.add_argument("--port", type=int, default=8080, help="http 출력 포트")
    ap.add_argument("--quality", type=int, default=80, help="http 출력 JPEG 품질 (낮추면 데이터량 감소)")
    ap.add_argument("--url", default="rtsp://@:8554/monitor", help="rtsp 출력 주소 (@ = 이 기기의 모든 주소)")
    ap.add_argument("--bitrate", type=int, default=2000000, help="rtsp 출력 비트레이트 [bps]")
    ap.add_argument("--fps", type=float, default=10.0, help="화면 송출 주기 (카메라는 그대로 30 fps로 읽음)")
    ap.add_argument("--range", type=float, default=2.0, help="LiDAR 그림 반경 [m]")
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "results", "monitor"), help="file 출력 폴더")
    ap.add_argument("--every", type=int, default=10, help="file 출력: N장마다 저장")
    ap.add_argument("--no-calib", action="store_true", help="보정 파일이 있어도 ROI·LiDAR 겹치기 안 함")
    ap.add_argument("--intrinsics", default=DEFAULT_INTRINSICS_PATH)
    ap.add_argument("--floor", default=DEFAULT_FLOOR_PATH)
    args = ap.parse_args()

    roi = params["roi"]
    lo = params["exclusion"].get("lidar_origin", [0.0, 0.0])
    lidar_pose = (float(lo[0]), float(lo[1]), 0.0)
    intr, H, maps = load_calibration(not args.no_calib, args.intrinsics, args.floor)

    if args.output == "http":
        out = DebugOutput({"output": "http", "http_port": args.port, "jpeg_quality": args.quality,
                           "title": "UOS_IR_2026 live monitor"})
        print("[monitor] 노트북 브라우저에서 열기: http://{}:{}".format(local_ip(), out.port), flush=True)
    elif args.output == "rtsp":
        out = DebugOutput({"output": "rtsp", "rtsp_url": args.url,
                           "rtsp_args": ["live_monitor.py", "--bitrate={}".format(args.bitrate)]})
        port = args.url.rsplit(":", 1)[-1]
        print("[monitor] 노트북 VLC에서 열기: rtsp://{}:{}".format(local_ip(), port), flush=True)
    elif args.output == "file":
        out = DebugOutput({"output": "file", "every_n": args.every}, args.out_dir)
        print("[monitor] {}장마다 저장: {}".format(args.every, args.out_dir), flush=True)
    else:
        out = WindowOutput()

    cam = None
    replay = None
    if args.dataset:
        replay = DatasetReplay(args.dataset, args.fps, loop=args.loop)
    elif not args.no_camera and not args.topic:
        cam = DirectCamera(args.camera, args.width, args.height, args.fourcc)
    ros = None
    if not args.dataset and not args.no_ros:
        ros = RosInputs(args.topic, lidar_pose)
    if ros is not None:
        print("[monitor] ROS_DOMAIN_ID={} 에서 /scan, /odom 구독".format(os.environ.get("ROS_DOMAIN_ID", "0(기본)")),
              flush=True)
    print("[monitor] 보정 {} | 입력: {} | ROS {}".format(
        "ON (ROI·LiDAR 겹치기)" if H is not None else "없음 (카메라 원본만)",
        "dataset" if replay else ("topic " + args.topic) if args.topic else "없음" if args.no_camera else args.camera,
        "ON" if ros else "OFF"), flush=True)

    cam_rate = RateMeter()
    replay_scan_rate = RateMeter()
    jetson_rail, rails_t = None, 0.0
    period = 1.0 / args.fps
    last_send = 0.0
    last_print = time.monotonic()  # 첫 출력은 주기가 잡힌 2초 뒤
    n = 0
    try:
        while True:
            frame, ctrl, scan_pts, odom, scan_rate, battery = None, None, None, None, None, None
            if replay is not None:
                item = replay.next()
                if item is None:
                    print("[monitor] 데이터셋 끝", flush=True)
                    break
                f, odom = item
                frame, scan_pts = f.image, f.scan
                cam_rate.tick()
                if scan_pts is not None:
                    replay_scan_rate.tick()
                scan_rate = replay_scan_rate
                time.sleep(period)
            else:
                if cam is not None:
                    frame = cam.read()      # 카메라 주기(약 30 fps)로 막힘
                    if frame is not None:
                        cam_rate.tick()
                    ctrl = cam.controls()
                else:
                    time.sleep(0.01)
                if ros is not None:
                    scan_pts, scan_rate, odom, img, battery = ros.snapshot()
                    if args.topic:
                        frame = img
                        cam_rate = ros.image_rate
            now = time.monotonic()
            if replay is None and now - last_send < period:
                continue
            last_send = now

            shown = frame
            calibrated = False
            if frame is not None and H is not None and (frame.shape[1], frame.shape[0]) == (intr.width, intr.height):
                shown = undistort(frame, maps)
                calibrated = True
            if now - rails_t > 1.0:  # 젯슨 전력 센서는 1초에 한 번만 읽음
                jetson_rail, rails_t = input_rail(read_rails()), now
            power = power_line(battery, jetson_rail, show_jetson=replay is None)
            lines = status_lines(
                cam_fps=None if (cam is None and not args.topic and replay is None) else cam_rate.rate(),
                scan_hz=None if (ros is None and replay is None) else (scan_rate.rate() if scan_rate else 0.0),
                n_points=0 if scan_pts is None else len(scan_pts),
                closest=None if scan_pts is None else closest_point(scan_pts),
                odom=odom, cam_ctrl=ctrl, calibrated=calibrated,
                scan_age=None if scan_rate is None else scan_rate.age(), power=power)
            st = {
                "cam_on": cam is not None or bool(args.topic) or replay is not None,
                "cam_fps": cam_rate.rate() if frame is not None or cam is not None else None,
                "ros_on": ros is not None or replay is not None,
                "scan_hz": scan_rate.rate() if scan_rate else None,
                "scan_age": None if scan_rate is None or scan_rate.age() == float("inf") else scan_rate.age(),
                "n_points": 0 if scan_pts is None else len(scan_pts),
                "closest": None if scan_pts is None else closest_point(scan_pts),
                "odom": odom,
                "odom_age": 0.0 if replay is not None else (ros.odom_rate.age() if ros is not None else None),
                "battery": battery, "jetson": jetson_rail if replay is None else None,
                "cam_ctrl": ctrl, "calibrated": calibrated,
            }
            img = compose_dashboard(camera_panel(shown, H if calibrated else None, roi, scan_pts),
                                    lidar_topdown(scan_pts, range_m=args.range, roi=roi), st)
            out.push(n, img)
            n += 1
            if now - last_print > 2.0:
                print("[monitor] " + lines[0], flush=True)
                last_print = now
    except KeyboardInterrupt:
        pass
    finally:
        out.close()
        if cam is not None:
            cam.close()
        if ros is not None:
            ros.close()
    print("[monitor] 종료 ({}장 송출)".format(n))


if __name__ == "__main__":
    main()
