#!/usr/bin/env python3
"""USB 웹캠(C920)으로 보정용 이미지를 찍어 저장한다. 화면 없이(SSH) 쓸 수 있다.

시작할 때 config/camera_settings.yaml의 고정 초점·노출을 적용한다 (초점이 바뀌면 보정이 무의미).

사용 예 (젯슨, conda test1001)
  # 내부 보정용: 브라우저 미리보기(http://<젯슨IP>:8080)를 보며 보드를 화면 구석구석·여러 각도로, 2초마다 20장
  python scripts/capture_images.py --out data/calib_intrinsics --count 20 --interval 2 --pattern 9x6
  # 바닥 보정용: Enter를 칠 때마다 1장
  python scripts/capture_images.py --out data/calib_floor --manual
"""

import argparse
import os
import select
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

import numpy as np  # noqa: E402

from floor_dirt.calibration import capture_preview, corner_coverage, find_chessboard  # noqa: E402
from floor_dirt.camera_settings import apply_config  # noqa: E402
from floor_dirt.config import REPO_ROOT  # noqa: E402


def parse_pattern(text):
    cols, rows = text.lower().split("x")
    return int(cols), int(rows)


def open_camera(device, width, height, fourcc):
    dev = int(device) if device.isdigit() else device
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit("카메라를 열 수 없습니다: {} (v4l2-ctl --list-devices로 확인)".format(device))
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def enter_pressed():
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if ready:
        sys.stdin.readline()
        return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fourcc", default="YUYV", help="YUYV(무압축, 기본) 또는 MJPG")
    ap.add_argument("--out", required=True, help="저장 폴더")
    ap.add_argument("--count", type=int, default=20, help="저장할 장수")
    ap.add_argument("--interval", type=float, default=2.0, help="자동 촬영 간격 [s]")
    ap.add_argument("--manual", action="store_true", help="Enter를 칠 때마다 1장 저장")
    ap.add_argument("--pattern", default=None, help="예: 9x6. 주면 체커보드가 보이는 프레임만 저장")
    ap.add_argument("--warmup", type=int, default=30, help="시작 시 버릴 프레임 수 (노출 안정화)")
    ap.add_argument("--no-settings", action="store_true", help="웹캠 고정값을 적용하지 않음 (기본은 적용)")
    ap.add_argument("--view", choices=["http", "none"], default="http",
                    help="미리보기: http = 브라우저 http://<젯슨IP>:<port> (모니터 대신), none = 끔")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pattern = parse_pattern(args.pattern) if args.pattern else None
    cap = open_camera(args.device, args.width, args.height, args.fourcc)
    if not args.no_settings:   # 보정 사진도 기준·정찰과 같은 초점·노출로 찍어야 한다
        apply_config(os.path.join(REPO_ROOT, "config", "camera_settings.yaml"), args.device)
    for _ in range(args.warmup):
        cap.read()

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("프레임을 읽지 못했습니다")
    h, w = frame.shape[:2]
    print("[capture] {} {}x{} → {}".format(args.device, w, h, args.out))
    if (w, h) != (args.width, args.height):
        print("[capture] 경고: 요청한 해상도 {}x{}와 다릅니다".format(args.width, args.height))
    print("[capture] {} (종료: Ctrl+C)".format("Enter = 1장 저장" if args.manual else "{}초마다 자동 저장".format(args.interval)))

    server = None
    if args.view == "http":
        from floor_dirt.mjpeg_server import MjpegServer
        server = MjpegServer(port=args.port, title="calibration capture")
        print("[capture] 미리보기: http://{}:{} (모니터는 끄고 볼 것)".format(local_ip(), server.port), flush=True)

    saved = 0
    last = 0.0
    saved_corners = []
    preview_corners, n = None, 0
    msg = ""
    start_index = len([f for f in os.listdir(args.out) if f.endswith(".png")])
    try:
        while saved < args.count:
            ok, frame = cap.read()
            if not ok:
                print("[capture] 프레임 읽기 실패, 재시도")
                time.sleep(0.1)
                continue
            n += 1
            if server is not None:
                if pattern is not None and n % 3 == 0:      # 미리보기 검출은 3장에 1번 (빠른 검사)
                    preview_corners = find_chessboard(frame, pattern, fast=True)
                cov = corner_coverage(saved_corners, (frame.shape[1], frame.shape[0])) if pattern is not None else None
                server.push(capture_preview(frame, preview_corners, pattern, saved, args.count, cov, msg))
            if args.manual:
                if not enter_pressed():
                    continue
            elif time.monotonic() - last < args.interval:
                continue
            corners = find_chessboard(frame, pattern) if pattern is not None else None
            if pattern is not None and corners is None:
                print("[capture] 체커보드 안 보임, 건너뜀")
                msg = "skipped: board not found"
                last = time.monotonic()
                continue
            path = os.path.join(args.out, "{:03d}.png".format(start_index + saved))
            cv2.imwrite(path, frame)
            if corners is not None:
                saved_corners.append(corners)
            saved += 1
            last = time.monotonic()
            msg = "saved {:03d}.png".format(start_index + saved - 1)
            print("[capture] {}/{} 저장: {}".format(saved, args.count, path), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if server is not None:
            server.close()
    print("[capture] 완료: {}장".format(saved))
    if saved_corners:
        cov = corner_coverage(saved_corners, (frame.shape[1], frame.shape[0]))
        print("[capture] 화면 3x3 구역별 코너 수:\n{}".format(cov))
        if (cov == 0).any():
            print("[capture] 경고: 코너가 없는 구역이 있습니다. 그 구석에 보드를 대고 몇 장 더 찍으세요 (같은 --out에 이어서 저장됨)")


def local_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "<젯슨IP>"
    finally:
        s.close()


if __name__ == "__main__":
    main()
