#!/usr/bin/env python3
"""USB 웹캠(C920)으로 보정용 이미지를 찍어 저장한다. 화면 없이(SSH) 쓸 수 있다.

시작할 때 config/camera_settings.yaml의 고정 초점·노출을 적용한다 (초점이 바뀌면 보정이 무의미).

사용 예 (젯슨, conda test1001)
  # 내부 보정용: 보드를 화면 구석구석, 여러 각도로 움직이며 2초마다 20장 (보드가 보일 때만 저장)
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

from floor_dirt.calibration import find_chessboard  # noqa: E402
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

    saved = 0
    last = 0.0
    start_index = len([f for f in os.listdir(args.out) if f.endswith(".png")])
    try:
        while saved < args.count:
            ok, frame = cap.read()
            if not ok:
                print("[capture] 프레임 읽기 실패, 재시도")
                time.sleep(0.1)
                continue
            if args.manual:
                if not enter_pressed():
                    continue
            elif time.monotonic() - last < args.interval:
                continue
            if pattern is not None and find_chessboard(frame, pattern) is None:
                print("[capture] 체커보드 안 보임, 건너뜀")
                last = time.monotonic()
                continue
            path = os.path.join(args.out, "{:03d}.png".format(start_index + saved))
            cv2.imwrite(path, frame)
            saved += 1
            last = time.monotonic()
            print("[capture] {}/{} 저장: {}".format(saved, args.count, path), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
    print("[capture] 완료: {}장".format(saved))


if __name__ == "__main__":
    main()
