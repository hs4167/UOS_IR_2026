#!/usr/bin/env python3
"""웹캠 노출·초점·화이트밸런스를 config/camera_settings.yaml 값으로 고정한다 (젯슨).

사용 예
  python scripts/camera_settings.py                       # 설정 파일 값 적용
  python scripts/camera_settings.py --show                # 현재 값만 보기
  python scripts/camera_settings.py --exposure 300 --focus 30      # 바꿔 보기 (모니터로 확인)
  python scripts/camera_settings.py --exposure 300 --focus 30 --save   # 마음에 들면 설정 파일에 저장
카메라 노드(ros_nodes/camera_node.py)도 시작할 때 같은 설정을 적용한다.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from floor_dirt.camera_settings import apply, describe, load_settings, read_controls, save_settings  # noqa: E402
from floor_dirt.config import REPO_ROOT  # noqa: E402

DEFAULT = os.path.join(REPO_ROOT, "config", "camera_settings.yaml")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT)
    ap.add_argument("--device", default=None, help="기본: 설정 파일의 device")
    ap.add_argument("--show", action="store_true", help="적용하지 않고 현재 값만 출력")
    ap.add_argument("--exposure", type=int, help="exposure_absolute (3~2047)")
    ap.add_argument("--focus", type=int, help="focus_absolute (0~250, 5 단위)")
    ap.add_argument("--wb", type=int, help="white_balance_temperature (2000~6500)")
    ap.add_argument("--gain", type=int, help="gain (0~255)")
    ap.add_argument("--save", action="store_true", help="바꾼 값을 설정 파일에 저장")
    args = ap.parse_args()

    settings = load_settings(args.config)
    device = args.device or settings.get("device", "/dev/video0")
    if args.show:
        now = read_controls(device)
        for k, c in now.items():
            print("  {:<32} {}{}".format(k, c.value, "  (inactive)" if c.inactive else ""))
        print("요약:", describe(now))
        return

    controls = dict(settings.get("controls") or {})
    for key, val in (("exposure_absolute", args.exposure), ("focus_absolute", args.focus),
                     ("white_balance_temperature", args.wb), ("gain", args.gain)):
        if val is not None:
            controls[key] = val
    result, warnings = apply(device, controls)
    print("{} 에 적용:".format(device))
    for name, want, got in result:
        print("  {:<32} {:>6}  {}".format(name, want, "OK" if want == got else "→ 읽힌 값 {}".format(got)))
    for w in warnings:
        print("경고:", w)
    print("요약:", describe(read_controls(device)))
    if args.save:
        settings["controls"] = controls
        save_settings(args.config, settings)
        print("저장: {}".format(args.config))


if __name__ == "__main__":
    main()
