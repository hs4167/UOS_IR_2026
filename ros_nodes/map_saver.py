#!/usr/bin/env python3
"""/map(SLAM 중인 지도)을 map_server 형식(pgm + yaml)으로 저장한다. nav2 map_saver 없이 동작.

실행: ./scripts/save_map.sh [이름]   → maps/<이름>.pgm, maps/<이름>.yaml (기본 classroom)
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from floor_dirt.config import REPO_ROOT  # noqa: E402
from floor_dirt.occupancy import map_stats, save_map  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "maps", "classroom"), help="확장자 없는 경로")
    ap.add_argument("--timeout", type=float, default=10.0)
    args = ap.parse_args()

    import rclpy
    from nav_msgs.msg import OccupancyGrid
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

    rclpy.init()
    node = rclpy.create_node("map_saver_py")
    got = {}
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(OccupancyGrid, "/map", lambda m: got.setdefault("map", m), latched)
    t0 = time.monotonic()
    while "map" not in got and time.monotonic() - t0 < args.timeout:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()
    if "map" not in got:
        raise SystemExit("{}초 동안 /map을 받지 못했습니다. SLAM(./scripts/slam.sh)이 켜져 있는지 확인하세요".format(args.timeout))
    m = got["map"]
    info = m.info
    o = info.origin.position
    if os.path.exists(args.out + ".yaml"):
        backup = args.out + "_backup_" + time.strftime("%Y%m%d_%H%M%S")
        for ext in (".yaml", ".pgm"):
            if os.path.exists(args.out + ext):
                os.replace(args.out + ext, backup + ext)
        print("기존 지도를 백업: {}.*".format(backup))
    pgm, yml = save_map(args.out, m.data, info.width, info.height, info.resolution, (o.x, o.y, 0.0))
    print("저장: {} , {}".format(pgm, yml))
    print("크기 {:.1f} x {:.1f} m, 해상도 {} m | {}".format(
        info.width * info.resolution, info.height * info.resolution, info.resolution, map_stats(m.data, info.resolution)))


if __name__ == "__main__":
    main()
