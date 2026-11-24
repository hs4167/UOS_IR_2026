#!/usr/bin/env python3
"""[4] 재방문 노드: 계획의 설 자리로 Nav2 이동 → 멈춤 → n장 촬영 → 구역별 복원율 판정. 젯슨 전용.

필요한 것 (각각 다른 터미널):
  ./scripts/robot.sh                 bringup
  ./scripts/nav.sh                   Nav2 (지도 + 위치 추정 + 경로 계획). localize.sh와 동시에 켜지 않는다
  ./scripts/camera.sh                카메라 노드 (/image_raw)
  python scripts/plan_revisit.py ... 재방문 계획 (results/<정찰>/revisit_plan.json). 없으면 이 노드가 만든다
실행:
  ./scripts/revisit.sh --results results/<정찰 run>
  ./scripts/revisit.sh --results results/<정찰 run> --record 2026-11-26_revisit --promote
결과: results/<정찰>/revisit.json (결과 웹에 표시), --record면 data/<run>/에 사진도 저장 (run_revisit.py로 다시 판정 가능)
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from floor_dirt.config import REPO_ROOT, load_params  # noqa: E402
from floor_dirt.dataset import Frame, FrameDatasetWriter  # noqa: E402
from floor_dirt.floor_grid import FloorGrid  # noqa: E402
from floor_dirt.masks import scan_to_points  # noqa: E402
from floor_dirt.monitor import image_msg_to_bgr  # noqa: E402
from floor_dirt.offline import load_camera, load_patrol_regions, plan_from_results, prepare_params  # noqa: E402
from floor_dirt.pipeline import RevisitProcessor  # noqa: E402
from floor_dirt.recording import PendingFrames, ScanBuffer, quat_to_yaw, transform_to_pose  # noqa: E402
from floor_dirt.revisit import CLEANED, Mission, promote_reference, write_report  # noqa: E402
from floor_dirt.tegralog import TegrastatsLogger, format_summary  # noqa: E402


def stamp_sec(st):
    return st.sec + st.nanosec * 1e-9


def main():
    params = load_params()
    rv = params["revisit"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="정찰 결과 폴더")
    ap.add_argument("--reference", default=os.path.join(REPO_ROOT, "maps", "reference"))
    ap.add_argument("--map", default=os.path.join(REPO_ROOT, "maps", "classroom.yaml"), help="계획이 없을 때 설 자리 계산용")
    ap.add_argument("--calib-dir", default=os.path.join(REPO_ROOT, "config"))
    ap.add_argument("--n-shots", type=int, default=int(rv["n_shots"]))
    ap.add_argument("--settle", type=float, default=1.0, help="멈춘 뒤 촬영 전 기다리는 시간 [s]")
    ap.add_argument("--shoot-timeout", type=float, default=15.0, help="구역당 촬영 최대 시간 [s]")
    ap.add_argument("--nav-timeout", type=float, default=120.0, help="구역당 이동 최대 시간 [s]")
    ap.add_argument("--record", default=None, help="재방문 사진을 data/<이름>/에도 저장")
    ap.add_argument("--promote", action="store_true", help="처리 완료 구역을 새 기준으로 (기존 기준 백업)")
    ap.add_argument("--no-dnn", action="store_true")
    ap.add_argument("--slop", type=float, default=0.05)
    args = ap.parse_args()

    import rclpy
    import tf2_ros
    from action_msgs.msg import GoalStatus
    from geometry_msgs.msg import PoseStamped
    from nav2_msgs.action import NavigateToPose
    from rclpy.action import ActionClient
    from rclpy.duration import Duration
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import Image, LaserScan

    params = prepare_params(params, args.no_dnn)
    H, maps = load_camera(os.path.join(args.calib_dir, "camera_intrinsics.yaml"),
                          os.path.join(args.calib_dir, "floor_homography.yaml"))
    regions, labels = load_patrol_regions(args.results, params)
    reference = FloorGrid.load(args.reference)

    rclpy.init()
    node = rclpy.create_node("revisit_node")
    log = node.get_logger()
    buf = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
    tf2_ros.TransformListener(buf, node)

    def now_s():
        return node.get_clock().now().nanoseconds * 1e-9

    def lookup(target, source, t=None):
        try:
            tf = buf.lookup_transform(target, source, t if t is not None else Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return None
        tr, q = tf.transform.translation, tf.transform.rotation
        return transform_to_pose(tr.x, tr.y, q.x, q.y, q.z, q.w)

    # 계획: 파일이 있으면 그대로, 없으면 지금 위치에서 출발하는 계획을 만든다
    plan_path = os.path.join(args.results, "revisit_plan.json")
    if os.path.exists(plan_path):
        import json
        plan = json.load(open(plan_path, encoding="utf-8"))["plan"]
    else:
        t0 = time.monotonic()
        start = None
        while start is None and time.monotonic() - t0 < 10.0:
            rclpy.spin_once(node, timeout_sec=0.2)
            start = lookup("map", "base_footprint")
        if start is None:
            raise SystemExit("지도 위 로봇 위치(map → base_footprint)를 알 수 없습니다. nav.sh가 켜져 있는지 확인")
        plan = plan_from_results(args.results, args.map, start, params)
    print("[revisit] 구역 {}개: {}".format(len(plan), " → ".join(str(p["id"]) for p in plan)), flush=True)

    rp = RevisitProcessor(params, reference, H, labels, undistort_maps=maps)
    mission = Mission(plan, n_shots=args.n_shots, settle_s=args.settle, shoot_timeout_s=args.shoot_timeout)
    writer = FrameDatasetWriter(os.path.join(REPO_ROOT, "data", args.record),
                                {"source": "revisit_node", "pose_frame": "map", "patrol": args.results}) \
        if args.record else None
    tlog = TegrastatsLogger(os.path.join(args.results, "revisit_tegrastats.log")).start()
    pending = PendingFrames(max_wait_s=1.0)
    scans = ScanBuffer()
    st = {"lidar": None, "goal_handle": None, "nav_t0": None, "n_saved": 0}

    nav = ActionClient(node, NavigateToPose, "navigate_to_pose")
    print("[revisit] Nav2 기다리는 중...", flush=True)
    if not nav.wait_for_server(timeout_sec=30.0):
        raise SystemExit("navigate_to_pose 액션이 없습니다. ./scripts/nav.sh 가 켜져 있는지 확인")

    def on_scan(msg):
        if st["lidar"] is None:
            st["lidar"] = lookup("base_footprint", msg.header.frame_id or "base_scan")
            if st["lidar"] is None:
                return
        pts = scan_to_points(msg.ranges, msg.angle_min, msg.angle_increment, msg.range_min, msg.range_max,
                             sensor_pose=st["lidar"])
        scans.add(stamp_sec(msg.header.stamp), pts.astype("float32"))

    def on_image(msg):
        t = stamp_sec(msg.header.stamp)
        if not mission.want_frame(t):
            return
        try:
            img = image_msg_to_bgr(msg.encoding, msg.height, msg.width, msg.step, msg.data)
        except ValueError:
            return
        pending.add(t, (img, Time.from_msg(msg.header.stamp)), time.monotonic())

    def send_goal(goal):
        x, y, yaw = goal
        ps = PoseStamped()
        ps.header.frame_id = "map"
        ps.header.stamp = node.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y = float(x), float(y)
        ps.pose.orientation.z, ps.pose.orientation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        g = NavigateToPose.Goal()
        g.pose = ps
        st["nav_t0"] = time.monotonic()
        print("[revisit] 구역 {} 로 이동: ({:+.2f}, {:+.2f}, {:+.0f}°)".format(
            mission.current()["id"], x, y, math.degrees(yaw)), flush=True)
        fut = nav.send_goal_async(g)
        fut.add_done_callback(on_goal_response)

    def on_goal_response(fut):
        gh = fut.result()
        if not gh.accepted:
            mission.nav_done(False, now_s(), "goal rejected")
            return
        st["goal_handle"] = gh
        gh.get_result_async().add_done_callback(on_nav_result)

    def on_nav_result(fut):
        status = fut.result().status
        st["goal_handle"] = None
        ok = status == GoalStatus.STATUS_SUCCEEDED
        mission.nav_done(ok, now_s(), "" if ok else "status {}".format(status))
        print("[revisit]   이동 {}".format("완료 → 촬영" if ok else "실패 (status {})".format(status)), flush=True)

    def on_timer():
        now = now_s()
        if mission.state == Mission.IDLE:
            goal = mission.next_goal(now)
            if goal is not None:
                send_goal(goal)
        elif mission.state == Mission.NAVIGATING and st["goal_handle"] is not None and \
                time.monotonic() - st["nav_t0"] > args.nav_timeout:
            st["goal_handle"].cancel_goal_async()
            st["goal_handle"] = None
            mission.nav_done(False, now, "timeout")
        mission.tick(now)
        for p in pending.ready(lambda q: buf.can_transform("map", "base_footprint", q.image[1]), time.monotonic()):
            if mission.state != Mission.SHOOTING:
                continue
            img, ros_t = p.image
            pose = lookup("map", "base_footprint", ros_t)
            if pose is None:
                continue
            frame = Frame(st["n_saved"], p.stamp, pose, img, scans.nearest(p.stamp, args.slop))
            ids = rp.process(frame)
            if writer is not None:
                writer.write(frame)
            st["n_saved"] += 1
            cur = mission.current()
            mission.frame_done(ids)
            if cur is not None and mission.current() is not cur:
                print("[revisit]   구역 {} 촬영 끝".format(cur["id"]), flush=True)
        if mission.state == Mission.DONE:
            raise SystemExit(0)

    node.create_subscription(Image, "/image_raw", on_image, qos_profile_sensor_data)
    node.create_subscription(LaserScan, "/scan", on_scan, qos_profile_sensor_data)
    node.create_timer(0.1, on_timer)
    interrupted = False
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        interrupted = True
        if st["goal_handle"] is not None:
            st["goal_handle"].cancel_goal_async()
    finally:
        if writer is not None:
            writer.close()
        res_t = tlog.stop()
        results = rp.verifier.results([r.id for r in regions], restore_ratio=float(rv["restore_ratio"]),
                                      min_coverage=float(rv["min_coverage"]), hit_frac=float(rv["hit_frac"]))
        nav_log = {e["id"]: e for e in mission.log}
        for res, r in zip(results, regions):
            res["center"] = [round(float(v), 3) for v in r.center]
            res["nav"] = nav_log.get(r.id)
        extra = {"source": "revisit_node", "patrol": args.results, "reference": args.reference,
                 "frames": st["n_saved"], "interrupted": interrupted, "resources": res_t,
                 "recorded": os.path.join("data", args.record) if args.record else None}
        if args.promote and not interrupted:
            cleaned = [r["id"] for r in results if r["status"] == CLEANED]
            extra["promote"] = promote_reference(args.reference, rp.new_floor, labels, cleaned,
                                                 min_count=int(params["reference"]["min_count"]), note="revisit_node")
        rep = write_report(os.path.join(args.results, "revisit.json"), plan, results, extra)
        rp.close()
        for r in rep["regions"]:
            print("[revisit] 구역 {:>2}: {} (복원율 {}, 본 비율 {:.0%}, 사진 {})".format(
                r["id"], r["status_ko"], "-" if r["restoration_ratio"] is None else "{:.0%}".format(r["restoration_ratio"]),
                r["coverage"], r["frames"]), flush=True)
        print("[revisit] " + format_summary(res_t), flush=True)
        print("[revisit] 저장: {}".format(os.path.join(args.results, "revisit.json")), flush=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
