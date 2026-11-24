#!/usr/bin/env python3
"""WSL 확인용 합성 주행 데이터 만들기 (실제 rosbag이 없을 때 전체 흐름 확인).

만드는 것 (기본 data/ 아래)
  synthetic_ref/       기준 주행 (청소 직후 바닥) 프레임 데이터셋
  synthetic_patrol/    정찰 주행 (오염 3개 + 가방 + 신발 + 조명 10% 밝음) 프레임 데이터셋
    truth.json         오염 정답 위치 (evaluate.py용)
  synthetic_revisit/   청소 후 재방문 (빨간 종이·회색 먼지는 치움, 초록 종이는 그대로, 조명 5% 밝음)
    truth_revisit.json 구역별 정답 처리 여부
  synthetic_map/       가짜 LiDAR 지도 classroom.yaml/pgm
  synthetic_calib/     합성 카메라 보정 파일 camera_intrinsics.yaml, floor_homography.yaml

사용 예
  python scripts/make_synthetic_run.py
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from floor_dirt.calibration import FloorCalibration, Intrinsics, save_floor_calibration, save_intrinsics  # noqa: E402
from floor_dirt.config import REPO_ROOT  # noqa: E402
from floor_dirt.dataset import Frame, FrameDatasetWriter  # noqa: E402
from floor_dirt.projection import map_to_robot  # noqa: E402
from floor_dirt.synthetic import (  # noqa: E402
    FloorTexture, _paint, add_sensor_noise, classroom_scene, item_outline, noisy, render, wide_camera,
)

MAP_ORIGIN = (-0.5, 0.0)
MAP_SIZE = (4.0, 3.0)   # m
MAP_RES = 0.05


def write_map(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    w, h = int(MAP_SIZE[0] / MAP_RES), int(MAP_SIZE[1] / MAP_RES)
    pgm = np.full((h, w), 205, np.uint8)      # 모름
    pgm[2:-2, 2:-2] = 254                     # 빈 곳
    pgm[1, 1:-1] = pgm[-2, 1:-1] = 0          # 벽
    pgm[1:-1, 1] = pgm[1:-1, -2] = 0
    cv2.imwrite(os.path.join(out_dir, "classroom.pgm"), pgm)
    with open(os.path.join(out_dir, "classroom.yaml"), "w") as f:
        f.write("image: classroom.pgm\nresolution: {}\norigin: [{}, {}, 0.0]\nnegate: 0\n"
                "occupied_thresh: 0.65\nfree_thresh: 0.196\n".format(MAP_RES, *MAP_ORIGIN))


def record(run_dir, scene, texture, rng, xy_sigma, yaw_sigma, with_objects, meta):
    cam = wide_camera()
    outline = item_outline(scene.bag) if with_objects else None
    with FrameDatasetWriter(run_dir, meta) as wr:
        for i, true_pose in enumerate(scene.poses):
            img = add_sensor_noise(render(cam, texture, true_pose), rng)
            scan = map_to_robot(true_pose, outline).astype(np.float32) if outline is not None else None
            wr.write(Frame(i, i * 0.2, noisy(true_pose, rng, xy_sigma, yaw_sigma), img, scan))
    return len(scene.poses)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "data"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scene = classroom_scene(seed=args.seed)
    rng = np.random.RandomState(args.seed + 1)
    cam = wide_camera()

    calib = os.path.join(args.out, "synthetic_calib")
    save_intrinsics(os.path.join(calib, "camera_intrinsics.yaml"),
                    Intrinsics(cam.K, np.zeros(5), cam.width, cam.height, 0.0), note="synthetic wide_camera")
    save_floor_calibration(os.path.join(calib, "floor_homography.yaml"),
                           FloorCalibration(cam.floor_homography(), cam.width, cam.height, 0.0, 0.0),
                           meta={"source": "synthetic"})
    write_map(os.path.join(args.out, "synthetic_map"))
    n1 = record(os.path.join(args.out, "synthetic_ref"), scene, scene.clean, rng, 0.003, 0.3, False,
                {"source": "synthetic", "kind": "reference", "light": "기준 조명"})
    patrol = os.path.join(args.out, "synthetic_patrol")
    n2 = record(patrol, scene, scene.dirty, rng, 0.004, 0.4, True,
                {"source": "synthetic", "kind": "patrol", "light": "조명 10% 밝음"})
    truth = {"match_radius": 0.05,
             "items": [{"name": it.name, "x": it.x, "y": it.y, "w": it.w, "h": it.h} for it in scene.dirt],
             "objects_not_dirt": [{"name": it.name, "x": it.x, "y": it.y} for it in (scene.bag, scene.shoes)]}
    with open(os.path.join(patrol, "truth.json"), "w", encoding="utf-8") as f:
        json.dump(truth, f, ensure_ascii=False, indent=2)
    # 청소 후 재방문: 초록 종이만 남김 (가방·신발도 치움), 조명 5% 밝음. 정찰 경로를 한 장 걸러 다시 찍음
    tex = scene.clean.image.astype(np.float32)
    left = [it for it in scene.dirt if it.name == "green_paper"]
    for it in left:
        _paint(tex, scene.clean.resolution, it, (scene.clean.origin_x, scene.clean.origin_y))
    revisit_tex = FloorTexture(np.clip(tex * 1.05, 0, 255).astype(np.uint8), scene.clean.origin_x,
                               scene.clean.origin_y, scene.clean.resolution)
    rv_scene = scene._replace(poses=scene.poses[::2])
    revisit = os.path.join(args.out, "synthetic_revisit")
    n3 = record(revisit, rv_scene, revisit_tex, rng, 0.004, 0.4, False,
                {"source": "synthetic", "kind": "revisit", "light": "조명 5% 밝음"})
    with open(os.path.join(revisit, "truth_revisit.json"), "w", encoding="utf-8") as f:
        json.dump({"items": [{"name": it.name, "x": it.x, "y": it.y, "cleaned": it not in left} for it in scene.dirt]},
                  f, ensure_ascii=False, indent=2)
    print("기준 {}장, 정찰 {}장, 재방문 {}장 → {}".format(n1, n2, n3, args.out))
    print("다음: python scripts/run_reference.py --run {} --map {} --calib-dir {} --no-dnn".format(
        os.path.relpath(os.path.join(args.out, "synthetic_ref")), os.path.relpath(os.path.join(args.out, "synthetic_map", "classroom.yaml")),
        os.path.relpath(calib)))


if __name__ == "__main__":
    main()
