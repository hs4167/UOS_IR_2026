"""오프라인 실행: 프레임 데이터셋 → 기준 바닥 지도 / 정찰 결과 / 평가.

scripts/run_reference.py, run_patrol.py, evaluate.py가 이 함수들을 부른다 (스크립트는 인자 처리만).
처리 자체는 pipeline.ReferenceBuilder / Patroller가 한다 (ROS 노드와 같은 코드).
"""

import datetime
import json
import math
import os
import time
from typing import List, Optional

import numpy as np
import yaml

from floor_dirt.calibration import load_floor_calibration, load_intrinsics
from floor_dirt.config import deep_merge
from floor_dirt.dataset import FrameDataset
from floor_dirt.debug_stream import DebugOutput
from floor_dirt.dirt_map import load_regions
from floor_dirt.floor_grid import FloorGrid, lidar_free_mask
from floor_dirt.pipeline import PATROL_STAGES, REFERENCE_STAGES, Patroller, ReferenceBuilder
from floor_dirt.projection import GridMeta, undistort_maps
from floor_dirt.tegralog import TegrastatsLogger
from floor_dirt.timing import StageTimer
from floor_dirt.viz import lidar_background

NO_DNN = {"exclusion": {"use_segnet": False, "use_detectnet": False}}


def load_camera(intrinsics_path, floor_path):
    """(H, 왜곡 보정 테이블 또는 None). 왜곡 계수가 모두 0이면 보정을 건너뛴다."""
    intr = load_intrinsics(intrinsics_path)
    floor = load_floor_calibration(floor_path)
    if (intr.width, intr.height) != (floor.width, floor.height):
        raise ValueError("내부 보정({}x{})과 바닥 보정({}x{}) 해상도가 다릅니다".format(
            intr.width, intr.height, floor.width, floor.height))
    maps = None if not np.any(intr.D) else undistort_maps(intr.K, intr.D, (intr.width, intr.height))
    return floor.H, maps


def prepare_params(params, no_dnn=False):
    return deep_merge(params, NO_DNN) if no_dnn else params


def _write_info(out_dir, info):
    os.makedirs(out_dir, exist_ok=True)
    info = dict(info, created=datetime.datetime.now().isoformat(timespec="seconds"))
    with open(os.path.join(out_dir, "run_info.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(info, f, allow_unicode=True, sort_keys=False)


def _progress(i, n, t0, every=20, prefix=""):
    if i % every == 0 or i == n:
        el = time.monotonic() - t0
        print("{}{}/{} 프레임 ({:.1f} fps)".format(prefix, i, n, i / el if el > 0 else 0.0), flush=True)


# ---------------------------------------------------------------- 기준 주행

def run_reference(run_dir, map_yaml, out_dir, params, H, maps=None, merge=False, results_dir=None, quiet=False,
                  excluder=None):
    # type: (str, str, str, dict, np.ndarray, Optional[tuple], bool, Optional[str], bool, object) -> dict
    """기준 주행 데이터셋 → out_dir(maps/reference)에 기준 바닥 지도. 반환: 바닥 덮인 비율 등."""
    grid = GridMeta.from_lidar_map(map_yaml, float(params["grid"]["resolution"]))
    run = os.path.basename(os.path.normpath(run_dir))
    results_dir = results_dir or os.path.join("results", run)
    os.makedirs(results_dir, exist_ok=True)
    timer = StageTimer(REFERENCE_STAGES, os.path.join(results_dir, "timing.csv"))
    rb = ReferenceBuilder(params, grid, H, undistort_maps=maps, timer=timer, excluder=excluder)
    if merge and os.path.exists(os.path.join(out_dir, "reference.npz")):
        old = FloorGrid.load(out_dir)
        if old.grid != grid:
            raise ValueError("기존 기준의 격자가 지도와 다릅니다. --merge 없이 새로 만드세요")
        rb.floor_grid.merge(old)
    ds = FrameDataset(run_dir)
    tlog = TegrastatsLogger(os.path.join(results_dir, "tegrastats.log")).start()   # 젯슨에서만 기록됨
    t0 = time.monotonic()
    try:
        for i, f in enumerate(ds, 1):
            rb.process(f)
            if not quiet:
                _progress(i, len(ds), t0, prefix="[reference] ")
    finally:
        resources = tlog.stop()
    floor = lidar_free_mask(map_yaml, grid)
    cov = rb.save(out_dir, run=run, floor_mask=floor)
    rb.close()
    _write_info(results_dir, {"kind": "reference", "run": run_dir, "map": map_yaml, "out": out_dir,
                              "frames": len(ds), "merge": bool(merge), "coverage": cov, "resources": resources,
                              "params": params})
    cov = dict(cov, resources=resources)
    return cov


# ---------------------------------------------------------------- 정찰 주행

def run_patrol(run_dir, reference_dir, out_dir, params, H, maps=None, map_yaml=None, debug_output="none",
               update_interval=None, quiet=False, excluder=None):
    # type: (str, str, str, dict, np.ndarray, Optional[tuple], Optional[str], str, Optional[float], bool, object) -> list
    """정찰 주행 데이터셋 → out_dir(results/<run>)에 결과. update_interval초마다 지도·json 중간 저장(웹 확인용)."""
    reference = FloorGrid.load(reference_dir)
    os.makedirs(out_dir, exist_ok=True)
    background = lidar_background(map_yaml, reference.grid) if map_yaml else None
    timer = StageTimer(PATROL_STAGES, os.path.join(out_dir, "timing.csv"))
    dbg_cfg = dict(params["debug"], output=debug_output)
    debug = DebugOutput(dbg_cfg, os.path.join(out_dir, "debug")) if debug_output != "none" else None
    pt = Patroller(params, reference, H, undistort_maps=maps, timer=timer, debug=debug, excluder=excluder)
    ds = FrameDataset(run_dir)
    run = os.path.basename(os.path.normpath(run_dir))
    interval = float(params["viz"]["update_interval_s"] if update_interval is None else update_interval)
    last = time.monotonic()
    t0 = last
    tlog = TegrastatsLogger(os.path.join(out_dir, "tegrastats.log")).start()   # 젯슨에서만 기록됨
    try:
        for i, f in enumerate(ds, 1):
            pt.process(f)
            if not quiet:
                _progress(i, len(ds), t0, prefix="[patrol] ")
            if interval > 0 and time.monotonic() - last >= interval:
                pt.export(out_dir, run=run, background=background)   # 중간 저장 (사진 없이 빠르게)
                last = time.monotonic()
    finally:
        resources = tlog.stop()
    regions = pt.export(out_dir, run=run, background=background, get_frame=ds.__getitem__)
    pt.close()
    _write_info(out_dir, {"kind": "patrol", "run": run_dir, "reference": reference_dir, "map": map_yaml,
                          "frames": len(ds), "regions": len(regions), "resources": resources, "params": params})
    return regions


# ---------------------------------------------------------------- 평가

def load_truth(path):
    """truth.json: {"items": [{"name", "x", "y", ...}], "match_radius": 0.05(선택)}."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(regions, truth_items, match_radius=0.05):
    # type: (list, List[dict], float) -> dict
    """정답 위치와 오염 구역을 가까운 순서로 짝짓는다 (한 구역은 한 정답에만).

    재현율 = 찾은 정답 수 / 정답 수, 오탐 = 짝이 없는 구역 수, 위치 오차 = 짝지은 거리 [m].
    """
    pairs = []
    for ti, t in enumerate(truth_items):
        for ri, r in enumerate(regions):
            d = math.hypot(r.center[0] - float(t["x"]), r.center[1] - float(t["y"]))
            if d <= match_radius:
                pairs.append((d, ti, ri))
    pairs.sort()
    used_t, used_r, matches = set(), set(), []
    for d, ti, ri in pairs:
        if ti in used_t or ri in used_r:
            continue
        used_t.add(ti)
        used_r.add(ri)
        matches.append((ti, ri, d))
    errs = [d for _, _, d in matches]
    items = []
    for ti, t in enumerate(truth_items):
        m = [x for x in matches if x[0] == ti]
        items.append({"name": t.get("name", "item{}".format(ti)), "x": t["x"], "y": t["y"],
                      "found": bool(m), "region": regions[m[0][1]].id if m else None,
                      "error_m": round(m[0][2], 4) if m else None})
    fps = [r for ri, r in enumerate(regions) if ri not in used_r]
    n = len(truth_items)
    return {
        "match_radius_m": match_radius,
        "truth": n, "regions": len(regions), "found": len(matches),
        "recall": (len(matches) / n) if n else None,
        "false_positives": len(fps),
        "false_positive_regions": [{"id": r.id, "center": list(r.center), "area_m2": r.area_m2} for r in fps],
        "error_mean_m": float(np.mean(errs)) if errs else None,
        "error_max_m": float(np.max(errs)) if errs else None,
        "items": items,
    }


def evaluate_results(results_dir, truth_path, match_radius=None):
    truth = load_truth(truth_path)
    radius = float(match_radius if match_radius is not None else truth.get("match_radius", 0.05))
    regions = load_regions(os.path.join(results_dir, "dirt_regions.json"))
    ev = evaluate(regions, truth["items"], radius)
    ev["results"] = results_dir
    ev["truth_file"] = truth_path
    with open(os.path.join(results_dir, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(ev, f, ensure_ascii=False, indent=2)
    return ev


def evaluation_markdown(ev):
    """보고서에 붙일 표."""
    def cm(v):
        return "-" if v is None else "{:.1f}".format(v * 100)
    lines = ["| 항목 | 값 |", "| --- | --- |",
             "| 정답 수 | {} |".format(ev["truth"]),
             "| 검출 (재현율) | {} / {} ({}) |".format(ev["found"], ev["truth"],
                                                    "-" if ev["recall"] is None else "{:.0%}".format(ev["recall"])),
             "| 오탐 구역 | {} |".format(ev["false_positives"]),
             "| 위치 오차 평균 / 최대 [cm] | {} / {} |".format(cm(ev["error_mean_m"]), cm(ev["error_max_m"])),
             "", "| 정답 | 위치 (x, y) [m] | 검출 | 구역 | 오차 [cm] |", "| --- | --- | --- | --- | --- |"]
    for it in ev["items"]:
        lines.append("| {} | ({:.2f}, {:.2f}) | {} | {} | {} |".format(
            it["name"], it["x"], it["y"], "O" if it["found"] else "X",
            "-" if it["region"] is None else it["region"], cm(it["error_m"])))
    return "\n".join(lines)


# ---------------------------------------------------------------- [4] 재방문

def load_patrol_regions(results_dir, params):
    """정찰 결과 폴더 → (구역 목록, 칸별 구역 번호). dirt_map.npz에서 다시 묶어 json과 같은지 확인한다."""
    from floor_dirt.dirt_map import DirtMap
    dm = DirtMap.load(results_dir, params["dirt_map"])
    regions, labels = dm.extract_regions()
    saved = load_regions(os.path.join(results_dir, "dirt_regions.json"))
    same = len(saved) == len(regions) and all(
        a.id == b.id and math.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1]) < 1e-3
        for a, b in zip(saved, regions))
    if not same:
        raise ValueError("dirt_regions.json과 dirt_map.npz에서 다시 묶은 구역이 다릅니다 "
                         "(정찰 후 params의 dirt_map 설정이 바뀌었는지 확인)")
    return regions, labels


def plan_from_results(results_dir, map_yaml, start_pose, params):
    """정찰 결과 + LiDAR 지도 → 재방문 계획 (방문 순서, 설 자리). revisit_plan.json·png 저장."""
    from floor_dirt.revisit import free_space_checker, plan_revisit
    from floor_dirt.viz import draw_revisit_plan, render_dirt_map, save_png_atomic
    from floor_dirt.dirt_map import DirtMap
    regions, _ = load_patrol_regions(results_dir, params)
    rv = params["revisit"]
    is_free = free_space_checker(map_yaml, float(rv["clearance"])) if map_yaml else None
    plan = plan_revisit(regions, start_pose, float(rv["standoff"]), is_free)
    with open(os.path.join(results_dir, "revisit_plan.json"), "w", encoding="utf-8") as f:
        json.dump({"start": list(start_pose), "map": map_yaml, "plan": plan}, f, ensure_ascii=False, indent=2)
    dm = DirtMap.load(results_dir, params["dirt_map"])
    bg = lidar_background(map_yaml, dm.grid) if map_yaml else None
    scale = 2
    img = render_dirt_map(dm, regions, background=bg, scale=scale, legend=False)
    draw_revisit_plan(img, dm.grid, plan, start_pose, scale)
    save_png_atomic(os.path.join(results_dir, "revisit_plan.png"), img)
    return plan


def run_revisit(run_dir, results_dir, reference_dir, params, H, maps=None, out_name="revisit", promote=False,
                excluder=None, quiet=False):
    """재방문 때 녹화한 데이터셋 → 구역별 복원율 판정 (results/<정찰>/revisit.json). promote면 기준 갱신."""
    from floor_dirt.pipeline import REVISIT_STAGES, RevisitProcessor
    from floor_dirt.revisit import CLEANED, promote_reference, write_report
    regions, labels = load_patrol_regions(results_dir, params)
    reference = FloorGrid.load(reference_dir)
    if reference.grid != patrol_grid(results_dir):
        raise ValueError("기준 격자와 정찰 결과 격자가 다릅니다")
    timer = StageTimer(REVISIT_STAGES, os.path.join(results_dir, out_name + "_timing.csv"))
    rp = RevisitProcessor(params, reference, H, labels, undistort_maps=maps, timer=timer, excluder=excluder)
    ds = FrameDataset(run_dir)
    t0 = time.monotonic()
    for i, f in enumerate(ds, 1):
        rp.process(f)
        if not quiet:
            _progress(i, len(ds), t0, prefix="[revisit] ")
    rv = params["revisit"]
    results = rp.verifier.results([r.id for r in regions], restore_ratio=float(rv["restore_ratio"]),
                                  min_coverage=float(rv["min_coverage"]), hit_frac=float(rv["hit_frac"]))
    for res, r in zip(results, regions):
        res["center"] = [round(float(v), 3) for v in r.center]
    rp.close()
    extra = {"run": run_dir, "patrol": results_dir, "reference": reference_dir, "frames": len(ds)}
    if promote:
        cleaned = [r["id"] for r in results if r["status"] == CLEANED]
        extra["promote"] = promote_reference(reference_dir, rp.new_floor, labels, cleaned,
                                             min_count=int(params["reference"]["min_count"]),
                                             note="revisit {}".format(os.path.basename(os.path.normpath(run_dir))))
    plan_path = os.path.join(results_dir, "revisit_plan.json")
    plan = json.load(open(plan_path, encoding="utf-8"))["plan"] if os.path.exists(plan_path) else []
    return write_report(os.path.join(results_dir, out_name + ".json"), plan, results, extra)


def patrol_grid(results_dir):
    """정찰 결과의 격자 정의 (dirt_map.npz)."""
    with np.load(os.path.join(results_dir, "dirt_map.npz")) as d:
        g = d["grid"]
    return GridMeta(float(g[0]), float(g[1]), float(g[2]), int(g[3]), int(g[4]))
