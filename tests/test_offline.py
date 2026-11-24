"""오프라인 실행(스크립트) · 평가 · 결과 웹 테스트."""

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

from floor_dirt.config import REPO_ROOT
from floor_dirt.dirt_map import Region
from floor_dirt.offline import evaluate, evaluation_markdown
from floor_dirt.web import ResultsServer, list_runs, safe_path, timing_medians


def _region(i, x, y):
    return Region(i, (x, y), (0.05, 0.05), 0.0025, (0, 0, 5, 5), 25, 0.9, 0.95, 3)


# ---------------------------------------------------------------- 평가

def test_evaluate_one_to_one_matching():
    truth = [{"name": "a", "x": 1.0, "y": 1.0}, {"name": "b", "x": 1.03, "y": 1.0}, {"name": "c", "x": 3.0, "y": 3.0}]
    regions = [_region(1, 1.01, 1.0), _region(2, 2.0, 2.0)]
    ev = evaluate(regions, truth, match_radius=0.05)
    # 구역 1은 a(1 cm)와 b(2 cm) 모두 가깝지만 한 정답에만 짝지어진다 (더 가까운 a)
    assert ev["found"] == 1 and ev["recall"] == pytest.approx(1 / 3)
    assert [it["found"] for it in ev["items"]] == [True, False, False]
    assert ev["items"][0]["region"] == 1 and ev["items"][0]["error_m"] == pytest.approx(0.01)
    assert ev["false_positives"] == 1 and ev["false_positive_regions"][0]["id"] == 2
    assert ev["error_mean_m"] == pytest.approx(0.01)
    md = evaluation_markdown(ev)
    assert "1 / 3 (33%)" in md and "| a | (1.00, 1.00) | O | 1 | 1.0 |" in md


def test_evaluate_empty_cases():
    ev = evaluate([], [{"name": "a", "x": 0, "y": 0}])
    assert ev["recall"] == 0 and ev["false_positives"] == 0 and ev["error_mean_m"] is None
    ev = evaluate([_region(1, 0, 0)], [])
    assert ev["recall"] is None and ev["false_positives"] == 1
    assert "-" in evaluation_markdown(ev)


# ---------------------------------------------------------------- 결과 웹

def _fake_run(root, name, n_regions=2):
    d = os.path.join(root, name)
    os.makedirs(os.path.join(d, "regions"))
    data = {"run": name, "n_regions": n_regions, "frames": 10, "summary": {"clean_m2": 1.0},
            "regions": [{"id": 1, "center": [1.0, 2.0], "size": [0.05, 0.05], "area_m2": 0.0025,
                         "bbox_cells": [0, 0, 5, 5], "n_cells": 25, "mean_p": 0.9, "max_p": 0.95, "best_frame": 3}],
            "photos": {"1": "regions/region_01.jpg"}}
    with open(os.path.join(d, "dirt_regions.json"), "w", encoding="utf-8") as f:
        json.dump(data, f)
    cv2.imwrite(os.path.join(d, "dirt_map.png"), np.zeros((10, 10, 3), np.uint8))
    with open(os.path.join(d, "timing.csv"), "w") as f:
        f.write("frame_id,warp_ms,compare_ms,total_ms\n0,1.0,2.0,3.0\n1,3.0,4.0,7.0\n2,2.0,3.0,5.0\n")
    return d


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def test_web_helpers(tmp_path):
    root = str(tmp_path)
    _fake_run(root, "run_a")
    os.makedirs(os.path.join(root, "not_a_run"))                # dirt_regions.json 없는 폴더는 목록에서 제외
    runs = list_runs(root)
    assert [r["name"] for r in runs] == ["run_a"] and runs[0]["n_regions"] == 2
    t = timing_medians(os.path.join(root, "run_a", "timing.csv"))
    assert t == {"warp": 2.0, "compare": 3.0, "total": 5.0, "frames": 3}
    assert safe_path(root, "run_a/dirt_map.png").endswith("dirt_map.png")
    assert safe_path(root, "../etc/passwd") is None
    assert list_runs(str(tmp_path / "없음")) == []


def test_results_server(tmp_path):
    root = str(tmp_path / "results")
    _fake_run(root, "run_a")
    (tmp_path / "secret.txt").write_text("x")
    srv = ResultsServer(root, port=0, host="127.0.0.1").start()
    base = "http://127.0.0.1:{}".format(srv.port)
    try:
        code, ctype, body = _get(base + "/")
        assert code == 200 and "text/html" in ctype and "청소 구역 탐지 결과".encode("utf-8") in body
        assert json.loads(_get(base + "/api/runs")[2])[0]["name"] == "run_a"
        summ = json.loads(_get(base + "/api/run/run_a")[2])
        assert summ["n_regions"] == 2 and summ["timing"]["total"] == 5.0 and summ["map_mtime"] is not None
        code, ctype, body = _get(base + "/files/run_a/dirt_map.png")
        assert ctype == "image/png" and body[:4] == b"\x89PNG"
        for bad in ("/api/run/nope", "/files/../secret.txt", "/files/%2e%2e/secret.txt", "/files/run_a/none.png"):
            with pytest.raises(urllib.error.HTTPError) as e:
                _get(base + bad)
            assert e.value.code == 404, bad
    finally:
        srv.close()


# ---------------------------------------------------------------- 스크립트 전체 흐름 (실제 프로세스)

def _run(args, cwd=REPO_ROOT):
    r = subprocess.run([sys.executable] + args, cwd=cwd, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_scripts_end_to_end(tmp_path):
    """합성 데이터 생성 → 기준 → 정찰 → 평가를 사용자가 치는 명령 그대로 실행."""
    data = str(tmp_path / "data")
    _run(["scripts/make_synthetic_run.py", "--out", data])
    for p in ("synthetic_ref/poses.csv", "synthetic_patrol/truth.json", "synthetic_map/classroom.yaml",
              "synthetic_calib/floor_homography.yaml"):
        assert os.path.exists(os.path.join(data, p)), p
    ref = str(tmp_path / "reference")
    out = _run(["scripts/run_reference.py", "--run", os.path.join(data, "synthetic_ref"),
                "--map", os.path.join(data, "synthetic_map", "classroom.yaml"),
                "--calib-dir", os.path.join(data, "synthetic_calib"), "--no-dnn",
                "--out", ref, "--results", str(tmp_path / "res_ref")])
    assert "기준 저장" in out and os.path.exists(os.path.join(ref, "reference.npz"))
    res = str(tmp_path / "res_patrol")
    out = _run(["scripts/run_patrol.py", "--run", os.path.join(data, "synthetic_patrol"),
                "--map", os.path.join(data, "synthetic_map", "classroom.yaml"), "--reference", ref,
                "--calib-dir", os.path.join(data, "synthetic_calib"), "--no-dnn", "--synthetic-detector",
                "--out", res, "--update", "0"])
    assert "오염 구역 3개" in out
    out = _run(["scripts/evaluate.py", "--results", res, "--truth", os.path.join(data, "synthetic_patrol", "truth.json")])
    ev = json.load(open(os.path.join(res, "evaluation.json"), encoding="utf-8"))
    assert ev["recall"] == 1.0 and ev["false_positives"] == 0 and ev["error_max_m"] <= 0.02
    assert "3 / 3 (100%)" in out
    for f in ("dirt_map.png", "dirt_regions.json", "timing.csv", "run_info.yaml", "regions/region_01.jpg"):
        assert os.path.exists(os.path.join(res, f)), f
    assert [r["name"] for r in list_runs(str(tmp_path))] == ["res_patrol"]
