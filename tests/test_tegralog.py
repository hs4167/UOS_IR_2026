"""tegrastats 기록 테스트: 줄 해석(젯슨 실제 형식), 요약, 가짜 tegrastats로 기록, 웹 요약."""

import os
import time

import pytest

from floor_dirt.tegralog import TegrastatsLogger, format_summary, parse_line, summarize_file, summarize_lines

# 젯슨(PE1000N, L4T r35)에서 실제로 나온 형식 (2026-11-19)
LINE = ("11-19-2026 17:32:35 RAM 1193/6833MB (lfb 1136x4MB) SWAP 0/11608MB (cached 0MB) "
        "CPU [26%@1420,20%@1420,17%@1420,off,off,off] EMC_FREQ 0% GR3D_FREQ 12% "
        "AO@34.5C CV0@-256C GPU@35.5C iwlwifi@47C PMIC@50C")


def test_parse_real_line():
    r = parse_line(LINE)
    assert r["ram_mb"] == 1193 and r["ram_total_mb"] == 6833
    assert r["cpu_pct"] == pytest.approx((26 + 20 + 17) / 3)          # 꺼진 코어(off)는 평균에서 제외
    assert r["gpu_pct"] == 12
    assert r["temps"]["GPU"] == 35.5 and r["temps"]["PMIC"] == 50.0
    assert "CV0" not in r["temps"]                                    # -256C = 꺼진 센서
    assert parse_line("not a tegrastats line") is None


def test_summary_and_format():
    lines = [LINE, LINE.replace("RAM 1193", "RAM 2500").replace("GR3D_FREQ 12%", "GR3D_FREQ 88%")
             .replace("GPU@35.5C", "GPU@61.0C"), "garbage"]
    s = summarize_lines(lines)
    assert s["samples"] == 2 and s["ram_max_mb"] == 2500
    assert s["gpu_mean_pct"] == 50.0 and s["gpu_max_pct"] == 88
    assert s["temp_max_c"]["GPU"] == 61.0
    text = format_summary(s)
    assert "메모리 최대 2500/6833 MB" in text and "GPU@" not in text and "61.0" in text
    assert summarize_lines(["x"]) is None and format_summary(None) == "tegrastats 기록 없음"


def _fake_tegrastats(tmp_path):
    d = tmp_path / "bin"
    d.mkdir()
    f = d / "tegrastats"
    f.write_text("#!/bin/sh\nwhile true; do echo '{}'; sleep 0.05; done\n".format(LINE))
    f.chmod(0o755)
    return str(d)


def test_logger_records_with_fake_tegrastats(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", _fake_tegrastats(tmp_path) + os.pathsep + os.environ["PATH"])
    path = str(tmp_path / "tegrastats.log")
    lg = TegrastatsLogger(path).start()
    assert lg.active
    time.sleep(0.4)
    s = lg.stop()
    assert not lg.active and s["samples"] >= 3 and s["ram_max_mb"] == 1193
    assert summarize_file(path)["samples"] == s["samples"]


def test_logger_is_noop_without_tegrastats(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))                        # tegrastats 없음 (WSL)
    lg = TegrastatsLogger(str(tmp_path / "t.log")).start()
    assert not lg.active and lg.stop() is None
    assert not os.path.exists(str(tmp_path / "t.log"))


def test_patrol_run_writes_tegrastats_and_web_shows_it(tmp_path, monkeypatch):
    """오프라인 정찰 중 tegrastats 기록 → run_info와 결과 웹 요약에 들어감."""
    import numpy as np
    from floor_dirt.config import load_params
    from floor_dirt.dataset import Frame, FrameDatasetWriter
    from floor_dirt.floor_grid import FloorGrid
    from floor_dirt.offline import NO_DNN, run_patrol
    from floor_dirt.config import deep_merge
    from floor_dirt.projection import GridMeta
    from floor_dirt.synthetic import FloorTexture, render, wide_camera
    from floor_dirt.web import run_summary

    monkeypatch.setenv("PATH", _fake_tegrastats(tmp_path) + os.pathsep + os.environ["PATH"])
    params = deep_merge(load_params(), NO_DNN)
    grid = GridMeta(0.01, 0.0, 0.0, 200, 200)
    cam = wide_camera()
    tex = FloorTexture(np.full((1000, 1000, 3), 150, np.uint8), 0, 0, 0.002)
    FloorGrid(grid, 1).save(str(tmp_path / "ref"))
    with FrameDatasetWriter(str(tmp_path / "run")) as wr:
        for i in range(8):
            p = (0.5 + 0.05 * i, 1.0, 0.0)
            wr.write(Frame(i, i * 0.1, p, render(cam, tex, p), None))
    out = str(tmp_path / "results" / "run")
    run_patrol(str(tmp_path / "run"), str(tmp_path / "ref"), out, params, cam.floor_homography(), quiet=True,
               update_interval=0)
    assert os.path.exists(os.path.join(out, "tegrastats.log"))
    import yaml
    info = yaml.safe_load(open(os.path.join(out, "run_info.yaml"), encoding="utf-8"))
    assert info["resources"] is None or info["resources"]["ram_total_mb"] == 6833   # 처리가 매우 빠르면 0줄일 수 있음
    with open(os.path.join(out, "tegrastats.log"), "a") as f:
        f.write(LINE + "\n")
    assert run_summary(str(tmp_path / "results"), "run")["resources"]["ram_total_mb"] == 6833
