"""1단계 공통 기반(config, timing, dataset) 테스트."""
import csv
import math
import time

import numpy as np
import pytest

from floor_dirt import config
from floor_dirt.dataset import Frame, FrameDataset, FrameDatasetWriter
from floor_dirt.timing import StageTimer


# ---------- config ----------

def test_default_params_load_and_required_keys():
    p = config.load_params()
    for key in ["roi.x_min", "roi.x_max", "grid.resolution", "compare.method",
                "dirt_map.p_hit", "dirt_map.p_miss", "exclusion.lidar_radius",
                "revisit.restore_ratio", "debug.output"]:
        config.get(p, key)  # 없으면 KeyError
    assert config.get(p, "roi.x_min") < config.get(p, "roi.x_max")
    assert config.get(p, "dirt_map.p_miss") < 0.5 < config.get(p, "dirt_map.p_hit")


def test_overrides_merge_without_touching_siblings():
    p = config.load_params(overrides={"compare": {"k": 5.0}})
    assert config.get(p, "compare.k") == 5.0
    assert config.get(p, "compare.method") == config.get(config.load_params(), "compare.method")  # 형제 키 유지


def test_get_missing_key():
    p = config.load_params()
    with pytest.raises(KeyError):
        config.get(p, "compare.no_such_key")
    assert config.get(p, "compare.no_such_key", None) is None


# ---------- timing ----------

def test_stage_timer_records_and_writes_csv(tmp_path):
    path = tmp_path / "timing.csv"
    timer = StageTimer(["a", "b"], str(path))
    with timer.stage("a"):
        time.sleep(0.01)
    with timer.stage("a"):  # 같은 단계 두 번 → 합산
        time.sleep(0.01)
    row = timer.end_frame(0)
    timer.end_frame(1)  # 아무 단계도 안 돈 프레임
    timer.close()

    assert row["a"] >= 18.0 and row["b"] == 0.0
    rows = list(csv.reader(open(str(path), encoding="utf-8")))
    assert rows[0] == ["frame_id", "a_ms", "b_ms", "total_ms"]
    assert len(rows) == 3 and float(rows[2][3]) == 0.0


def test_stage_timer_rejects_unknown_stage():
    timer = StageTimer(["a"])
    with pytest.raises(KeyError):
        with timer.stage("zzz"):
            pass


# ---------- dataset ----------

def test_dataset_roundtrip(tmp_path):
    run = str(tmp_path / "run1")
    rng = np.random.default_rng(0)
    frames = []
    for i in range(3):
        img = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
        scan = rng.normal(size=(10, 2)).astype(np.float32) if i != 1 else None
        frames.append(Frame(i, 100.0 + 0.1 * i, (1.0 + i, -2.0, math.pi / 3), img, scan))

    with FrameDatasetWriter(run, meta={"lighting": "형광등 켬", "time": "14:00"}) as w:
        for f in frames:
            w.write(f)

    ds = FrameDataset(run)
    assert len(ds) == 3
    assert ds.meta["lighting"] == "형광등 켬"
    for orig, got in zip(frames, ds):
        assert got.frame_id == orig.frame_id
        assert got.stamp == orig.stamp            # repr로 저장해 정밀도 손실 없음
        assert got.pose == orig.pose
        assert np.array_equal(got.image, orig.image)  # PNG 무손실
        if orig.scan is None:
            assert got.scan is None
        else:
            assert np.array_equal(got.scan, orig.scan)


def test_env_override_file(tmp_path, monkeypatch):
    """FLOOR_DIRT_PARAMS_OVERRIDE: params.yaml을 고치지 않고 일부 항목만 바꾼다 (테스트·비교 실험용)."""
    f = tmp_path / "ov.yaml"
    f.write_text("compare:\n  k: 7.5\n")
    monkeypatch.setenv(config.OVERRIDE_ENV, str(f))
    p = config.load_params(overrides={"compare": {"min_thr": 1.0}})
    assert config.get(p, "compare.k") == 7.5 and config.get(p, "compare.min_thr") == 1.0
    assert config.get(p, "compare.method")                                # 나머지는 그대로
