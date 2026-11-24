"""녹화 준비 테스트: 카메라 설정 계획, 녹화 로직(주기·스캔 짝·TF 대기열), ROS 노드·스크립트 문법."""

import math
import py_compile
import subprocess
import glob
import os

import pytest

from floor_dirt.camera_settings import describe, parse_list_ctrls, plan, resolve_name
from floor_dirt.config import REPO_ROOT
from floor_dirt.recording import PendingFrames, RateGate, ScanBuffer, quat_to_yaw, transform_to_pose

# 젯슨 C920에서 실제로 나온 출력 (2026-11-24)
C920_CTRLS = """
                     brightness 0x00980900 (int)    : min=0 max=255 step=1 default=128 value=128
                       contrast 0x00980901 (int)    : min=0 max=255 step=1 default=128 value=128
                     saturation 0x00980902 (int)    : min=0 max=255 step=1 default=128 value=128
 white_balance_temperature_auto 0x0098090c (bool)   : default=1 value=1
                           gain 0x00980913 (int)    : min=0 max=255 step=1 default=0 value=0
           power_line_frequency 0x00980918 (menu)   : min=0 max=2 default=2 value=2
      white_balance_temperature 0x0098091a (int)    : min=2000 max=6500 step=1 default=4000 value=4000 flags=inactive
                      sharpness 0x0098091b (int)    : min=0 max=255 step=1 default=128 value=128
         backlight_compensation 0x0098091c (int)    : min=0 max=1 step=1 default=0 value=0
                  exposure_auto 0x009a0901 (menu)   : min=0 max=3 default=3 value=3
              exposure_absolute 0x009a0902 (int)    : min=3 max=2047 step=1 default=250 value=250 flags=inactive
         exposure_auto_priority 0x009a0903 (bool)   : default=0 value=1
                   pan_absolute 0x009a0908 (int)    : min=-36000 max=36000 step=3600 default=0 value=0
                  tilt_absolute 0x009a0909 (int)    : min=-36000 max=36000 step=3600 default=0 value=0
                 focus_absolute 0x009a090a (int)    : min=0 max=250 step=5 default=0 value=0 flags=inactive
                     focus_auto 0x009a090c (bool)   : default=1 value=1
                  zoom_absolute 0x009a090d (int)    : min=100 max=500 step=1 default=100 value=100
"""


# ---------------------------------------------------------------- 카메라 설정

def test_parse_c920_controls():
    c = parse_list_ctrls(C920_CTRLS)
    assert len(c) == 17
    assert c["exposure_auto"].value == 3 and c["exposure_auto"].kind == "menu"
    assert c["exposure_absolute"].minimum == 3 and c["exposure_absolute"].maximum == 2047
    assert c["exposure_absolute"].inactive and not c["gain"].inactive
    assert c["focus_auto"].minimum is None and c["focus_auto"].value == 1
    assert describe(c) == {"AE": "AUTO", "AF": "AUTO", "AWB": "AUTO"}   # 지금 젯슨 상태: 전부 자동


def test_plan_turns_auto_off_first_and_clamps():
    c = parse_list_ctrls(C920_CTRLS)
    steps, warnings = plan({"exposure_absolute": 5000, "focus_absolute": 30, "focus_auto": 0,
                            "exposure_auto": 1, "zoom_magic": 3}, c)
    names = [n for n, _ in steps]
    assert names.index("exposure_auto") < names.index("exposure_absolute")
    assert names.index("focus_auto") < names.index("focus_absolute")
    assert ("exposure_absolute", 2047) in steps                         # 최대값으로 잘림
    assert any("zoom_magic" in w for w in warnings) and any("2047" in w for w in warnings)


def test_new_kernel_names_are_resolved():
    new = parse_list_ctrls(C920_CTRLS.replace("exposure_auto ", "auto_exposure ")
                           .replace("focus_auto ", "focus_automatic_continuous "))
    assert resolve_name("exposure_auto", new) == "auto_exposure"
    assert resolve_name("focus_automatic_continuous", parse_list_ctrls(C920_CTRLS)) == "focus_auto"
    steps, _ = plan({"exposure_auto": 1, "focus_auto": 0}, new)
    assert steps == [("auto_exposure", 1), ("focus_automatic_continuous", 0)]


def test_repo_camera_settings_fit_c920():
    from floor_dirt.camera_settings import load_settings
    s = load_settings(os.path.join(REPO_ROOT, "config", "camera_settings.yaml"))
    steps, warnings = plan(s["controls"], parse_list_ctrls(C920_CTRLS))
    assert warnings == []                                               # 모든 이름·범위가 C920에 맞음
    d = dict(steps)
    assert d["exposure_auto"] == 1 and d["focus_auto"] == 0 and d["white_balance_temperature_auto"] == 0
    assert d["exposure_auto_priority"] == 0


# ---------------------------------------------------------------- 녹화 로직

def test_rate_gate():
    g = RateGate(5.0)
    times = [0.0, 0.05, 0.1, 0.2, 0.33, 0.4, 0.6]
    assert [g.accept(t) for t in times] == [True, False, False, True, False, True, True]


def test_scan_buffer_picks_nearest_within_slop():
    b = ScanBuffer(keep_s=1.0)
    for t in (0.0, 0.2, 0.4, 0.6):
        b.add(t, "scan@{}".format(t))
    assert b.nearest(0.43, 0.05) == "scan@0.4"
    assert b.nearest(0.3, 0.05) is None                                 # 가장 가까운 것도 0.1 s 차이
    b.add(1.5, "late")
    assert [t for t, _ in b.items] == [0.6, 1.5]                        # 1초보다 오래된 스캔은 버림


def test_pending_frames_wait_for_tf_then_drop():
    q = PendingFrames(max_wait_s=1.0, max_len=3)
    q.add(1.0, "a", now=10.0)
    q.add(1.2, "b", now=10.2)
    tf_until = [1.1]                                                     # TF는 1.1 s까지만 들어와 있음
    can = lambda p: p.stamp <= tf_until[0]
    assert [p.image for p in q.ready(can, now=10.3)] == ["a"]
    assert [p.image for p in q.items] == ["b"]
    assert q.ready(can, now=11.3) == [] and q.dropped == 1               # 1초 넘게 TF가 없어서 버림
    for i in range(4):
        q.add(2.0 + i, i, now=12.0)
    assert q.dropped == 2 and [p.image for p in q.items] == [1, 2, 3]    # 꽉 차면 오래된 것부터 밀려남


def test_transform_to_pose():
    a = math.radians(30)
    x, y, yaw = transform_to_pose(1.0, -2.0, 0.0, 0.0, math.sin(a / 2), math.cos(a / 2))
    assert (x, y) == (1.0, -2.0) and yaw == pytest.approx(a)
    assert quat_to_yaw(0, 0, 1, 0) == pytest.approx(math.pi)


# ---------------------------------------------------------------- 젯슨 전용 파일 문법

def test_ros_nodes_compile_and_scripts_parse():
    for f in glob.glob(os.path.join(REPO_ROOT, "ros_nodes", "*.py")) + \
            glob.glob(os.path.join(REPO_ROOT, "launch", "*.py")):
        py_compile.compile(f, doraise=True)
    for f in glob.glob(os.path.join(REPO_ROOT, "scripts", "*.sh")):
        r = subprocess.run(["bash", "-n", f], capture_output=True, text=True)
        assert r.returncode == 0, (f, r.stderr)


def test_record_sh_requires_run_name():
    r = subprocess.run(["bash", os.path.join(REPO_ROOT, "scripts", "record.sh")], capture_output=True, text=True)
    assert r.returncode == 1 and "사용법" in r.stdout


def test_apply_config_skips_without_v4l2(tmp_path, monkeypatch):
    from floor_dirt import camera_settings as cs
    msgs = []
    assert cs.apply_config(str(tmp_path / "none.yaml"), log=msgs.append) is False and "없어" in msgs[0]
    monkeypatch.setattr(cs.shutil, "which", lambda name: None)           # v4l2-ctl 없음 (WSL)
    msgs.clear()
    assert cs.apply_config(os.path.join(REPO_ROOT, "config", "camera_settings.yaml"), log=msgs.append) is False
    assert "실패" in msgs[0]
