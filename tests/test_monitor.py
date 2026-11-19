import math
import subprocess
import sys

import cv2
import numpy as np
import pytest

from floor_dirt.config import REPO_ROOT, load_params
from floor_dirt.dataset import Frame, FrameDatasetWriter
from floor_dirt.monitor import (
    COL_POINT, COL_ROI, LIDAR_SIZE, RateMeter, camera_panel, closest_point, compose, image_msg_to_bgr,
    lidar_on_camera, lidar_topdown, quaternion_to_yaw, status_lines,
)
from floor_dirt.projection import robot_to_pixel
from synthetic import FloorTexture, c920_like, render

ROI = load_params()["roi"]


def _is(img, v, u, col, tol=40):
    return np.abs(img[v, u].astype(int) - np.array(col)).max() <= tol


def test_lidar_topdown_orientation():
    pts = np.array([[1.0, 0.0], [0.0, 1.0], [-0.5, -0.5]])  # 앞 1 m, 왼쪽 1 m, 오른쪽 뒤
    img = lidar_topdown(pts, size=480, range_m=2.0)
    c, s = 240, 120  # 1 m = 120 px
    assert _is(img, c - s, c, COL_POINT)          # 앞 = 위
    assert _is(img, c, c - s, COL_POINT)          # 왼쪽 = 왼쪽
    assert _is(img, c + 60, c + 60, COL_POINT)    # 뒤 오른쪽 = 아래 오른쪽
    assert img.shape == (480, 480, 3)


def test_lidar_topdown_roi_and_far_points():
    img = lidar_topdown(np.array([[10.0, 0.0]]), range_m=2.0, roi=ROI)  # 범위 밖 점은 그리지 않음 (오류 없음)
    s = 120
    v = 240 - int(round(ROI["x_max"] * s))
    assert _is(img, v, 240 + 10, COL_ROI)          # ROI 앞쪽 변
    assert lidar_topdown(None).shape == (LIDAR_SIZE, LIDAR_SIZE, 3)


def test_closest_point():
    d, ang = closest_point(np.array([[2.0, 0.0], [0.3, 0.3], [-1.0, 0.0]]))
    assert d == pytest.approx(math.hypot(0.3, 0.3)) and ang == pytest.approx(45.0)
    assert closest_point(np.zeros((0, 2))) is None


def test_lidar_points_land_on_camera_floor():
    cam = c920_like()
    H = cam.floor_homography()
    pts = np.array([[0.40, 0.05], [0.30, -0.10], [-0.5, 0.0], [5.0, 3.0]])  # 앞 2개만 화면에
    uv = lidar_on_camera(H, pts, (cam.height, cam.width))
    assert len(uv) == 2
    np.testing.assert_allclose(uv, robot_to_pixel(H, pts[:2]), atol=1e-6)


def test_camera_panel_with_and_without_calibration():
    cam = c920_like()
    H = cam.floor_homography()
    frame = render(cam, FloorTexture(np.full((1000, 1000, 3), 150, np.uint8), 0, 0, 0.002), (0.5, 1.0, 0.0))
    plain = camera_panel(frame)
    assert np.array_equal(plain, frame)
    over = camera_panel(frame, H, ROI, np.array([[0.40, 0.0]]))
    u, v = np.round(robot_to_pixel(H, [[0.40, 0.0]])[0]).astype(int)
    assert _is(over, v, u, COL_POINT)
    assert (np.abs(over.astype(int) - frame.astype(int)).sum(axis=2) > 0).sum() > 500   # ROI 선
    small = camera_panel(cv2.resize(frame, (320, 240)), None)
    assert small.shape == (480, 640, 3)
    assert camera_panel(None).shape == (480, 640, 3)


def test_status_and_compose():
    lines = status_lines(cam_fps=29.7, scan_hz=5.1, n_points=358, closest=(0.42, -15.0),
                         odom={"x": 1.0, "y": -0.5, "yaw": math.radians(30), "v": 0.1, "w": 0.0},
                         cam_ctrl={"AE": "MANUAL", "AF": "OFF"}, calibrated=True, scan_age=0.1)
    assert "29.7 fps" in lines[0] and "358 pts" in lines[0] and "0.42 m" in lines[0] and "calib ON" in lines[0]
    assert "yaw  +30.0" in lines[1] and "AE MANUAL" in lines[2]
    stale = status_lines(scan_hz=0.0, scan_age=5.0)
    assert "NO DATA" in stale[0] and "odom --" in stale[1] and "cam --" in stale[0]
    assert "scan off" in status_lines()[0]
    img = compose(np.zeros((480, 640, 3), np.uint8), np.zeros((480, 480, 3), np.uint8), lines)
    assert img.shape == (480 + 10 + 22 * 3, 1120, 3)


def test_rate_meter():
    t = [0.0]
    rm = RateMeter(window=2.0, clock=lambda: t[0])
    assert rm.rate() == 0.0 and rm.age() == float("inf")
    for i in range(11):
        t[0] = i * 0.2
        rm.tick()
    assert rm.rate() == pytest.approx(5.0)
    t[0] = 5.0
    assert rm.rate() == 0.0 and rm.age() == pytest.approx(3.0)   # 끊기면 0 Hz, 마지막 수신 3초 전


def test_image_msg_conversion():
    bgr = (np.random.RandomState(0).rand(4, 6, 3) * 255).astype(np.uint8)
    pad = np.zeros((4, 2 * 3), np.uint8)  # step에 여분이 있는 경우
    data = np.hstack([bgr.reshape(4, 18), pad]).tobytes()
    np.testing.assert_array_equal(image_msg_to_bgr("bgr8", 4, 6, 24, data), bgr)
    rgb = bgr[..., ::-1].copy()
    np.testing.assert_array_equal(image_msg_to_bgr("rgb8", 4, 6, 18, rgb.tobytes()), bgr)
    yuyv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_YUYV) if hasattr(cv2, "COLOR_BGR2YUV_YUYV") else None
    gray = np.arange(24, dtype=np.uint8).reshape(4, 6)
    assert image_msg_to_bgr("mono8", 4, 6, 6, gray.tobytes()).shape == (4, 6, 3)
    uyvy = np.full((4, 6, 2), 128, np.uint8)
    assert image_msg_to_bgr("yuv422", 4, 6, 12, uyvy.tobytes()).shape == (4, 6, 3)
    with pytest.raises(ValueError):
        image_msg_to_bgr("bayer_rggb8", 4, 6, 6, gray.tobytes())
    del yuyv


def test_quaternion_to_yaw():
    a = math.radians(70)
    assert quaternion_to_yaw(0, 0, math.sin(a / 2), math.cos(a / 2)) == pytest.approx(a)


def test_live_monitor_replays_dataset_to_files(tmp_path):
    """WSL: 데이터셋 재생 → file 출력. 스크립트를 실제 프로세스로 실행 (ROS·카메라 불필요)."""
    cam = c920_like()
    tex = FloorTexture(np.full((1500, 1500, 3), (150, 160, 170), np.uint8), 0, 0, 0.002)
    with FrameDatasetWriter(str(tmp_path / "run")) as wr:
        for i in range(6):
            pose = (0.5 + 0.05 * i, 1.5, 0.0)
            scan = np.array([[0.5, 0.2], [1.2, -0.4]], np.float32)
            wr.write(Frame(i, i * 0.1, pose, render(cam, tex, pose), scan))
    out = tmp_path / "mon"
    r = subprocess.run([sys.executable, "scripts/live_monitor.py", "--dataset", str(tmp_path / "run"),
                        "--output", "file", "--out-dir", str(out), "--every", "2", "--fps", "50", "--no-calib"],
                       cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    files = sorted(p.name for p in out.iterdir())
    assert files == ["000000.jpg", "000002.jpg", "000004.jpg"]
    img = cv2.imread(str(out / files[-1]))
    assert img.shape[1] == 1120
    assert "데이터셋 끝" in r.stdout


# ---------------------------------------------------------------- 브라우저 출력 (MJPEG over HTTP)

def _get(url, timeout=5):
    import urllib.request
    return urllib.request.urlopen(url, timeout=timeout)


def test_mjpeg_server_page_snapshot_and_stream():
    from floor_dirt.mjpeg_server import MjpegServer
    srv = MjpegServer(port=0, host="127.0.0.1", title="test monitor")  # 비어 있는 포트 자동 배정
    base = "http://127.0.0.1:{}".format(srv.port)
    try:
        html = _get(base + "/").read().decode("utf-8")
        assert "test monitor" in html and 'src="/stream.mjpg"' in html

        import urllib.error
        with pytest.raises(urllib.error.HTTPError) as e:
            _get(base + "/snapshot.jpg", timeout=5)            # 아직 화면 없음
        assert e.value.code == 503

        img = np.zeros((60, 80, 3), np.uint8)
        img[:, :40] = (0, 0, 255)
        srv.push(img)
        snap = cv2.imdecode(np.frombuffer(_get(base + "/snapshot.jpg").read(), np.uint8), cv2.IMREAD_COLOR)
        assert snap.shape == (60, 80, 3) and snap[30, 10, 2] > 200 and snap[30, 70, 2] < 50

        resp = _get(base + "/stream.mjpg")
        assert resp.headers["Content-Type"].startswith("multipart/x-mixed-replace")
        import threading
        threading.Timer(0.2, lambda: srv.push(img)).start()     # 스트림은 '새' 화면이 올 때 보낸다
        head = b""
        while b"\r\n\r\n" not in head:
            head += resp.read(1)
        assert head.startswith(b"--frame") and b"image/jpeg" in head
        length = int([l for l in head.split(b"\r\n") if l.startswith(b"Content-Length")][0].split(b":")[1])
        part = cv2.imdecode(np.frombuffer(resp.read(length), np.uint8), cv2.IMREAD_COLOR)
        assert part.shape == (60, 80, 3)
        resp.close()

        with pytest.raises(urllib.error.HTTPError):
            _get(base + "/../config/params.yaml")
    finally:
        srv.close()


def test_debug_output_http_mode():
    from floor_dirt.debug_stream import DebugOutput
    out = DebugOutput({"output": "http", "http_port": 0})
    try:
        assert out.wants() and out.port > 0
        out.push(0, np.zeros((10, 10, 3), np.uint8))
        data = _get("http://127.0.0.1:{}/snapshot.jpg".format(out.port)).read()
        assert data[:2] == b"\xff\xd8"                          # JPEG 시작 표시
    finally:
        out.close()


# ---------------------------------------------------------------- 전원 표시

def test_battery_percent_and_power_line():
    from floor_dirt.jetson_power import Rail
    from floor_dirt.monitor import battery_percent, power_line
    assert battery_percent(12.3) == 100.0 and battery_percent(10.5) == 0.0
    assert battery_percent(11.4) == pytest.approx(50.0) and battery_percent(13.0) == 100.0
    line = power_line({"voltage": 12.05, "percentage": 86.0}, Rail("VDD_IN", 19.05, 8.2))
    assert line == "OpenCR bat 12.05 V (86%) | Jetson VDD_IN 19.05 V 8.2 W"
    assert "LOW" in power_line({"voltage": 11.05, "percentage": None}, None)
    assert "CHARGE NOW" in power_line({"voltage": 10.9, "percentage": None}, None)
    assert power_line(None, None) == "OpenCR bat -- | Jetson in n/a"
    assert power_line(None, None, show_jetson=False) == "OpenCR bat --"
    lines = status_lines(power="OpenCR bat 12.00 V", cam_ctrl={"AE": "AUTO"})
    assert lines[2] == "OpenCR bat 12.00 V" and lines[3].startswith("AE")


def test_jetson_power_reads_ina3221(tmp_path):
    from floor_dirt.jetson_power import input_rail, read_rails
    assert read_rails(str(tmp_path)) == []                     # 센서 없는 보드
    other = tmp_path / "hwmon0"
    other.mkdir()
    (other / "name").write_text("cpu_thermal\n")
    dev = tmp_path / "hwmon3"
    dev.mkdir()
    (dev / "name").write_text("ina3221\n")
    for n, (label, mv, ma) in enumerate([("VDD_IN", 19040, 450), ("VDD_CPU_GPU_CV", 19040, 120), ("NC", 0, 0)], 1):
        (dev / "in{}_label".format(n)).write_text(label + "\n")
        (dev / "in{}_input".format(n)).write_text(str(mv) + "\n")
        (dev / "curr{}_input".format(n)).write_text(str(ma) + "\n")
    rails = read_rails(str(tmp_path))
    assert [r.label for r in rails] == ["VDD_IN", "VDD_CPU_GPU_CV"]    # NC(연결 안 됨)는 제외
    r = input_rail(rails)
    assert r.label == "VDD_IN" and r.volts == pytest.approx(19.04) and r.watts == pytest.approx(19.04 * 0.45)


# ---------------------------------------------------------------- 계기판

def test_level_colors():
    from floor_dirt.monitor import C_BAD, C_OFF, C_OK, C_WARN, level
    assert level(29.0, 20, 1) == C_OK and level(5.0, 20, 1) == C_WARN and level(0.0, 20, 1) == C_BAD
    assert level(None, 20, 1) == C_OFF
    assert level(0.6, 0.5, 0.25) == C_OK and level(0.3, 0.5, 0.25) == C_WARN and level(0.1, 0.5, 0.25) == C_BAD


def _near(img, v, u, col, tol=30):
    return np.abs(img[v, u].astype(int) - np.array(col)).max() <= tol


def test_dashboard_lights_and_bars():
    from floor_dirt.jetson_power import Rail
    from floor_dirt.monitor import C_BAD, C_OFF, C_OK, C_WARN, DASH_H, compose_dashboard, dashboard
    good = dict(cam_on=True, cam_fps=30.0, ros_on=True, scan_hz=5.0, scan_age=0.1, n_points=300,
                closest=(1.5, 0.0), odom={"x": 0, "y": 0, "yaw": 0, "v": 0.2, "w": 0.0}, odom_age=0.05,
                battery={"voltage": 12.1, "percentage": 89}, jetson=Rail("VDD_IN", 19.0, 8.0),
                cam_ctrl={"AE": "MANUAL", "AF": "OFF", "AWB": "OFF"}, calibrated=True)
    img = dashboard(1120, good)
    assert img.shape == (DASH_H, 1120, 3)
    x = 14 + 9
    for y in (40, 64, 88, 112):                                   # CAM, LiDAR, ODOM, CALIB 신호등
        assert _near(img, y, x, C_OK)

    bad = dict(cam_on=True, cam_fps=0.0, ros_on=True, scan_age=None, odom=None,
               battery={"voltage": 10.8, "percentage": 17}, cam_ctrl={"AE": "AUTO", "AF": "AUTO", "AWB": "AUTO"})
    img = dashboard(1120, bad)
    for y in (40, 64, 88):
        assert _near(img, y, x, C_BAD)
    assert _near(img, 112, x, C_OFF)                              # 보정 없음 = 회색
    bx = int(1120 * 0.26) + 14 + 5
    assert _near(img, 45, bx, C_BAD)                              # 배터리 아이콘 빨강
    chips = img[104:127, int(1120 * 0.76) + 14:]
    assert (np.abs(chips.astype(int) - np.array(C_WARN)).max(axis=2) <= 30).sum() > 200   # 자동 = 노랑 칩

    off = dashboard(1120, {})                                     # 아무 정보 없음: 오류 없이 회색
    assert off.shape == (DASH_H, 1120, 3)
    full = compose_dashboard(np.zeros((480, 640, 3), np.uint8), np.zeros((480, 480, 3), np.uint8), good)
    assert full.shape == (480 + DASH_H, 1120, 3)


def test_dashboard_speed_bar_direction():
    from floor_dirt.monitor import dashboard
    base = dict(ros_on=True, odom_age=0.0)
    fwd = dashboard(1120, dict(base, odom={"x": 0, "y": 0, "yaw": 0, "v": 0.2, "w": 0.0}))
    back = dashboard(1120, dict(base, odom={"x": 0, "y": 0, "yaw": 0, "v": -0.2, "w": 0.0}))
    x0, x1 = int(1120 * 0.50) + 14, int(1120 * 0.76) - 14
    cx = (x0 + x1) // 2
    row_f, row_b = fwd[53, x0:x1], back[53, x0:x1]
    blue = lambda row: np.nonzero((row[:, 0] > 180) & (row[:, 2] < 100))[0] + x0
    assert blue(row_f).min() >= cx - 1 and blue(row_f).max() > cx + 50     # 전진 = 오른쪽으로
    assert blue(row_b).max() <= cx + 1 and blue(row_b).min() < cx - 50     # 후진 = 왼쪽으로


def test_tegrastats_parsing_and_fallback(tmp_path, monkeypatch):
    from floor_dirt import jetson_power as jp
    line5 = ("11-19-2026 16:16:27 RAM 2345/6855MB (lfb 2x4MB) CPU [12%@1190,off] GR3D_FREQ 0% "
             "CPU@45.5C VDD_IN 4523mW/4400mW VDD_CPU_GPU_CV 610mW/610mW VDD_SOC 1424mW/1424mW")
    r = jp.parse_tegrastats(line5)
    assert r.label == "VDD_IN" and r.volts is None and r.watts == pytest.approx(4.523)
    r4 = jp.parse_tegrastats("RAM 1000/3964MB CPU [5%@102] POM_5V_IN 3000/2900 POM_5V_GPU 0/0")
    assert r4.label == "POM_5V_IN" and r4.watts == pytest.approx(3.0)
    assert jp.parse_tegrastats("RAM 1/2MB CPU [1%]") is None

    # 센서도 tegrastats도 없는 PC(WSL): 오류 없이 n/a
    monkeypatch.setattr(jp.shutil, "which", lambda cmd: None)
    p = jp.JetsonPower(root=str(tmp_path))
    assert p.source == "none" and p.read() is None
    p.close()

    from floor_dirt.monitor import dashboard, power_line
    assert power_line(None, r) == "OpenCR bat -- | Jetson VDD_IN 4.5 W"
    assert dashboard(1120, {"jetson": r}).shape[1] == 1120


def test_tegrastats_reader_with_fake_command(tmp_path):
    """tegrastats 흉내 스크립트로 백그라운드 읽기 확인."""
    import os
    import time
    from floor_dirt.jetson_power import TegrastatsReader
    fake = tmp_path / "tegrastats"
    fake.write_text("#!/bin/sh\nwhile true; do echo 'RAM 1/2MB VDD_IN 5100mW/5000mW'; sleep 0.1; done\n")
    fake.chmod(0o755)
    rd = TegrastatsReader(cmd=str(fake))
    try:
        t0 = time.time()
        while rd.latest() is None and time.time() - t0 < 3:
            time.sleep(0.05)
        assert rd.latest().watts == pytest.approx(5.1)
    finally:
        rd.close()
    assert not os.path.exists("/proc/{}".format(rd._proc.pid)) or rd._proc.poll() is not None
