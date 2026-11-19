"""젯슨 전원 상태 읽기 (INA3221 전력 센서, Linux hwmon).

보조배터리는 잔량(%)을 젯슨에 알려주지 않는다. 대신 젯슨에 들어오는 전압과 소비 전력은
보드의 INA3221 센서로 읽을 수 있다. 보조배터리는 거의 다 쓸 때까지 출력 전압을 유지하므로
전압이 떨어지기 시작하면 곧 꺼진다는 신호로 본다.

JetPack 5 (L4T r35): /sys/class/hwmon/hwmon*/name == "ina3221"
  in{N}_label (레일 이름, 예: VDD_IN), in{N}_input (mV), curr{N}_input (mA)
ASUS PE1000N 등 캐리어 보드에 센서가 없으면 빈 목록을 돌려준다.

센서 파일을 못 찾으면 젯슨 기본 도구 tegrastats 출력에서 소비 전력만 읽는다 (전압은 없음).
  JetPack 5: "... VDD_IN 4523mW/4523mW VDD_CPU_GPU_CV 610mW/610mW ..."
  JetPack 4: "... POM_5V_IN 3000/3000 ..."
"""

import glob
import os
import re
import shutil
import subprocess
import threading
from typing import List, NamedTuple, Optional

HWMON_ROOT = "/sys/class/hwmon"


class Rail(NamedTuple):
    label: str
    volts: Optional[float]   # tegrastats로 읽으면 None
    watts: Optional[float]   # 전류 값이 없으면 None


def _read(path):
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except OSError:
        return None


def read_rails(root=HWMON_ROOT):
    # type: (str) -> List[Rail]
    """INA3221 레일 목록. 센서가 없거나 읽을 수 없으면 []."""
    rails = []
    for dev in sorted(glob.glob(os.path.join(root, "hwmon*"))):
        if (_read(os.path.join(dev, "name")) or "").lower() != "ina3221":
            continue
        for lab_path in sorted(glob.glob(os.path.join(dev, "in*_label"))):
            n = os.path.basename(lab_path)[2:-len("_label")]
            label = _read(lab_path)
            mv = _read(os.path.join(dev, "in{}_input".format(n)))
            if not label or label.upper() == "NC" or mv is None:
                continue
            try:
                volts = int(mv) / 1000.0
            except ValueError:
                continue
            ma = _read(os.path.join(dev, "curr{}_input".format(n)))
            watts = None
            if ma is not None:
                try:
                    watts = volts * int(ma) / 1000.0
                except ValueError:
                    pass
            rails.append(Rail(label, volts, watts))
    return rails


def input_rail(rails):
    # type: (List[Rail]) -> Optional[Rail]
    """전체 입력 레일 (VDD_IN 우선, 없으면 첫 레일)."""
    for r in rails:
        if r.label.upper() in ("VDD_IN", "VIN", "VDD_IN_SYS"):
            return r
    return rails[0] if rails else None


# ---------------------------------------------------------------- tegrastats (대체 방법)

_TEGRA_RE = re.compile(r"\b(VDD_IN|POM_5V_IN|VIN_SYS_5V0|VDD_IN_SYS)\s+(\d+)(?:mW)?/(\d+)(?:mW)?")


def parse_tegrastats(line):
    # type: (str) -> Optional[Rail]
    """tegrastats 한 줄 → 입력 레일 소비 전력 (현재값). 없으면 None."""
    m = _TEGRA_RE.search(line)
    if not m:
        return None
    return Rail(m.group(1), None, int(m.group(2)) / 1000.0)


class TegrastatsReader:
    """tegrastats를 백그라운드로 돌리며 최신 입력 전력을 보관한다."""

    def __init__(self, interval_ms=1000, cmd="tegrastats"):
        path = shutil.which(cmd)
        if path is None:
            raise FileNotFoundError("tegrastats가 없습니다")
        self._latest = None
        self._lock = threading.Lock()
        self._proc = subprocess.Popen([path, "--interval", str(int(interval_ms))], stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, universal_newlines=True, bufsize=1)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        for line in self._proc.stdout:
            r = parse_tegrastats(line)
            if r is not None:
                with self._lock:
                    self._latest = r

    def latest(self):
        # type: () -> Optional[Rail]
        with self._lock:
            return self._latest

    def close(self):
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()


class JetsonPower:
    """센서 파일(INA3221) → 없으면 tegrastats → 둘 다 없으면 None."""

    def __init__(self, root=HWMON_ROOT, use_tegrastats=True):
        self.root = root
        self.source = "none"
        self._tegra = None
        if read_rails(root):
            self.source = "ina3221"
        elif use_tegrastats:
            try:
                self._tegra = TegrastatsReader()
                self.source = "tegrastats"
            except (OSError, FileNotFoundError):
                pass

    def read(self):
        # type: () -> Optional[Rail]
        if self.source == "ina3221":
            return input_rail(read_rails(self.root))
        if self._tegra is not None:
            return self._tegra.latest()
        return None

    def close(self):
        if self._tegra is not None:
            self._tegra.close()
