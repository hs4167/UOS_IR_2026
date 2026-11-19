"""젯슨 전원 상태 읽기 (INA3221 전력 센서, Linux hwmon).

보조배터리는 잔량(%)을 젯슨에 알려주지 않는다. 대신 젯슨에 들어오는 전압과 소비 전력은
보드의 INA3221 센서로 읽을 수 있다. 보조배터리는 거의 다 쓸 때까지 출력 전압을 유지하므로
전압이 떨어지기 시작하면 곧 꺼진다는 신호로 본다.

JetPack 5 (L4T r35): /sys/class/hwmon/hwmon*/name == "ina3221"
  in{N}_label (레일 이름, 예: VDD_IN), in{N}_input (mV), curr{N}_input (mA)
ASUS PE1000N 등 캐리어 보드에 센서가 없으면 빈 목록을 돌려준다.
"""

import glob
import os
from typing import List, NamedTuple, Optional

HWMON_ROOT = "/sys/class/hwmon"


class Rail(NamedTuple):
    label: str
    volts: float
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
