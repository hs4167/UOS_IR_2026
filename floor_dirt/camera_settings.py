"""웹캠(UVC) 수동 설정: config/camera_settings.yaml → v4l2-ctl 명령.

기준·정찰 주행에서 노출·화이트밸런스·초점이 바뀌면 바닥 색과 투영이 달라져 오염으로 잡힌다 (CLAUDE.md §5).
그래서 자동 기능을 끄고 고정값을 쓴다.

순서가 중요하다: 자동 기능(exposure_auto 등)을 먼저 끄고 나서 값(exposure_absolute 등)을 넣어야 한다.
자동이 켜져 있는 동안 값 컨트롤은 'inactive'라 써도 무시된다.

커널 버전에 따라 컨트롤 이름이 다르다 (C920, 젯슨 L4T r35는 옛 이름).
  옛 이름                         새 이름
  exposure_auto                   auto_exposure
  exposure_absolute               exposure_time_absolute
  exposure_auto_priority          exposure_dynamic_framerate
  white_balance_temperature_auto  white_balance_automatic
  focus_auto                      focus_automatic_continuous
"""

import re
import shutil
import subprocess
from typing import Dict, List, NamedTuple, Optional, Tuple

import yaml

ALIASES = {
    "exposure_auto": "auto_exposure",
    "exposure_absolute": "exposure_time_absolute",
    "exposure_auto_priority": "exposure_dynamic_framerate",
    "white_balance_temperature_auto": "white_balance_automatic",
    "focus_auto": "focus_automatic_continuous",
}
# 자동 기능 스위치 → 먼저 적용
AUTO_FIRST = ("exposure_auto", "exposure_auto_priority", "white_balance_temperature_auto", "focus_auto")


class Control(NamedTuple):
    name: str
    kind: str                 # int / bool / menu
    minimum: Optional[int]
    maximum: Optional[int]
    value: Optional[int]
    inactive: bool


_LINE = re.compile(r"^\s*(\w+)\s+0x[0-9a-fA-F]+\s+\((\w+)\)\s*:\s*(.*)$")


def parse_list_ctrls(text):
    # type: (str) -> Dict[str, Control]
    """`v4l2-ctl --list-ctrls` 출력 → {이름: Control}."""
    out = {}
    for line in text.splitlines():
        m = _LINE.match(line)
        if not m:
            continue
        name, kind, rest = m.groups()
        fields = dict(re.findall(r"(\w+)=(-?\d+)", rest))
        out[name] = Control(name, kind,
                            int(fields["min"]) if "min" in fields else None,
                            int(fields["max"]) if "max" in fields else None,
                            int(fields["value"]) if "value" in fields else None,
                            "inactive" in rest)
    return out


def resolve_name(name, available):
    # type: (str, Dict[str, Control]) -> Optional[str]
    """설정 파일의 이름을 이 카메라의 실제 컨트롤 이름으로 (옛/새 이름 모두 허용)."""
    if name in available:
        return name
    alt = ALIASES.get(name)
    if alt and alt in available:
        return alt
    back = {v: k for k, v in ALIASES.items()}.get(name)
    if back and back in available:
        return back
    return None


def plan(controls, available):
    # type: (Dict[str, int], Dict[str, Control]) -> Tuple[List[Tuple[str, int]], List[str]]
    """적용 순서대로 (실제 이름, 값) 목록과 경고 목록. 자동 스위치 먼저, 범위 밖 값은 잘라낸다."""
    ordered = [k for k in AUTO_FIRST if k in controls] + [k for k in controls if k not in AUTO_FIRST]
    steps, warnings = [], []
    for key in ordered:
        real = resolve_name(key, available)
        if real is None:
            warnings.append("카메라에 '{}' 컨트롤이 없어 건너뜀".format(key))
            continue
        val = int(controls[key])
        c = available[real]
        if c.minimum is not None and val < c.minimum:
            warnings.append("{}={} → 최소 {}로 조정".format(real, val, c.minimum))
            val = c.minimum
        if c.maximum is not None and val > c.maximum:
            warnings.append("{}={} → 최대 {}로 조정".format(real, val, c.maximum))
            val = c.maximum
        steps.append((real, val))
    return steps, warnings


def load_settings(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_settings(path, settings):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(settings, f, allow_unicode=True, sort_keys=False)


# ---------------------------------------------------------------- 실제 적용 (젯슨)

def _v4l2(args):
    exe = shutil.which("v4l2-ctl")
    if exe is None:
        raise FileNotFoundError("v4l2-ctl이 없습니다 (sudo apt install v4l-utils)")
    r = subprocess.run([exe] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    return r.returncode, r.stdout, r.stderr


def read_controls(device):
    # type: (str) -> Dict[str, Control]
    code, out, err = _v4l2(["-d", device, "--list-ctrls"])
    if code != 0:
        raise RuntimeError("v4l2-ctl --list-ctrls 실패: {}".format(err.strip()))
    return parse_list_ctrls(out)


def apply(device, controls):
    # type: (str, Dict[str, int]) -> Tuple[List[Tuple[str, int, Optional[int]]], List[str]]
    """설정 적용 후 다시 읽어 확인. 반환: [(이름, 원한 값, 실제 값)], 경고."""
    steps, warnings = plan(controls, read_controls(device))
    for name, val in steps:   # 하나씩 넣어야 '자동 끄기 → 값' 순서가 지켜진다
        code, _, err = _v4l2(["-d", device, "--set-ctrl={}={}".format(name, val)])
        if code != 0:
            warnings.append("{}={} 실패: {}".format(name, val, err.strip()))
    now = read_controls(device)
    result = [(n, v, now[n].value if n in now else None) for n, v in steps]
    for n, want, got in result:
        if got != want:
            warnings.append("{}: {}로 설정했지만 {}로 읽힘".format(n, want, got))
    return result, warnings


def describe(available):
    # type: (Dict[str, Control]) -> Dict[str, str]
    """모니터·로그용 요약: 노출/초점/화이트밸런스가 고정인지."""
    def val(key):
        real = resolve_name(key, available)
        return None if real is None else available[real].value
    ae, af, awb = val("exposure_auto"), val("focus_auto"), val("white_balance_temperature_auto")
    return {
        "AE": "-" if ae is None else ("MANUAL" if ae == 1 else "AUTO"),
        "AF": "-" if af is None else ("OFF" if af == 0 else "AUTO"),
        "AWB": "-" if awb is None else ("OFF" if awb == 0 else "AUTO"),
    }
