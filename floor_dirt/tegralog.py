"""젯슨 자원 사용 기록 (tegrastats) — CLAUDE.md §7: 실행마다 results/<run>/tegrastats.log.

녹화·오프라인 처리 동안 tegrastats를 뒤에서 돌려 파일에 남기고, 끝나면 요약(메모리 최대, CPU·GPU 평균, 최고 온도)을 낸다.
tegrastats가 없는 PC(WSL)에서는 아무것도 하지 않는다.

젯슨(L4T r35) 출력 예:
  11-19-2026 17:32:35 RAM 1193/6833MB (lfb 1136x4MB) SWAP 0/11608MB (cached 0MB)
  CPU [26%@1420,20%@1420,17%@1420,off,off,off] EMC_FREQ 0% GR3D_FREQ 0% CPU@34.5C GPU@35.5C PMIC@50C
"""

import re
import shutil
import statistics
import subprocess
from typing import Dict, List, Optional

_RAM = re.compile(r"\bRAM (\d+)/(\d+)MB")
_CPU = re.compile(r"\bCPU \[([^\]]*)\]")
_GPU = re.compile(r"\bGR3D_FREQ (\d+)%")
_TEMP = re.compile(r"\b([A-Za-z0-9_]+)@(-?\d+(?:\.\d+)?)C\b")


def parse_line(line):
    # type: (str) -> Optional[Dict]
    """tegrastats 한 줄 → {ram_mb, ram_total_mb, cpu_pct(켜진 코어 평균), gpu_pct, temps{이름: °C}}. 형식이 아니면 None."""
    m = _RAM.search(line)
    if not m:
        return None
    out = {"ram_mb": int(m.group(1)), "ram_total_mb": int(m.group(2)), "cpu_pct": None, "gpu_pct": None, "temps": {}}
    c = _CPU.search(line)
    if c:
        loads = [int(x.split("%")[0]) for x in c.group(1).split(",") if "%" in x]
        if loads:
            out["cpu_pct"] = sum(loads) / len(loads)
    g = _GPU.search(line)
    if g:
        out["gpu_pct"] = int(g.group(1))
    for name, val in _TEMP.findall(line):
        t = float(val)
        if t > -50:          # 꺼진 센서는 -256C 같은 값이 나옴
            out["temps"][name] = t
    return out


def summarize_lines(lines):
    # type: (List[str]) -> Optional[Dict]
    rows = [r for r in (parse_line(l) for l in lines) if r is not None]
    if not rows:
        return None
    cpu = [r["cpu_pct"] for r in rows if r["cpu_pct"] is not None]
    gpu = [r["gpu_pct"] for r in rows if r["gpu_pct"] is not None]
    temps = {}
    for r in rows:
        for k, v in r["temps"].items():
            temps[k] = max(temps.get(k, v), v)
    return {
        "samples": len(rows),
        "ram_max_mb": max(r["ram_mb"] for r in rows),
        "ram_total_mb": rows[0]["ram_total_mb"],
        "cpu_mean_pct": round(statistics.mean(cpu), 1) if cpu else None,
        "cpu_max_pct": round(max(cpu), 1) if cpu else None,
        "gpu_mean_pct": round(statistics.mean(gpu), 1) if gpu else None,
        "gpu_max_pct": max(gpu) if gpu else None,
        "temp_max_c": temps,
    }


def summarize_file(path):
    # type: (str) -> Optional[Dict]
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return summarize_lines(f.readlines())
    except OSError:
        return None


def format_summary(s):
    # type: (Optional[Dict]) -> str
    if not s:
        return "tegrastats 기록 없음"
    hot = max(s["temp_max_c"].items(), key=lambda kv: kv[1]) if s["temp_max_c"] else None
    return "젯슨 자원: 메모리 최대 {}/{} MB, CPU 평균 {}% (최대 {}%), GPU 평균 {}% (최대 {}%){} [{}초]".format(
        s["ram_max_mb"], s["ram_total_mb"], s["cpu_mean_pct"], s["cpu_max_pct"], s["gpu_mean_pct"], s["gpu_max_pct"],
        "" if hot is None else ", 최고 온도 {} {:.1f}°C".format(*hot), s["samples"])


class TegrastatsLogger:
    """with TegrastatsLogger(path): ... 동안 tegrastats를 path에 기록. tegrastats가 없으면 아무것도 안 함."""

    def __init__(self, path, interval_ms=1000, cmd="tegrastats"):
        self.path = path
        self.interval_ms = int(interval_ms)
        self.cmd = cmd
        self._proc = None
        self._file = None

    @property
    def active(self):
        return self._proc is not None

    def start(self):
        exe = shutil.which(self.cmd)
        if exe is None:
            return self
        self._file = open(self.path, "w", encoding="utf-8")
        self._proc = subprocess.Popen([exe, "--interval", str(self.interval_ms)], stdout=self._file,
                                      stderr=subprocess.DEVNULL)
        return self

    def stop(self):
        # type: () -> Optional[Dict]
        if self._proc is None:
            return None
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._file.close()
        self._proc = None
        return summarize_file(self.path)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.summary = self.stop()
