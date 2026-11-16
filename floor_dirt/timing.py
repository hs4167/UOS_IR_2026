"""처리 단계별 소요 시간(ms)을 프레임마다 기록한다 (CLAUDE.md §7).

사용 예:
    timer = StageTimer(["projection", "segnet", "compare"], "results/run1/timing.csv")
    with timer.stage("projection"):
        ...
    timer.end_frame(frame_id)
    timer.close()
"""
import csv
import time
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional


class StageTimer:
    def __init__(self, stages: List[str], csv_path: Optional[str] = None):
        self.stages = list(stages)
        self._current = {}  # type: Dict[str, float]
        self._file = None
        self._writer = None
        if csv_path is not None:
            self._file = open(csv_path, "w", newline="", encoding="utf-8")
            self._writer = csv.writer(self._file)
            self._writer.writerow(["frame_id"] + [s + "_ms" for s in self.stages] + ["total_ms"])

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if name not in self.stages:
            raise KeyError("등록되지 않은 단계: {}".format(name))
        t0 = time.perf_counter()
        try:
            yield
        finally:
            # 같은 단계가 한 프레임에 여러 번 불리면 합산
            self._current[name] = self._current.get(name, 0.0) + (time.perf_counter() - t0) * 1000.0

    def end_frame(self, frame_id: int) -> Dict[str, float]:
        """현재 프레임 기록을 확정하고 {단계: ms}를 돌려준다. 실행 안 된 단계는 0."""
        row = {s: self._current.get(s, 0.0) for s in self.stages}
        self._current = {}
        if self._writer is not None:
            self._writer.writerow([frame_id] + ["{:.3f}".format(row[s]) for s in self.stages]
                                  + ["{:.3f}".format(sum(row.values()))])
            self._file.flush()
        return row

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None
