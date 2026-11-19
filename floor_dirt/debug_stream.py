"""디버그 영상 출력 (params debug.output).

- none: 아무것도 안 함 (기본, 성능 영향 없음)
- file: results/<run>/debug/에 N프레임마다 PNG 저장 (WSL·오프라인)
- rtsp: 젯슨 전용. jetson-utils videoOutput으로 하드웨어 인코딩 송출.
        노트북 VLC에서 rtsp://<젯슨IP>:8554/debug 접속.
"""

import os

import cv2
import numpy as np

MODES = ("none", "file", "rtsp")


class DebugOutput:
    def __init__(self, cfg, out_dir=None):
        self.mode = cfg.get("output", "none")
        if self.mode not in MODES:
            raise ValueError("debug.output은 {} 중 하나여야 합니다: {}".format(MODES, self.mode))
        self.every_n = max(int(cfg.get("every_n", 1)), 1)
        self.out_dir = out_dir
        self.count = 0
        self.written = 0
        self._video = None
        self._ju = None
        if self.mode == "file":
            if not out_dir:
                raise ValueError("file 모드에는 out_dir이 필요합니다")
            os.makedirs(out_dir, exist_ok=True)
        elif self.mode == "rtsp":
            try:
                import jetson_utils as ju  # type: ignore
            except ImportError:
                try:
                    import jetson.utils as ju  # type: ignore
                except ImportError as e:
                    raise ImportError("rtsp 출력은 젯슨(jetson-utils)에서만 됩니다: {}".format(e))
            self._ju = ju
            self._video = ju.videoOutput(cfg.get("rtsp_url", "rtsp://@:8554/debug"))

    @property
    def enabled(self):
        return self.mode != "none"

    def wants(self):
        """이번 프레임을 내보낼 차례인지 (오버레이 그리는 비용을 아끼려고 먼저 묻는다)."""
        if self.mode == "none":
            return False
        if self.mode == "rtsp":
            return True
        return self.count % self.every_n == 0

    def push(self, frame_id, img_bgr):
        """wants()가 True일 때 그린 오버레이를 넘긴다. 호출할 때마다 프레임 카운터가 1 늘어난다."""
        send = self.wants()
        self.count += 1
        if not send:
            return False
        if self.mode == "file":
            cv2.imwrite(os.path.join(self.out_dir, "{:06d}.jpg".format(int(frame_id))), img_bgr)
        elif self.mode == "rtsp":
            rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            self._video.Render(self._ju.cudaFromNumpy(rgb))
        self.written += 1
        return True

    def skip(self):
        """오버레이를 안 그린 프레임도 카운터는 센다."""
        self.count += 1

    def close(self):
        if self._video is not None and hasattr(self._video, "Close"):
            self._video.Close()
        self._video = None
