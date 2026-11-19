"""브라우저용 실시간 영상 서버 (MJPEG over HTTP). 표준 라이브러리만 사용.

브라우저는 RTSP를 직접 재생하지 못한다. 대신 JPEG를 연속으로 보내는 MJPEG 스트림은
<img src="/stream.mjpg"> 한 줄로 영상처럼 보인다. 노트북에 설치할 것이 없다.

  http://<젯슨IP>:8080/              보기 페이지
  http://<젯슨IP>:8080/stream.mjpg   영상 스트림
  http://<젯슨IP>:8080/snapshot.jpg  현재 화면 한 장
"""

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

BOUNDARY = "frame"

PAGE = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ margin: 0; background: #181818; color: #ddd; font-family: sans-serif; }}
  header {{ padding: 8px 12px; font-size: 14px; display: flex; gap: 16px; align-items: center; }}
  #state {{ color: #8c8; }}
  img {{ display: block; max-width: 100%; height: auto; margin: 0 auto; }}
  a {{ color: #9cf; }}
</style>
</head>
<body>
<header>
  <strong>{title}</strong>
  <span id="state">연결 중...</span>
  <a href="/snapshot.jpg" download="snapshot.jpg">현재 화면 저장</a>
</header>
<img id="view" src="/stream.mjpg" alt="live">
<script>
  // 연결이 끊기면 2초마다 다시 붙는다 (젯슨 쪽 모니터를 다시 켠 경우 등)
  const img = document.getElementById("view");
  const state = document.getElementById("state");
  img.onload = () => {{ state.textContent = "수신 중"; state.style.color = "#8c8"; }};
  img.onerror = () => {{
    state.textContent = "연결 끊김, 다시 연결 중...";
    state.style.color = "#e88";
    setTimeout(() => {{ img.src = "/stream.mjpg?t=" + Date.now(); }}, 2000);
  }};
</script>
</body>
</html>
"""


class MjpegServer:
    """push(img)로 넣은 최신 화면을 접속한 모든 브라우저에 보낸다. 백그라운드 스레드에서 동작."""

    def __init__(self, port=8080, host="0.0.0.0", quality=80, title="live monitor"):
        self.quality = int(quality)
        self.title = title
        self._jpeg = None
        self._seq = 0
        self._cond = threading.Condition()
        self._closed = False
        self.clients = 0
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # 요청마다 터미널에 찍히지 않게
                pass

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path in ("/", "/index.html"):
                    body = PAGE.format(title=server.title).encode("utf-8")
                    self._send(200, "text/html; charset=utf-8", body)
                elif path == "/snapshot.jpg":
                    jpeg = server.latest(timeout=2.0)
                    if jpeg is None:
                        self._send(503, "text/plain; charset=utf-8", "아직 화면이 없습니다".encode("utf-8"))
                    else:
                        self._send(200, "image/jpeg", jpeg)
                elif path == "/stream.mjpg":
                    self._stream()
                else:
                    self._send(404, "text/plain; charset=utf-8", b"not found")

            def _send(self, code, ctype, body):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _stream(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=" + BOUNDARY)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                with server._cond:
                    server.clients += 1
                seq = -1
                try:
                    while not server._closed:
                        jpeg, seq = server._wait_new(seq, timeout=5.0)
                        if jpeg is None:
                            continue
                        self.wfile.write(("--{}\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n"
                                          .format(BOUNDARY, len(jpeg))).encode("ascii"))
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass  # 브라우저 탭을 닫음
                finally:
                    with server._cond:
                        server.clients -= 1

        self.httpd = ThreadingHTTPServer((host, int(port)), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]  # port=0이면 비어 있는 포트가 배정됨
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()

    def push(self, img_bgr):
        """새 화면. JPEG 인코딩은 여기서 한 번만 하고 모든 접속자가 같은 바이트를 받는다."""
        ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        with self._cond:
            self._jpeg = buf.tobytes()
            self._seq += 1
            self._cond.notify_all()

    def latest(self, timeout=0.0):
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._jpeg is None and not self._closed:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self._cond.wait(left)
            return self._jpeg

    def _wait_new(self, seq, timeout):
        with self._cond:
            if self._seq == seq and not self._closed:
                self._cond.wait(timeout)
            if self._jpeg is None or self._seq == seq:
                return None, seq
            return self._jpeg, self._seq

    def close(self):
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        self.httpd.shutdown()
        self.httpd.server_close()
