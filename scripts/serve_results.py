#!/usr/bin/env python3
"""결과 웹 페이지: results/ 아래 정찰 결과를 브라우저로 본다 (설치 불필요).

사용 예
  python scripts/serve_results.py                 # http://<주소>:8000
  python scripts/serve_results.py --port 8001 --results results
WSL에서 실행하면 Windows 브라우저에서 http://localhost:8000
젯슨에서 실행하면 노트북 브라우저에서 http://<젯슨IP>:8000 (실시간 모니터는 8080)
종료: Ctrl+C
"""

import argparse
import os
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from floor_dirt.config import REPO_ROOT  # noqa: E402
from floor_dirt.web import ResultsServer  # noqa: E402


def local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "localhost"
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(REPO_ROOT, "results"))
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    srv = ResultsServer(args.results, args.port)
    print("[serve_results] {} 를 제공합니다".format(os.path.abspath(args.results)))
    print("[serve_results] 브라우저: http://localhost:{0}  (다른 기기: http://{1}:{0})".format(srv.port, local_ip()),
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()


if __name__ == "__main__":
    main()
