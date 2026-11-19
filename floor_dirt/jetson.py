"""jetson-inference / jetson-utils 불러오기 (젯슨 전용).

conda(test1001) Python은 jetson-inference가 설치된 시스템 경로를 기본으로 보지 못할 수 있다.
그래서 import가 실패하면 시스템 경로를 sys.path '맨 뒤'에 붙여 다시 시도한다.
맨 뒤에 붙이는 이유: conda의 numpy·OpenCV가 시스템 것보다 먼저 쓰이게 하기 위해.
"""

import importlib
import os
import sys

SYSTEM_PATHS = (
    "/usr/lib/python3.8/dist-packages",
    "/usr/local/lib/python3.8/dist-packages",
    "/usr/lib/python3/dist-packages",
)


def _try(names):
    for new, old in names:
        try:
            return importlib.import_module(new)
        except ImportError:
            try:
                return importlib.import_module(old)
            except ImportError:
                continue
    return None


def import_jetson(need_inference=True):
    """(jetson_inference 또는 None, jetson_utils) 반환. 못 찾으면 ImportError (원인 안내 포함)."""
    names_u = [("jetson_utils", "jetson.utils")]
    names_i = [("jetson_inference", "jetson.inference")]
    ju = _try(names_u)
    ji = _try(names_i) if need_inference else None
    if ju is None or (need_inference and ji is None):
        added = [p for p in SYSTEM_PATHS if os.path.isdir(p) and p not in sys.path]
        sys.path.extend(added)
        ju = ju or _try(names_u)
        ji = (ji or _try(names_i)) if need_inference else None
    if ju is None or (need_inference and ji is None):
        raise ImportError(
            "jetson-inference/jetson-utils를 불러올 수 없습니다. 젯슨에서 실행 중인지, "
            "~/jetson-inference를 'sudo make install' 했는지 확인하세요. 찾아본 경로: {}".format(", ".join(SYSTEM_PATHS)))
    return ji, ju
