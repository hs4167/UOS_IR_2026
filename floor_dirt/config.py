"""config/params.yaml 로드와 점 표기(dotted key) 조회."""
import copy
import os
from typing import Any, Dict, Optional

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PARAMS_PATH = os.path.join(REPO_ROOT, "config", "params.yaml")

_MISSING = object()


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """override 값을 base 위에 재귀적으로 덮어쓴 새 dict를 돌려준다 (원본은 그대로)."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_params(path: Optional[str] = None,
                overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """params.yaml을 읽고 overrides(실험용 덮어쓰기)를 병합한다."""
    with open(path or DEFAULT_PARAMS_PATH, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f) or {}
    if overrides:
        params = deep_merge(params, overrides)
    return params


def get(params: Dict[str, Any], dotted_key: str, default: Any = _MISSING) -> Any:
    """get(p, "compare.k") 형태로 조회. 키가 없고 default도 없으면 KeyError."""
    node = params
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is _MISSING:
                raise KeyError("params에 '{}' 항목이 없습니다".format(dotted_key))
            return default
        node = node[part]
    return node
