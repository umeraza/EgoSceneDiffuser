from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


class ConfigNode(dict):
    """Dictionary with attribute access and recursive conversion."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    __setattr__ = dict.__setitem__

    @classmethod
    def wrap(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls({k: cls.wrap(v) for k, v in value.items()})
        if isinstance(value, list):
            return [cls.wrap(v) for v in value]
        return value

    def to_dict(self) -> dict[str, Any]:
        def unwrap(value: Any) -> Any:
            if isinstance(value, ConfigNode):
                return {k: unwrap(v) for k, v in value.items()}
            if isinstance(value, list):
                return [unwrap(v) for v in value]
            return value

        return unwrap(self)


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_raw(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    base_ref = data.pop("_base_", None)
    if base_ref is None:
        return data
    base_path = (path.parent / base_ref).resolve()
    return deep_merge(_load_raw(base_path), data)


def _parse_scalar(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def apply_overrides(config: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override}")
        key, raw_value = override.split("=", 1)
        parts = key.split(".")
        cursor = result
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise ValueError(f"Cannot set nested override under non-dict key: {key}")
        cursor[parts[-1]] = _parse_scalar(raw_value)
    return result


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> ConfigNode:
    path = Path(path).resolve()
    raw = apply_overrides(_load_raw(path), overrides)
    raw.setdefault("_meta", {})["config_path"] = str(path)
    return ConfigNode.wrap(raw)


def save_config(config: ConfigNode | dict[str, Any], path: str | Path) -> None:
    data = config.to_dict() if isinstance(config, ConfigNode) else config
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
