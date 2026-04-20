"""Named-protocol preset loader.

A **protocol preset** is a pinned set of dataset-preparation and
evaluation parameters that a published paper uses (e.g. NYUv2
Eigen-2014 means rawDepths + Eigen crop + post-alignment clip to
[1e-3, 10] m + 654 samples). Reproduction YAMLs can declare
``protocol: <name>`` to inherit these settings.

The runner merges a protocol's ``fixed`` fields into the reproduction
config. If the reproduction YAML sets any of those fields to a
different value the merge raises — preventing silent drift between
reproductions of the same paper protocol.

Protocol YAMLs live at ``protocols/<name>.yaml`` in the repo root.
See ``protocols/nyu_eigen_2014.yaml`` for the reference shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from plumbline.paths import PROTOCOLS_DIR

__all__ = [
    "ProtocolConflictError",
    "load_protocol",
    "apply_protocol",
]


class ProtocolConflictError(ValueError):
    """Raised when a reproduction tries to override a protocol-fixed field."""


def load_protocol(name: str) -> dict[str, Any]:
    """Load ``protocols/<name>.yaml`` by short name."""
    path = PROTOCOLS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"No protocol preset for '{name}'. Looked at {path}."
        )
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "fixed" not in cfg:
        raise ValueError(f"{path}: missing required top-level 'fixed' block")
    return cfg


def apply_protocol(repro_cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve a reproduction's ``protocol`` field (if any).

    Returns a new dict that deep-merges the protocol's ``fixed`` fields
    into the reproduction config. Raises :class:`ProtocolConflictError`
    if the reproduction sets any fixed field to a different value.

    If the reproduction has no ``protocol`` field this is an identity
    operation — callers can apply it unconditionally.
    """
    protocol_name = repro_cfg.get("protocol")
    if protocol_name is None:
        return repro_cfg
    protocol = load_protocol(protocol_name)
    fixed = protocol.get("fixed", {})

    # First pass: validate every fixed leaf against the reproduction.
    conflicts = list(_find_conflicts(fixed, repro_cfg, []))
    if conflicts:
        lines = [
            f"  {'.'.join(path)}: protocol fixes {fix!r}, reproduction sets {repro!r}"
            for path, fix, repro in conflicts
        ]
        raise ProtocolConflictError(
            f"reproduction conflicts with protocol '{protocol_name}':\n" + "\n".join(lines)
        )

    # Merge is deep and protocol-wins for dict branches; at leaves
    # we've already confirmed the reproduction either matches or is
    # absent, so "protocol wins" never clobbers user intent.
    merged = _deep_merge(fixed, repro_cfg)
    return merged


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


_MISSING = object()


def _find_conflicts(
    fixed: Any,
    repro: Any,
    path: list[str],
):
    """Yield (path, fixed_value, repro_value) tuples for mismatches.

    Recurses through dict branches; at leaves (non-dict values), a
    conflict exists iff the reproduction has a different value. If
    the reproduction doesn't set the leaf at all, no conflict.
    """
    if isinstance(fixed, dict):
        if repro is None:
            # `kwargs:` with no children parses as None. Equivalent to
            # "reproduction doesn't override anything in this subtree" —
            # no conflict.
            return
        if not isinstance(repro, dict):
            # Reproduction has a scalar where protocol expects a dict — conflict.
            yield (path, fixed, repro)
            return
        for k, v in fixed.items():
            child_repro = repro.get(k, _MISSING) if isinstance(repro, dict) else _MISSING
            if child_repro is _MISSING:
                continue
            yield from _find_conflicts(v, child_repro, path + [k])
    else:
        if repro is _MISSING:
            return
        if fixed != repro:
            yield (path, fixed, repro)


def _deep_merge(base: Any, override: Any) -> Any:
    """Deep-merge ``override`` into ``base``; scalar override wins.

    Used to layer a protocol's fixed fields under the reproduction's.
    We've already validated no leaf conflicts, so "override wins" is
    safe — it falls back to protocol values when the reproduction
    doesn't set a field.
    """
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override if override is not None else base
    out = dict(base)
    for k, v in override.items():
        if k in out:
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
