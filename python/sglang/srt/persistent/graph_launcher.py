from __future__ import annotations

import os
from typing import Optional

import torch
from torch.utils.cpp_extension import load

_ABS_PATH = os.path.dirname(os.path.abspath(__file__))
_MODULE = None
_FAILED = False


def _load_module():
    global _MODULE
    global _FAILED
    if _FAILED:
        raise RuntimeError("Persistent graph launcher is unavailable.")
    if _MODULE is None:
        try:
            _MODULE = load(
                name="sglang_persistent_graph_launcher",
                sources=[os.path.join(_ABS_PATH, "graph_launcher.cpp")],
                extra_cflags=["-O3", "-std=c++17"],
                verbose=False,
            )
        except Exception:
            _FAILED = True
            raise
    return _MODULE


def replay_graph(graph) -> None:
    """Replay a captured CUDA graph via the C++ extension."""
    try:
        _load_module().replay_graph(graph)
    except Exception:
        # Fall back to Python replay when the extension is unavailable.
        if hasattr(graph, "replay"):
            graph.replay()
        elif hasattr(graph, "_graph") and hasattr(graph._graph, "replay"):
            graph._graph.replay()
        else:
            raise


def try_get_launcher() -> Optional[callable]:
    return replay_graph
