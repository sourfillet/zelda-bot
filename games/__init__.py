"""Game registry.

Each game lives in its own subdirectory (matching its retro integration name)
and exposes a ``get_adapter(state)`` factory in ``<game>/adapter.py``.
"""

import importlib

from games.base import GameAdapter

__all__ = ["GameAdapter", "load_adapter"]


def load_adapter(name, state=None):
    """Import games/<name>/adapter.py and return its adapter instance."""
    module = importlib.import_module(f"games.{name}.adapter")
    return module.get_adapter(state)
