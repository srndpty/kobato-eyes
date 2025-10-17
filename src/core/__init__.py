"""Core logic for kobato-eyes."""


from typing import Any


# Avoid importing heavy submodules at package import time.


def __getattr__(name: str) -> Any:
    if name == "pipeline":
        import importlib

        return importlib.import_module(".pipeline", __name__)
    raise AttributeError(name)
