"""Core logic for kobato-eyes."""


# Avoid importing heavy submodules at package import time.
def __getattr__(name: str):
    if name == "pipeline":
        import importlib

        return importlib.import_module(".pipeline", __name__)
    raise AttributeError(name)
