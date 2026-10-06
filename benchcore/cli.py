"""Compatibility alias for :mod:`envcore.cli`."""

import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("envcore.cli", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("envcore.cli")
