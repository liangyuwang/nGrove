"""Compatibility alias for :mod:`envcore.results`."""

import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("envcore.results", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("envcore.results")
