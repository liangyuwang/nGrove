"""Compatibility alias for :mod:`envcore.challenge`."""

import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("envcore.challenge", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("envcore.challenge")
