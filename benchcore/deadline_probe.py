"""Compatibility alias for :mod:`envcore.deadline_probe`."""

import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("envcore.deadline_probe", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("envcore.deadline_probe")
