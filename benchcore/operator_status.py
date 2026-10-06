"""Compatibility alias for :mod:`envcore.operator_status`."""

import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("envcore.operator_status", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("envcore.operator_status")
