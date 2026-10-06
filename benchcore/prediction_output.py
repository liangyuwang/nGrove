"""Compatibility alias for :mod:`envcore.prediction_output`."""

import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("envcore.prediction_output", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("envcore.prediction_output")
