"""Compatibility alias for :mod:`envcore.candidate_offline_runner`."""

import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("envcore.candidate_offline_runner", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("envcore.candidate_offline_runner")
