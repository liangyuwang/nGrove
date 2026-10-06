"""Compatibility alias for :mod:`envcore.candidate_protocol`."""

import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("envcore.candidate_protocol", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("envcore.candidate_protocol")
