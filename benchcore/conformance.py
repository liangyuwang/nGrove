"""Compatibility alias for :mod:`envcore.conformance`."""

import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("envcore.conformance", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("envcore.conformance")
