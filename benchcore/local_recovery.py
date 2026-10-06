"""Compatibility alias for :mod:`envcore.local_recovery`."""

import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("envcore.local_recovery", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("envcore.local_recovery")
