"""Compatibility alias for :mod:`envcore.worker_protocol`."""

import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("envcore.worker_protocol", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("envcore.worker_protocol")
