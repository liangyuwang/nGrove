"""Compatibility alias for :mod:`envcore.docker_recovery`."""

import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("envcore.docker_recovery", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("envcore.docker_recovery")
