"""Compatibility alias for :mod:`envcore.host_agent`."""

import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("envcore.host_agent", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("envcore.host_agent")
