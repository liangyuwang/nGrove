"""Compatibility alias for :mod:`envcore.agent_cli`."""

import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("envcore.agent_cli", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("envcore.agent_cli")
