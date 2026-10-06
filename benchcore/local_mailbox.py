"""Compatibility alias for :mod:`envcore.local_mailbox`."""

import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("envcore.local_mailbox", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("envcore.local_mailbox")
