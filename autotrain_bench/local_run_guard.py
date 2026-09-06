"""Best-effort operator protection for the bundled local launcher.

The guard does not acquire the Coordinator lock or cancel Python Worker code.
It can revoke local workloads while a synchronous Worker call is waiting, but
shares the launcher's process. Process loss, a stuck interpreter, and runtime
outages still require an independent deployment watchdog.
"""

from __future__ import annotations

import threading
from typing import Callable, List, Optional, Tuple


class LocalResearchGuard:
    def __init__(
        self,
        remaining: Callable[[], Optional[float]],
        terminate: Callable[[], List[Tuple[str, str]]],
    ) -> None:
        self.remaining = remaining
        self.terminate = terminate
        self.failures: List[Tuple[str, str]] = []
        self.cancelled = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name="operator-research-guard", daemon=True
        )

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        try:
            while not self.cancelled.is_set():
                remaining = self.remaining()
                if remaining is None:
                    return
                if remaining <= 0:
                    self.failures.extend(self.terminate())
                    return
                self.cancelled.wait(min(remaining, 0.05))
        except Exception as exc:
            self.failures.append(
                ("operator_guard_failure", f"{type(exc).__name__}: {exc}")
            )

    def close(self) -> List[Tuple[str, str]]:
        """Disarm and finish any bounded revocation before ordinary cleanup."""

        self.cancelled.set()
        if self.thread.ident is not None:
            self.thread.join()
        return list(self.failures)
