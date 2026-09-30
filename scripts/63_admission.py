"""Admission gate for the published Q200v2 runner on the live FINAL3 serve.

The published subset imports admission_control and does not ship it. The local
Qwen kit refuses unless MemAvailable is at least 16 GiB. FINAL3 at
mem-fraction 0.90 holds the unified-memory reservation, so both ranks sit
under that floor (measured 4.8 GiB and 8.7 GiB). Dropping mem-fraction to
satisfy 16 GiB would publish a different profile. docs/PROCEDURES.md section 4
records a Q200v2 text lane whose minimum MemAvailable was 5.07 GiB.

This gate reads /proc/meminfo on both ranks over SSH and refuses the next
batch if either node is below 4 GiB, which is above the 3 GiB docker kill.
It does not report ADMIT when the read fails or the floor is missed.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import subprocess
import time
from pathlib import Path
from typing import Any, Iterator


FLOOR_BYTES = 4 * (1 << 30)
SSH_OPTS = [
    "-o",
    "IdentitiesOnly=yes",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=8",
    "-o",
    "StrictHostKeyChecking=yes",
]


class AdmissionError(RuntimeError):
    pass


class AdmissionNotGranted(AdmissionError):
    pass


def _read_available_kib(host: str, identity: str) -> int:
    proc = subprocess.run(
        [
            "ssh",
            "-i",
            identity,
            *SSH_OPTS,
            host,
            "awk '/MemAvailable:/ {print $2}' /proc/meminfo",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if proc.returncode != 0:
        raise AdmissionError(f"{host}: meminfo read failed: {proc.stderr.strip()}")
    text = proc.stdout.strip()
    if not text.isdigit():
        raise AdmissionError(f"{host}: meminfo parse failed: {text!r}")
    return int(text)


class AdmissionCoordinator:
    def __init__(self, config: dict[str, Any]) -> None:
        ranks = config.get("ranks")
        if not isinstance(ranks, list) or len(ranks) != 2:
            raise AdmissionError("admission config must list exactly two ranks")
        self.ranks = ranks
        floor = config.get("floor_bytes", FLOOR_BYTES)
        if not isinstance(floor, int) or floor < FLOOR_BYTES:
            raise AdmissionError("floor_bytes must be an int >= 4 GiB")
        self.floor_bytes = floor
        self._held = False

    @classmethod
    def from_path(cls, path: str | Path) -> "AdmissionCoordinator":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise AdmissionError("admission config must be an object")
        return cls(raw)

    def _sample(self) -> list[dict[str, Any]]:
        rows = []
        for rank in self.ranks:
            kib = _read_available_kib(rank["host"], rank["identity"])
            available = kib * 1024
            rows.append(
                {
                    "host": rank["host"],
                    "mem_available_bytes": available,
                    "floor_bytes": self.floor_bytes,
                    "admit": available >= self.floor_bytes,
                }
            )
        return rows

    @contextlib.contextmanager
    def request(self, row_id: str) -> Iterator[dict[str, Any]]:
        if self._held:
            raise AdmissionError("coordinator already holds a lease")
        sample = self._sample()
        denied = [row for row in sample if not row["admit"]]
        if denied:
            detail = ", ".join(
                f"{row['host']}={row['mem_available_bytes']}" for row in denied
            )
            raise AdmissionNotGranted(f"NOT_ADMITTED below {self.floor_bytes} bytes: {detail}")
        lease = {
            "lease_id": secrets.token_hex(32),
            "row_id": row_id,
            "sampled_at": time.time(),
            "guard_states": sample,
            "floor_note": "4 GiB live floor; Qwen 16 GiB kit floor is not met on FINAL3 and is disclosed",
        }
        self._held = True
        try:
            yield lease
        finally:
            self._held = False
