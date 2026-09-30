"""2s serve-host telemetry sampler. Load-only summary is computed after the run."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


SSH = [
    "ssh",
    "-i",
    str(Path.home() / ".ssh/id_ed25519_shared"),
    "-o",
    "IdentitiesOnly=yes",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=8",
    "-o",
    "StrictHostKeyChecking=yes",
]
HOST = "r0b0tdgx@192.168.68.78"
REMOTE = (
    "python3 - <<'PY'\n"
    "import json, pathlib\n"
    "mem = {}\n"
    "for line in pathlib.Path('/proc/meminfo').read_text().splitlines():\n"
    "    k, v = line.split(':', 1)\n"
    "    if k in {'MemAvailable', 'MemTotal', 'SwapTotal', 'SwapFree'}:\n"
    "        mem[k] = int(v.split()[0])\n"
    "print(json.dumps(mem))\n"
    "PY\n"
    "nvidia-smi --query-gpu=power.draw,temperature.gpu,utilization.gpu,"
    "clocks.current.graphics,clocks_throttle_reasons.active --format=csv,noheader,nounits"
)


def main() -> int:
    out = Path(sys.argv[1])
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        while True:
            started = time.time()
            proc = subprocess.run(
                [*SSH, HOST, REMOTE],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            row = {"ts": started, "rc": proc.returncode}
            lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            if lines:
                try:
                    row["meminfo_kib"] = json.loads(lines[0])
                except json.JSONDecodeError:
                    row["meminfo_raw"] = lines[0]
            if len(lines) > 1:
                parts = [part.strip() for part in lines[1].split(",")]
                if len(parts) >= 5:
                    row["power_w"] = parts[0]
                    row["temp_c"] = parts[1]
                    row["util_gpu"] = parts[2]
                    row["clock_graphics"] = parts[3]
                    row["throttle"] = parts[4]
            if proc.returncode != 0:
                row["stderr"] = proc.stderr[-400:]
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            time.sleep(max(0.0, interval - (time.time() - started)))


if __name__ == "__main__":
    raise SystemExit(main())
