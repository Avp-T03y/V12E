#!/usr/bin/env bash
# capture_evidence.sh - record what the platform runs on, for the Platform page.
# Run on the VM host (not inside a container):
#   bash scripts/capture_evidence.sh             # writes ~/evidence/evidence.json
#   bash scripts/capture_evidence.sh OUT.json    # writes OUT.json
# Writes outside the repo by default, so the VM's checkout stays identical to main.
# Records no secrets: no environment variables, tokens, keys or IP addresses.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$HOME/evidence/evidence.json}"
mkdir -p "$(dirname "$OUT")"
cd "$REPO"

python3 - "$OUT" <<'PY'
import datetime, json, os, platform, re, shutil, subprocess, sys

def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:
        return f"(failed: {' '.join(cmd)}: {e})"

out = sys.argv[1]
cpuinfo = open("/proc/cpuinfo").read()
lscpu = run(["lscpu"])

m = re.search(r"machine\s*=\s*([0-9A-Za-z]+)", cpuinfo) or re.search(r"Machine type:\s*(\S+)", lscpu)
machine_type = m.group(1) if m else None
feat = re.search(r"^features\s*:\s*(.*)$", cpuinfo, re.M)
features = feat.group(1).split() if feat else []

mem_kb = int(re.search(r"MemTotal:\s*(\d+)", open("/proc/meminfo").read()).group(1))
disk = shutil.disk_usage("/")

containers = []
ids = run(["docker", "compose", "ps", "-q"]).split()
if ids and not ids[0].startswith("(failed"):
    for line in run(["docker", "stats", "--no-stream", "--format", "{{json .}}", *ids]).splitlines():
        try:
            s = json.loads(line)
        except ValueError:
            continue
        status = run(["docker", "inspect", "-f", "{{.State.Status}}", s["Name"]])
        containers.append({"name": s["Name"], "status": status,
                           "cpu": s["CPUPerc"], "mem": s["MemUsage"]})

ev = {
    "captured_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    "host": platform.node(),
    "arch": platform.machine(),
    "cpus": os.cpu_count(),
    "mem_total_gb": round(mem_kb / 1024**2, 1),
    "disk_total_gb": round(disk.total / 1024**3, 1),
    "machine_type": machine_type,
    "nnpa": "nnpa" in features,
    "cpu_features": features,
    "tls": False,
    "lscpu": lscpu,
    "docker_ps": run(["docker", "compose", "ps", "--format",
                      "table {{.Name}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"]),
    "containers": containers,
}
with open(out, "w") as f:
    json.dump(ev, f, indent=2)
print(f"wrote {out}: arch {ev['arch']}, machine type {machine_type}, "
      f"nnpa {ev['nnpa']}, {len(containers)} containers")
PY