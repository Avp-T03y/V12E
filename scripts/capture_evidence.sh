#!/usr/bin/env bash
# Host-side platform evidence for the Platform view. Run ON THE LINUXONE SERVER, from the repo
# root, AFTER `docker compose up` (it records the running containers):
#     bash scripts/capture_evidence.sh
# Writes $EVIDENCE_DIR/evidence.json (default /srv/v12e/evidence) and the raw outputs next to it.
# Mount the DIRECTORY read-only into the web container (see README), so it is fine to run this
# before or after the stack starts; re-run it any time to refresh the numbers.
#
# Hardware facts (machine type, NNPA flag, vCPUs, memory, Snap ML version, tree format) also come
# from /api/health, read inside the api container; this script adds what only the host can see:
# lscpu, docker compose ps and docker stats. It never needs the Docker socket inside a container.
set -euo pipefail
DIR="${EVIDENCE_DIR:-/srv/v12e/evidence}"
mkdir -p "$DIR/raw"
lscpu > "$DIR/raw/lscpu.txt"
grep -E "^(machine|processor|features|# processors)" /proc/cpuinfo > "$DIR/raw/cpuinfo.txt" || true
cp /proc/meminfo "$DIR/raw/meminfo.txt"
df -B1 / > "$DIR/raw/df.txt"
docker compose ps --format "table {{.Name}}\t{{.Service}}\t{{.Status}}" > "$DIR/raw/docker_ps.txt" 2>&1 || true
docker stats --no-stream --format '{{json .}}' > "$DIR/raw/docker_stats.jsonl" 2>/dev/null || true

python3 - "$DIR" <<'EOF'
import datetime, json, os, re, socket, sys
d = sys.argv[1]
read = lambda name: open(os.path.join(d, "raw", name)).read()
lscpu, cpuinfo, meminfo = read("lscpu.txt"), read("cpuinfo.txt"), read("meminfo.txt")
arch = re.search(r"Architecture:\s*(\S+)", lscpu)
cpus = re.search(r"^CPU\(s\):\s*(\d+)", lscpu, re.M)
mtype = re.search(r"machine\s*[=:]\s*(\w+)", cpuinfo)
flags = set()
for line in cpuinfo.splitlines():
    if line.split(":", 1)[0].strip() == "features" and ":" in line:
        flags.update(line.split(":", 1)[1].split())
mem = re.search(r"MemTotal:\s*(\d+)\s*kB", meminfo)
df = read("df.txt").splitlines()[-1].split()
containers = []
for line in read("docker_stats.jsonl").splitlines():
    try:
        s = json.loads(line)
    except ValueError:
        continue
    containers.append({"name": s.get("Name"), "cpu": s.get("CPUPerc"),
                       "mem": s.get("MemUsage", "").split(" /")[0], "status": "running"})
keep = ("Architecture", "CPU op-mode(s)", "Byte Order", "CPU(s)", "Vendor ID", "Model name",
        "Machine type", "Hypervisor vendor", "Virtualization type")
evidence = {
    "captured_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    "host": socket.gethostname(),
    "arch": arch.group(1) if arch else None,
    "cpus": int(cpus.group(1)) if cpus else None,
    "mem_total_gb": round(int(mem.group(1)) / 1024 ** 2, 1) if mem else None,
    "disk_total_gb": round(int(df[1]) / 1024 ** 3, 1) if len(df) > 1 and df[1].isdigit() else None,
    "machine_type": mtype.group(1) if mtype else None,
    "nnpa": "nnpa" in flags,               # AVAILABLE on the CPU; whether the model uses it is in /api/health
    "lscpu": "\n".join(l for l in lscpu.splitlines() if l.split(":")[0].strip() in keep),
    "docker_ps": read("docker_ps.txt").strip(),
    "containers": containers,
}
tmp = os.path.join(d, ".evidence.json.tmp")
with open(tmp, "w") as fh:
    json.dump(evidence, fh, indent=2)
os.replace(tmp, os.path.join(d, "evidence.json"))   # atomic: nginx never serves a half-written file
print(f"wrote {d}/evidence.json: {evidence['arch']} · machine {evidence['machine_type']} · "
      f"nnpa {'available' if evidence['nnpa'] else 'not in flags'} · {evidence['cpus']} vCPUs · "
      f"{evidence['mem_total_gb']} GB · {len(containers)} containers")
EOF
