#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE=".env"
ARTIFACTS_DIR="$ROOT_DIR/artifacts"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      ENV_FILE="$2"
      shift 2
      ;;
    --artifacts-dir)
      ARTIFACTS_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ARTIFACTS_DIR/ops_guard_run_$TS"
CMD_DIR="$RUN_DIR/cmd"
mkdir -p "$CMD_DIR"

WATCHDOG="$RUN_DIR/run_with_timeout.py"
cat > "$WATCHDOG" <<'PY'
#!/usr/bin/env python3
import argparse
import json
import subprocess
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--timeout-sec", type=int, required=True)
    ap.add_argument("--cwd", required=True)
    ap.add_argument("--stdout", required=True)
    ap.add_argument("--stderr", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise SystemExit("No command provided")

    out = Path(args.stdout)
    err = Path(args.stderr)
    meta = Path(args.meta)
    out.parent.mkdir(parents=True, exist_ok=True)
    err.parent.mkdir(parents=True, exist_ok=True)
    meta.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    timed_out = False
    status = "ok"
    exit_code = None

    with out.open("w", encoding="utf-8") as out_f, err.open("w", encoding="utf-8") as err_f:
        proc = subprocess.Popen(cmd, cwd=args.cwd, stdout=out_f, stderr=err_f, text=True)
        try:
            exit_code = proc.wait(timeout=max(1, int(args.timeout_sec)))
            status = "ok" if exit_code == 0 else "failed"
        except subprocess.TimeoutExpired:
            timed_out = True
            status = "timeout"
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except Exception:
                    pass
            exit_code = proc.returncode if proc.returncode is not None else -15

    payload = {
        "name": args.name,
        "status": status,
        "exit_code": int(exit_code) if exit_code is not None else None,
        "duration_sec": round(time.time() - started, 3),
        "timed_out": timed_out,
        "timeout_sec": int(args.timeout_sec),
        "command": cmd,
    }
    meta.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
chmod +x "$WATCHDOG"

run_step() {
  local name="$1"
  local timeout_sec="$2"
  shift 2
  "$WATCHDOG" \
    --name "$name" \
    --timeout-sec "$timeout_sec" \
    --cwd "$ROOT_DIR" \
    --stdout "$CMD_DIR/${name}.out" \
    --stderr "$CMD_DIR/${name}.err" \
    --meta "$CMD_DIR/${name}.meta.json" \
    -- "$@"
}

write_report() {
  python3 - <<'PY' "$RUN_DIR" "$1" "$2"
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
incident_status = sys.argv[2]
first_failure_step = sys.argv[3]
cmd_dir = run_dir / "cmd"

steps = ["selfcheck", "ask_in", "ask_out", "quality_gate"]
meta = {}
for step in steps:
    p = cmd_dir / f"{step}.meta.json"
    if p.exists():
        meta[step] = json.loads(p.read_text(encoding="utf-8"))
    else:
        meta[step] = None

payload = {
    "status": incident_status,
    "first_failure_step": first_failure_step if first_failure_step != "none" else None,
    "step_meta": meta,
    "evidence_hint": [
        "cmd/*.err",
        "cmd/*.meta.json",
        "cmd/selfcheck.out",
    ],
}
(run_dir / "ops_guard_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

lines = [
    "# Ops Guard Report",
    "",
    f"Status: **{incident_status}**",
    f"First failure step: `{payload['first_failure_step']}`",
    "",
    "## Evidence",
    "- cmd/*.err",
    "- cmd/*.meta.json",
    "- cmd/selfcheck.out",
]
(run_dir / "ops_guard_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"status": incident_status, "first_failure_step": payload["first_failure_step"]}, ensure_ascii=False))
PY
}

echo "ops_guard: selfcheck"
run_step selfcheck 240 "$ROOT_DIR/.venv/bin/python" -m rag_system.cli --env "$ENV_FILE" selfcheck || true
cp "$CMD_DIR/selfcheck.out" "$RUN_DIR/selfcheck.json" 2>/dev/null || true

self_meta_status="$(python3 - <<'PY' "$CMD_DIR/selfcheck.meta.json"
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.exists():
  print('missing')
else:
  print(json.loads(p.read_text(encoding='utf-8')).get('status','missing'))
PY
)"

selfcheck_ok="false"
if [[ -f "$RUN_DIR/selfcheck.json" ]]; then
  selfcheck_ok="$(python3 - <<'PY' "$RUN_DIR/selfcheck.json"
import json,sys
try:
  obj=json.loads(open(sys.argv[1],encoding='utf-8').read())
  print('true' if obj.get('ok') is True else 'false')
except Exception:
  print('false')
PY
)"
fi

if [[ "$self_meta_status" != "ok" || "$selfcheck_ok" != "true" ]]; then
  echo "ops_guard: stop at selfcheck"
  write_report "operational" "selfcheck"
  echo "Ops guard run completed: $RUN_DIR"
  exit 1
fi

echo "ops_guard: ask in/out"
run_step ask_in 180 "$ROOT_DIR/.venv/bin/python" -m rag_system.cli --env "$ENV_FILE" ask "Какие ключевые темы в отчете Сбера за 2024 год?" || true
run_step ask_out 180 "$ROOT_DIR/.venv/bin/python" -m rag_system.cli --env "$ENV_FILE" ask "Расскажи про ядерный реактор на Луне в отчете Сбера 2015." || true

ask_in_ok="$(python3 - <<'PY' "$CMD_DIR/ask_in.out"
import re,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.exists():
  print('false'); raise SystemExit
text=p.read_text(encoding='utf-8',errors='ignore')
m=re.search(r'"grounded_refusal"\s*:\s*(true|false)',text)
print('true' if (m and m.group(1)=='false' and 'source=' in text) else 'false')
PY
)"

ask_out_ok="$(python3 - <<'PY' "$CMD_DIR/ask_out.out"
import re,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.exists():
  print('false'); raise SystemExit
text=p.read_text(encoding='utf-8',errors='ignore')
m=re.search(r'"grounded_refusal"\s*:\s*(true|false)',text)
left=text.find('=== CITATIONS ===')
right=text.find('=== TRACE ===')
block=text[left:right] if left>=0 and right>left else text
print('true' if (m and m.group(1)=='true' and 'source=' not in block) else 'false')
PY
)"

if [[ "$ask_in_ok" != "true" ]]; then
  write_report "functional" "ask_in"
  echo "Ops guard run completed: $RUN_DIR"
  exit 1
fi
if [[ "$ask_out_ok" != "true" ]]; then
  write_report "functional" "ask_out"
  echo "Ops guard run completed: $RUN_DIR"
  exit 1
fi

echo "ops_guard: quality_gate"
run_step quality_gate 1200 "$ROOT_DIR/.venv/bin/python" -m rag_system.cli --env "$ENV_FILE" quality-gate --goldset artifacts/goldset_sber_qa.jsonl --reranker amberoad --metric-k 10 --min-recall 0.55 --min-mrr 0.45 --min-ndcg 0.50 --output "$RUN_DIR/quality_gate.json" || true

qg_ok="$(python3 - <<'PY' "$RUN_DIR/quality_gate.json"
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.exists():
  print('false'); raise SystemExit
try:
  obj=json.loads(p.read_text(encoding='utf-8'))
  print('true' if obj.get('ok') is True else 'false')
except Exception:
  print('false')
PY
)"

if [[ "$qg_ok" != "true" ]]; then
  write_report "functional" "quality_gate"
  echo "Ops guard run completed: $RUN_DIR"
  exit 1
fi

write_report "healthy" "none"
echo "Ops guard run completed: $RUN_DIR"
