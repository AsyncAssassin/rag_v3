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
RUN_DIR="$ARTIFACTS_DIR/demo_run_$TS"
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
  local out="$CMD_DIR/${name}.out"
  local err="$CMD_DIR/${name}.err"
  local meta="$CMD_DIR/${name}.meta.json"
  "$WATCHDOG" \
    --name "$name" \
    --timeout-sec "$timeout_sec" \
    --cwd "$ROOT_DIR" \
    --stdout "$out" \
    --stderr "$err" \
    --meta "$meta" \
    -- "$@"
}

echo "selfcheck: running"
run_step selfcheck 240 "$ROOT_DIR/.venv/bin/python" -m rag_system.cli --env "$ENV_FILE" selfcheck
cp "$CMD_DIR/selfcheck.out" "$RUN_DIR/selfcheck.json"

echo "ask_in: running"
run_step ask_in 300 "$ROOT_DIR/.venv/bin/python" -m rag_system.cli --env "$ENV_FILE" ask "Какие ключевые темы в отчете Сбера за 2024 год?"
cp "$CMD_DIR/ask_in.out" "$RUN_DIR/ask_in.txt"

echo "ask_out: running"
run_step ask_out 300 "$ROOT_DIR/.venv/bin/python" -m rag_system.cli --env "$ENV_FILE" ask "Расскажи про ядерный реактор на Луне в отчете Сбера 2015."
cp "$CMD_DIR/ask_out.out" "$RUN_DIR/ask_out.txt"

echo "quality_gate: running"
run_step quality_gate 1200 "$ROOT_DIR/.venv/bin/python" -m rag_system.cli --env "$ENV_FILE" quality-gate --goldset artifacts/goldset_sber_qa.jsonl --reranker amberoad --metric-k 10 --min-recall 0.55 --min-mrr 0.45 --min-ndcg 0.50 --output "$RUN_DIR/quality_gate.json"

python3 - <<'PY' "$RUN_DIR"
import json
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
cmd_dir = run_dir / "cmd"

checks = {}
errors = []

# step meta
for step in ("selfcheck", "ask_in", "ask_out", "quality_gate"):
    p = cmd_dir / f"{step}.meta.json"
    if not p.exists():
        checks[f"step_{step}_ok"] = False
        errors.append(f"missing {p.name}")
        continue
    meta = json.loads(p.read_text(encoding="utf-8"))
    checks[f"step_{step}_ok"] = meta.get("status") == "ok"

# selfcheck
selfcheck = json.loads((run_dir / "selfcheck.json").read_text(encoding="utf-8"))
checks["selfcheck_ok"] = bool(selfcheck.get("ok"))
by = {c.get("name"): bool(c.get("ok")) for c in selfcheck.get("checks", []) if isinstance(c, dict)}
checks["selfcheck_chat_ok"] = by.get("chat", False)
checks["selfcheck_embeddings_ok"] = by.get("embeddings", False)

# ask parsing
ask_in_text = (run_dir / "ask_in.txt").read_text(encoding="utf-8", errors="ignore")
ask_out_text = (run_dir / "ask_out.txt").read_text(encoding="utf-8", errors="ignore")

m_in = re.search(r'"grounded_refusal"\s*:\s*(true|false)', ask_in_text)
m_out = re.search(r'"grounded_refusal"\s*:\s*(true|false)', ask_out_text)

checks["ask_in_grounded_false"] = bool(m_in and m_in.group(1) == "false")
checks["ask_in_has_source"] = "source=" in ask_in_text
checks["ask_out_grounded_true"] = bool(m_out and m_out.group(1) == "true")

left = ask_out_text.find("=== CITATIONS ===")
right = ask_out_text.find("=== TRACE ===")
if left >= 0 and right > left:
    out_citations = ask_out_text[left:right]
else:
    out_citations = ask_out_text
checks["ask_out_no_source"] = "source=" not in out_citations

# quality gate
qg = json.loads((run_dir / "quality_gate.json").read_text(encoding="utf-8"))
retrieval = qg.get("retrieval", {})
queries = qg.get("queries", [])
checks["quality_gate_ok"] = bool(qg.get("ok"))
checks["quality_gate_thresholds_ok"] = (
    float(retrieval.get("mean_recall_at_k", -1)) >= 0.55
    and float(retrieval.get("mean_mrr", -1)) >= 0.45
    and float(retrieval.get("mean_ndcg_at_k", -1)) >= 0.50
)
checks["quality_gate_stage_entity_ok"] = (
    qg.get("metric_stage") == "retriever_candidates"
    and qg.get("metric_entity") == "source_path"
)
checks["quality_gate_queries_20"] = len(queries) == 20
checks["quality_gate_query_fields_ok"] = all(
    all(key in item for key in ("retrieved_ids_stage", "retrieved_sources_stage", "relevant_sources_stage"))
    for item in queries
)

status = "pass" if all(checks.values()) else "fail"

payload = {
    "status": status,
    "checks": checks,
    "errors": errors,
}
(run_dir / "demo_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

lines = ["# Demo Flow Result", "", f"Status: **{status.upper()}**", "", "## Checks"]
for key, value in checks.items():
    lines.append(f"- {key}: `{value}`")
if errors:
    lines += ["", "## Errors"] + [f"- {e}" for e in errors]
(run_dir / "demo_result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(json.dumps({"status": status}, ensure_ascii=False))
if status != "pass":
    raise SystemExit(1)
PY

echo "Demo run completed: $RUN_DIR"
