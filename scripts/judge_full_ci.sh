#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE=".env"
GOLDSET="artifacts/goldset_sber_qa.jsonl"
RERANKER="amberoad"
METRIC_K="10"
MIN_RECALL="0.55"
MIN_MRR="0.45"
MIN_NDCG="0.50"
PROVIDERS="anthropic,gigachat"
ANTHROPIC_MODEL="claude-sonnet-4-20250514"
GIGACHAT_MODEL=""
PER_QUERY_TIMEOUT_SEC="900"
MAX_RETRIES="2"
RETRY_BACKOFF_SEC="5"
MAX_RESUME_CYCLES="3"
RUN_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      ENV_FILE="$2"
      shift 2
      ;;
    --goldset)
      GOLDSET="$2"
      shift 2
      ;;
    --reranker)
      RERANKER="$2"
      shift 2
      ;;
    --metric-k)
      METRIC_K="$2"
      shift 2
      ;;
    --min-recall)
      MIN_RECALL="$2"
      shift 2
      ;;
    --min-mrr)
      MIN_MRR="$2"
      shift 2
      ;;
    --min-ndcg)
      MIN_NDCG="$2"
      shift 2
      ;;
    --providers)
      PROVIDERS="$2"
      shift 2
      ;;
    --anthropic-model)
      ANTHROPIC_MODEL="$2"
      shift 2
      ;;
    --gigachat-model)
      GIGACHAT_MODEL="$2"
      shift 2
      ;;
    --per-query-timeout-sec)
      PER_QUERY_TIMEOUT_SEC="$2"
      shift 2
      ;;
    --max-retries)
      MAX_RETRIES="$2"
      shift 2
      ;;
    --retry-backoff-sec)
      RETRY_BACKOFF_SEC="$2"
      shift 2
      ;;
    --max-resume-cycles)
      MAX_RESUME_CYCLES="$2"
      shift 2
      ;;
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'HELP'
Usage: scripts/judge_full_ci.sh [options]

Options:
  --env <path>                    Env file (default: .env)
  --goldset <path>                Goldset JSONL (default: artifacts/goldset_sber_qa.jsonl)
  --providers <csv>               Judge providers (default: anthropic,gigachat)
  --anthropic-model <name>        Anthropic judge model (default: claude-sonnet-4-20250514)
  --gigachat-model <name>         Optional GigaChat judge model override
  --per-query-timeout-sec <sec>   Timeout per single-query run (default: 900)
  --max-retries <n>               Runner retry count (default: 2)
  --retry-backoff-sec <sec>       Backoff between retries (default: 5)
  --max-resume-cycles <n>         Additional resume cycles after initial run (default: 3)
  --run-dir <path>                Fixed run dir (default: artifacts/judge_full_runner_ci_<ts>)
HELP
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$RUN_DIR" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  RUN_DIR="artifacts/judge_full_runner_ci_${TS}"
fi

mkdir -p "$ROOT_DIR/$RUN_DIR/cmd"
COMMANDS_LOG="$ROOT_DIR/$RUN_DIR/commands.log"

log() {
  local msg="$1"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$msg" | tee -a "$COMMANDS_LOG" >/dev/null
}

write_step_meta() {
  local step_name="$1"
  local status="$2"
  local exit_code="$3"
  local duration_sec="$4"
  local cmd="$5"
  local meta="$ROOT_DIR/$RUN_DIR/cmd/${step_name}.meta.json"
  python3 - <<'PY' "$meta" "$step_name" "$status" "$exit_code" "$duration_sec" "$cmd"
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
p.write_text(json.dumps({
    "name": sys.argv[2],
    "status": sys.argv[3],
    "exit_code": int(sys.argv[4]),
    "duration_sec": float(sys.argv[5]),
    "command": sys.argv[6],
}, ensure_ascii=False, indent=2), encoding='utf-8')
PY
}

PY_BIN="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$PY_BIN" ]]; then
  PY_BIN="python3"
fi

log "run_dir=$RUN_DIR"
log "precheck: validating env/index/runner availability"

PRECHECK_JSON="$ROOT_DIR/$RUN_DIR/precheck.json"
set +e
python3 - <<'PY' "$ROOT_DIR" "$ENV_FILE" "$PRECHECK_JSON"
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
env_path = root / sys.argv[2]
out_path = Path(sys.argv[3])

checks = {}
errors = []

def add(name: str, ok: bool, detail: str):
    checks[name] = {"ok": bool(ok), "detail": detail}
    if not ok:
        errors.append(f"{name}: {detail}")

add("env_exists", env_path.exists(), str(env_path))
if env_path.exists():
    text = env_path.read_text(encoding="utf-8", errors="ignore")
    for key in ("GIGA_API_KEY", "ANTHROPIC_API_KEY"):
        m = re.search(rf"^{key}=(.*)$", text, flags=re.MULTILINE)
        ok = bool(m and m.group(1).strip())
        add(f"{key}_set", ok, "present" if ok else "missing")
else:
    add("GIGA_API_KEY_set", False, "env file missing")
    add("ANTHROPIC_API_KEY_set", False, "env file missing")

meta = root / ".rag_index" / "meta.json"
chunks = root / ".rag_index" / "chunks.json"
add("index_meta_exists", meta.exists(), str(meta))
add("index_chunks_exists", chunks.exists(), str(chunks))
if chunks.exists():
    add("index_chunks_non_empty", chunks.stat().st_size > 2, f"size={chunks.stat().st_size}")
else:
    add("index_chunks_non_empty", False, "missing")

if meta.exists():
    try:
        meta_obj = json.loads(meta.read_text(encoding="utf-8"))
        fqb = meta_obj.get("file_quality_by_path")
        count = len(fqb) if isinstance(fqb, dict) else 0
        add("index_file_quality_present", count > 0, f"count={count}")
    except Exception as exc:
        add("index_file_quality_present", False, f"parse_error={type(exc).__name__}")
else:
    add("index_file_quality_present", False, "meta missing")

payload = {
    "ok": len(errors) == 0,
    "checks": checks,
    "errors": errors,
}
out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
raise SystemExit(0 if payload["ok"] else 1)
PY
PRECHECK_RC=$?
set -e

if [[ $PRECHECK_RC -ne 0 ]]; then
  log "precheck failed"
  cat "$PRECHECK_JSON" >&2
  python3 - <<'PY' "$ROOT_DIR/$RUN_DIR/ci_verdict.json" "$ROOT_DIR/$RUN_DIR/ci_verdict.md" "$RUN_DIR"
import json, sys
from pathlib import Path
j=Path(sys.argv[1]); m=Path(sys.argv[2]); run_dir=sys.argv[3]
payload={
  "status":"FAIL",
  "reason":"precheck_failed",
  "run_dir":run_dir,
}
j.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
m.write_text("# Judge Full CI Verdict\n\n- status: **FAIL**\n- reason: `precheck_failed`\n", encoding='utf-8')
PY
  exit 1
fi

set +e
"$PY_BIN" "$ROOT_DIR/scripts/judge_full_runner.py" --help >"$ROOT_DIR/$RUN_DIR/cmd/runner_help.out" 2>"$ROOT_DIR/$RUN_DIR/cmd/runner_help.err"
HELP_RC=$?
set -e
if [[ $HELP_RC -ne 0 ]]; then
  log "runner help failed"
  exit 1
fi

log "precheck passed"

CYCLE=0
FINAL_STATUS=""
FINAL_REASON=""
LATEST_EVAL_JSON="$ROOT_DIR/$RUN_DIR/eval_cycle.json"

while true; do
  STEP_NAME="judge_cycle_${CYCLE}"
  STEP_OUT="$ROOT_DIR/$RUN_DIR/cmd/${STEP_NAME}.out"
  STEP_ERR="$ROOT_DIR/$RUN_DIR/cmd/${STEP_NAME}.err"
  CMD=(
    "$PY_BIN" "$ROOT_DIR/scripts/judge_full_runner.py"
    --env "$ENV_FILE"
    --goldset "$GOLDSET"
    --reranker "$RERANKER"
    --metric-k "$METRIC_K"
    --min-recall "$MIN_RECALL"
    --min-mrr "$MIN_MRR"
    --min-ndcg "$MIN_NDCG"
    --providers "$PROVIDERS"
    --anthropic-model "$ANTHROPIC_MODEL"
    --per-query-timeout-sec "$PER_QUERY_TIMEOUT_SEC"
    --max-retries "$MAX_RETRIES"
    --retry-backoff-sec "$RETRY_BACKOFF_SEC"
    --run-dir "$RUN_DIR"
  )
  if [[ -n "$GIGACHAT_MODEL" ]]; then
    CMD+=(--gigachat-model "$GIGACHAT_MODEL")
  fi
  if [[ $CYCLE -gt 0 ]]; then
    CMD+=(--resume)
  fi

  log "cycle=$CYCLE running runner"
  START_TS="$(python3 - <<'PY'
import time
print(time.monotonic())
PY
)"
  set +e
  "${CMD[@]}" >"$STEP_OUT" 2>"$STEP_ERR"
  RC=$?
  set -e
  END_TS="$(python3 - <<'PY'
import time
print(time.monotonic())
PY
)"
  DURATION="$(python3 - <<'PY' "$START_TS" "$END_TS"
import sys
print(round(float(sys.argv[2]) - float(sys.argv[1]), 3))
PY
)"
  STEP_STATUS="ok"
  if [[ $RC -ne 0 ]]; then
    STEP_STATUS="failed"
  fi
  write_step_meta "$STEP_NAME" "$STEP_STATUS" "$RC" "$DURATION" "${CMD[*]}"

  set +e
  python3 - <<'PY' "$ROOT_DIR" "$RUN_DIR" "$LATEST_EVAL_JSON"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_dir = root / sys.argv[2]
out_json = Path(sys.argv[3])
comparison_path = run_dir / "judge_comparison.json"

retryable = {"timeout", "llm_did_not_finish", "rate_limit", "server_5xx", "network"}

res = {
    "comparison_exists": comparison_path.exists(),
    "strict_pass": False,
    "non_retryable_fail": False,
    "non_retryable_errors": [],
    "provider_progress": {},
    "overall_status": None,
    "judge_quality_warning": None,
}

if comparison_path.exists():
    obj = json.loads(comparison_path.read_text(encoding="utf-8"))
    res["overall_status"] = obj.get("overall_status")
    res["judge_quality_warning"] = obj.get("judge_quality_warning")
    providers = obj.get("providers") or {}
    expected_providers = ["anthropic", "gigachat"]
    strict = True
    for p in expected_providers:
      pv = providers.get(p) or {}
      status = pv.get("provider_status")
      completed = int(pv.get("completed_queries") or 0)
      res["provider_progress"][p] = {
          "provider_status": status,
          "completed_queries": completed,
      }
      if not (status == "complete" and completed == 20):
          strict = False
    res["strict_pass"] = strict

for provider in ("anthropic", "gigachat"):
    cp = run_dir / f"checkpoint_{provider}.jsonl"
    if not cp.exists():
        continue
    latest = {}
    for line in cp.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        qid = str(row.get("query_id") or "")
        if qid:
            latest[qid] = row
    for qid, row in latest.items():
        status = str(row.get("status") or "")
        err = str(row.get("error_class") or "")
        if status == "failed" and err not in retryable:
            res["non_retryable_fail"] = True
            res["non_retryable_errors"].append({"provider": provider, "query_id": qid, "error_class": err})
        if err.startswith("output_parse_error"):
            res["non_retryable_fail"] = True
            res["non_retryable_errors"].append({"provider": provider, "query_id": qid, "error_class": err})

out_json.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
raise SystemExit(0)
PY
  EVAL_RC=$?
  set -e

  if [[ $EVAL_RC -ne 0 ]]; then
    FINAL_STATUS="FAIL"
    FINAL_REASON="evaluation_parser_failed"
    break
  fi

  STRICT_PASS="$(python3 - <<'PY' "$LATEST_EVAL_JSON"
import json, sys
obj=json.loads(open(sys.argv[1],encoding='utf-8').read())
print('true' if obj.get('strict_pass') else 'false')
PY
)"
  NON_RETRY_FAIL="$(python3 - <<'PY' "$LATEST_EVAL_JSON"
import json, sys
obj=json.loads(open(sys.argv[1],encoding='utf-8').read())
print('true' if obj.get('non_retryable_fail') else 'false')
PY
)"

  if [[ "$STRICT_PASS" == "true" ]]; then
    FINAL_STATUS="PASS"
    FINAL_REASON="strict_complete_20x20"
    break
  fi

  if [[ "$NON_RETRY_FAIL" == "true" ]]; then
    FINAL_STATUS="FAIL"
    FINAL_REASON="non_retryable_error"
    break
  fi

  if [[ $CYCLE -ge $MAX_RESUME_CYCLES ]]; then
    FINAL_STATUS="INCONCLUSIVE"
    FINAL_REASON="incomplete_after_resume_limit"
    break
  fi

  CYCLE=$((CYCLE + 1))
  log "cycle incomplete; scheduling resume cycle=$CYCLE"
  sleep "$RETRY_BACKOFF_SEC"
done

python3 - <<'PY' "$ROOT_DIR" "$RUN_DIR" "$FINAL_STATUS" "$FINAL_REASON" "$CYCLE" "$MAX_RESUME_CYCLES" "$LATEST_EVAL_JSON"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_dir = root / sys.argv[2]
status = sys.argv[3]
reason = sys.argv[4]
cycle = int(sys.argv[5])
max_cycles = int(sys.argv[6])
eval_path = Path(sys.argv[7])
comparison_path = run_dir / "judge_comparison.json"

comparison = None
if comparison_path.exists():
    try:
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    except Exception:
        comparison = None

eval_obj = None
if eval_path.exists():
    try:
        eval_obj = json.loads(eval_path.read_text(encoding="utf-8"))
    except Exception:
        eval_obj = None

payload = {
    "status": status,
    "reason": reason,
    "run_dir": str(run_dir),
    "cycles_used": cycle,
    "max_resume_cycles": max_cycles,
    "comparison_path": str(comparison_path),
    "comparison_exists": comparison_path.exists(),
    "judge_quality_warning": (comparison or {}).get("judge_quality_warning") if comparison else None,
    "provider_progress": (eval_obj or {}).get("provider_progress") if eval_obj else None,
    "non_retryable_errors": (eval_obj or {}).get("non_retryable_errors") if eval_obj else None,
}

(run_dir / "ci_verdict.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

md = [
    "# Judge Full CI Verdict",
    "",
    f"- status: **{status}**",
    f"- reason: `{reason}`",
    f"- cycles_used: `{cycle}`",
    f"- max_resume_cycles: `{max_cycles}`",
]
if payload["judge_quality_warning"]:
    md.append(f"- judge_quality_warning: `{payload['judge_quality_warning']}`")
if payload["provider_progress"]:
    md.append("\n## Provider Progress")
    for name, info in payload["provider_progress"].items():
        md.append(f"- {name}: status=`{info.get('provider_status')}`, completed=`{info.get('completed_queries')}`")
if payload["non_retryable_errors"]:
    md.append("\n## Non-retryable Errors")
    for row in payload["non_retryable_errors"]:
        md.append(f"- {row}")
(run_dir / "ci_verdict.md").write_text("\n".join(md) + "\n", encoding="utf-8")
PY

log "final_status=$FINAL_STATUS reason=$FINAL_REASON"
log "artifacts=$RUN_DIR"

case "$FINAL_STATUS" in
  PASS)
    exit 0
    ;;
  INCONCLUSIVE)
    exit 2
    ;;
  *)
    exit 1
    ;;
esac
