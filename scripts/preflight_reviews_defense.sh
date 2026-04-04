#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_URL="https://ldy55.tw1.ru/RAG_over_reviews"
ARTIFACTS_DIR="$ROOT_DIR/artifacts"
TIMEOUT_SEC=45
STRICT_EXPECTED=0

EXPECT_TOTAL=58919
EXPECT_PERIOD_START="01-11-2024"
EXPECT_PERIOD_END="08-03-2026"
EXPECT_BENCHMARK=13

usage() {
  cat <<'USAGE'
Usage: scripts/preflight_reviews_defense.sh [options]

Options:
  --base-url <url>             Base URL for the demo stand.
  --artifacts-dir <path>       Artifacts root dir (default: <repo>/artifacts).
  --timeout-sec <int>          HTTP timeout in seconds (default: 45).
  --strict-expected            Fail if expected snapshot values changed.
  --expect-total <int>         Expected total review count (default: 58919).
  --expect-period-start <str>  Expected period start (default: 01-11-2024).
  --expect-period-end <str>    Expected period end (default: 08-03-2026).
  --expect-benchmark <int>     Expected benchmark size (default: 13).
  --help                       Show help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    --artifacts-dir)
      ARTIFACTS_DIR="$2"
      shift 2
      ;;
    --timeout-sec)
      TIMEOUT_SEC="$2"
      shift 2
      ;;
    --strict-expected)
      STRICT_EXPECTED=1
      shift
      ;;
    --expect-total)
      EXPECT_TOTAL="$2"
      shift 2
      ;;
    --expect-period-start)
      EXPECT_PERIOD_START="$2"
      shift 2
      ;;
    --expect-period-end)
      EXPECT_PERIOD_END="$2"
      shift 2
      ;;
    --expect-benchmark)
      EXPECT_BENCHMARK="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing command: $1" >&2
    exit 1
  fi
}

require_cmd curl
require_cmd jq
require_cmd python3

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ARTIFACTS_DIR/defense_preflight_$TS"
mkdir -p "$RUN_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

fetch_get() {
  local path="$1"
  local out="$2"
  curl --http1.1 -sS -f --max-time "$TIMEOUT_SEC" \
    "$BASE_URL$path" > "$out"
}

fetch_post_json() {
  local path="$1"
  local payload="$2"
  local out="$3"
  curl --http1.1 -sS -f --max-time "$TIMEOUT_SEC" \
    -X POST "$BASE_URL$path" \
    -H 'Content-Type: application/json' \
    -d "$payload" > "$out"
}

log "Preflight start: $BASE_URL"

log "GET /health"
fetch_get "/health" "$RUN_DIR/health.json"

log "GET /api/dashboard-stats"
fetch_get "/api/dashboard-stats" "$RUN_DIR/dashboard_stats.json"

log "GET /api/search-metrics"
fetch_get "/api/search-metrics" "$RUN_DIR/search_metrics.json"

log "POST /api/ask (volume)"
fetch_post_json "/api/ask" '{"question":"Сколько отзывов в базе? За какой период собраны отзывы?"}' "$RUN_DIR/ask_volume.json"

log "POST /api/ask (Sber 5)"
fetch_post_json "/api/ask" '{"question":"Сколько отзывов у рейтинга 5 по бренду Sber? Покажи статистику по товарам."}' "$RUN_DIR/ask_sber5.json"

log "POST /api/ask (OOD)"
fetch_post_json "/api/ask" '{"question":"Расскажи про ядерный реактор на Луне в отзывах на колонки"}' "$RUN_DIR/ask_ood.json"

python3 - <<'PY' \
  "$RUN_DIR" "$BASE_URL" "$STRICT_EXPECTED" \
  "$EXPECT_TOTAL" "$EXPECT_PERIOD_START" "$EXPECT_PERIOD_END" "$EXPECT_BENCHMARK"
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

run_dir = Path(sys.argv[1])
base_url = sys.argv[2]
strict_expected = sys.argv[3] == "1"
expect_total = int(sys.argv[4])
expect_period_start = sys.argv[5]
expect_period_end = sys.argv[6]
expect_benchmark = int(sys.argv[7])

def load(name: str):
    return json.loads((run_dir / name).read_text(encoding="utf-8"))

health = load("health.json")
dashboard = load("dashboard_stats.json")
search_metrics = load("search_metrics.json")
ask_volume = load("ask_volume.json")
ask_sber5 = load("ask_sber5.json")
ask_ood = load("ask_ood.json")

overview = dashboard.get("overview", {})
quality = dashboard.get("search_quality", {})
quality_metrics = quality.get("metrics", {})
search_metrics_values = search_metrics.get("metrics", {})

checks = {}
notes = []

checks["health_status_ok"] = health.get("status") == "ok"
checks["health_details_ready"] = str(health.get("details", "")).lower() == "ready"

checks["dashboard_total_reviews_positive"] = int(overview.get("total_reviews", 0)) > 0
checks["dashboard_period_present"] = bool(overview.get("period_start")) and bool(overview.get("period_end"))
checks["dashboard_quality_block_present"] = isinstance(quality_metrics, dict) and len(quality_metrics) > 0

checks["search_metrics_benchmark_positive"] = int(search_metrics.get("benchmark_size", 0)) > 0
checks["search_metrics_query_level_present"] = isinstance(search_metrics.get("queries"), list) and len(search_metrics.get("queries")) > 0
checks["search_metrics_has_precision"] = "precision_at_10" in search_metrics_values
checks["search_metrics_has_ndcg"] = "ndcg_at_10" in search_metrics_values

volume_answer = str(ask_volume.get("answer", ""))
sber5_answer = str(ask_sber5.get("answer", ""))
ood_answer = str(ask_ood.get("answer", "")).lower()

checks["ask_volume_has_total"] = "Всего отзывов" in volume_answer
checks["ask_volume_has_period"] = "период" in volume_answer.lower()
checks["ask_sber5_has_filter"] = "Brand: Sber" in sber5_answer
checks["ask_sber5_has_totals"] = "Всего отзывов" in sber5_answer
checks["ask_ood_refusal"] = "нет информации" in ood_answer

snapshot = {
    "total_reviews": overview.get("total_reviews"),
    "period_start": overview.get("period_start"),
    "period_end": overview.get("period_end"),
    "benchmark_size": quality.get("benchmark_size"),
    "corpus_size": quality.get("corpus_size"),
    "precision_at_10": quality_metrics.get("precision_at_10"),
    "recall_at_10": quality_metrics.get("recall_at_10"),
    "mrr": quality_metrics.get("mrr"),
    "ndcg_at_10": quality_metrics.get("ndcg_at_10"),
}

if strict_expected:
    checks["expected_total_reviews_match"] = snapshot["total_reviews"] == expect_total
    checks["expected_period_start_match"] = snapshot["period_start"] == expect_period_start
    checks["expected_period_end_match"] = snapshot["period_end"] == expect_period_end
    checks["expected_benchmark_match"] = snapshot["benchmark_size"] == expect_benchmark
else:
    if snapshot["total_reviews"] != expect_total:
        notes.append(
            f"Snapshot changed: total_reviews={snapshot['total_reviews']} (expected {expect_total})."
        )
    if snapshot["period_start"] != expect_period_start or snapshot["period_end"] != expect_period_end:
        notes.append(
            f"Snapshot changed: period={snapshot['period_start']}..{snapshot['period_end']} "
            f"(expected {expect_period_start}..{expect_period_end})."
        )
    if snapshot["benchmark_size"] != expect_benchmark:
        notes.append(
            f"Snapshot changed: benchmark_size={snapshot['benchmark_size']} (expected {expect_benchmark})."
        )

status = "pass" if all(checks.values()) else "fail"

payload = {
    "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "base_url": base_url,
    "status": status,
    "strict_expected": strict_expected,
    "checks": checks,
    "snapshot": snapshot,
    "notes": notes,
}

(run_dir / "defense_preflight_report.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
)

lines = [
    "# Defense Preflight Report",
    "",
    f"- Base URL: `{base_url}`",
    f"- Status: **{status.upper()}**",
    f"- Strict expected: `{strict_expected}`",
    "",
    "## Snapshot",
    f"- Total reviews: `{snapshot['total_reviews']}`",
    f"- Period: `{snapshot['period_start']}` -> `{snapshot['period_end']}`",
    f"- Benchmark size: `{snapshot['benchmark_size']}`",
    f"- Corpus size: `{snapshot['corpus_size']}`",
    f"- Precision@10: `{snapshot['precision_at_10']}`",
    f"- Recall@10: `{snapshot['recall_at_10']}`",
    f"- MRR: `{snapshot['mrr']}`",
    f"- nDCG@10: `{snapshot['ndcg_at_10']}`",
    "",
    "## Checks",
]

for key, value in checks.items():
    mark = "PASS" if value else "FAIL"
    lines.append(f"- {key}: `{mark}`")

if notes:
    lines.extend(["", "## Notes"])
    for note in notes:
        lines.append(f"- {note}")

(run_dir / "defense_preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(json.dumps({"status": status, "run_dir": str(run_dir)}, ensure_ascii=False))
if status != "pass":
    raise SystemExit(1)
PY

log "Preflight done: $RUN_DIR"
