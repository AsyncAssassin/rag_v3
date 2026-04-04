#!/usr/bin/env bash
set -euo pipefail

LIVE_MODE=0
FAST_MODE=0

usage() {
  cat <<'USAGE'
Usage: scripts/rehearsal_timeline_reviews.sh [options]

Options:
  --live     Print prompts in real-time according to 10-minute timeline.
  --fast     With --live, run in 10x speed (good for dry rehearsal).
  --help     Show help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --live)
      LIVE_MODE=1
      shift
      ;;
    --fast)
      FAST_MODE=1
      shift
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

print_static() {
  cat <<'TEXT'
Timeline (10 minutes):

00:00-00:30  Главная
  Клик: оставить стартовую страницу
  Реплика: «Покажу 3 критерия: рабочий RAG, web UI, advanced retrieval.»

00:30-02:00  О проекте
  Клик: вкладка «О проекте»
  Реплика: «Dense + MMR + BM25 + dedup + rerank.»

02:00-03:00  Отзывы / Общий объем
  Клик: вкладка «Отзывы», карточка «Общий объем»
  Реплика: «База и период подтягиваются live.»

03:00-04:00  Отзывы / Sber 5★
  Клик: карточка «Sber 5★»
  Реплика: «Фильтры + агрегаты по товарам.»

04:00-05:00  OOD-вопрос
  Клик: вставить вопрос про Луну и нажать «Спросить»
  Реплика: «Guardrail: нет данных в контексте -> отказ без галлюцинаций.»

05:00-06:30  Статистика
  Клик: вкладка «Статистика»
  Реплика: «UI закрывает чат и аналитический дашборд.»

06:30-07:30  Метрики качества поиска
  Клик: блок «Метрики качества поиска»
  Реплика: «Retrieval меряется отдельно, не на глаз.»

07:30-08:20  (Опц.) /api/search-metrics
  Клик: новая вкладка с JSON
  Реплика: «Низкий recall@k объясняется широкими запросами и малым k.»

08:20-09:00  (Опц.) /health и /openapi.json
  Клик: открыть URL
  Реплика: «Backend healthy, API контракт открыт.»

09:00-10:00  Финал
  Клик: возврат на главную
  Реплика: «Критерии закрыты. Готов к вопросам.»
TEXT
}

if [[ "$LIVE_MODE" -eq 0 ]]; then
  print_static
  exit 0
fi

starts=(0 30 120 180 240 300 390 450 500 540)
ends=(30 120 180 240 300 390 450 500 540 600)
labels=(
  "Главная"
  "О проекте"
  "Отзывы / Общий объем"
  "Отзывы / Sber 5★"
  "OOD-вопрос"
  "Статистика"
  "Метрики качества поиска"
  "Опц. /api/search-metrics"
  "Опц. /health + /openapi.json"
  "Финал"
)

scale="1.0"
if [[ "$FAST_MODE" -eq 1 ]]; then
  scale="0.1"
fi

echo "Live rehearsal started (scale=$scale)."
for i in "${!starts[@]}"; do
  s="${starts[$i]}"
  e="${ends[$i]}"
  label="${labels[$i]}"
  echo
  printf '[%02d:%02d-%02d:%02d] %s\n' "$((s/60))" "$((s%60))" "$((e/60))" "$((e%60))" "$label"

  duration="$((e-s))"
  sleep_for="$(python3 - <<'PY' "$duration" "$scale"
import sys
print(float(sys.argv[1]) * float(sys.argv[2]))
PY
)"
  sleep "$sleep_for"
done

echo
echo "Live rehearsal finished."
