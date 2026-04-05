# Scripts Map

Назначение скриптов разделено на три класса:

- `core`: ежедневный запуск, демо, воспроизводимый прогон.
- `ops/qa`: диагностика и проверка качества/устойчивости.

| Script | Class | Зачем | Когда запускать | Обязателен для защиты |
|---|---|---|---|---|
| `demo_flow.sh` | core | Быстрый end-to-end демо-контур (`selfcheck -> ask -> quality-gate`) | Перед демо/проверкой | Да |
| `final_ready_run.py` | core | Полный оркестратор готовности с авто-вердиктом | Перед финальной сдачей | Да |
| `ops_guard.sh` | core | Короткий operational guard с ранним fail | Если есть риск деградации API | Да |
| `record_presentation_live.sh` | core | Запись скринкаста с terminal/UI этапами | При подготовке видео | Нет |
| `capture_ui.py` | core | Автоскриншот Streamlit UI | Для артефактов/пруфов | Нет |
| `playwright_ui_demo.py` | core | UI smoke-сценарий через Playwright | Перед записью/релизом | Нет |
| `pdf_regression.py` | core | Regression по извлечению PDF | После изменения extraction | Нет |
| `judge_full_runner.py` | ops/qa | Full-run сравнение judge-провайдеров | Глубокая проверка качества | Нет |
| `judge_full_ci.sh` | ops/qa | CI-обертка над full judge runner | Для CI/ночных прогонов | Нет |

## Judge CI (`judge_full_ci.sh`)

Назначение:
- прогоняет full judge-проверку через `judge_full_runner.py`;
- выполняет precheck окружения и индекса;
- умеет автоматически делать `--resume` циклы;
- формирует machine-readable и human-readable verdict.

Когда запускать:
- ночные/регулярные quality-прогоны;
- перед релизом, если нужен строгий verdict по judge-метрикам;
- после изменений в retrieval/rerank/pipeline, влияющих на качество ответов.

Prerequisites:
- заполненный `.env` с `GIGA_API_KEY` и `ANTHROPIC_API_KEY`;
- построенный индекс `.rag_index/meta.json` и `.rag_index/chunks.json`;
- goldset JSONL (по умолчанию `artifacts/goldset_sber_qa.jsonl`);
- установленное окружение зависимостей (`.venv` или `python3` с нужными пакетами).

Базовый запуск:

```bash
./scripts/judge_full_ci.sh --env .env
```

Типичный расширенный запуск:

```bash
./scripts/judge_full_ci.sh \
  --env .env \
  --goldset artifacts/goldset_sber_qa.jsonl \
  --providers anthropic,gigachat \
  --per-query-timeout-sec 900 \
  --max-retries 2 \
  --max-resume-cycles 3 \
  --run-dir artifacts/judge_full_runner_ci_manual
```

Ключевые флаги:
- `--providers <csv>`: провайдеры-судьи (default: `anthropic,gigachat`);
- `--per-query-timeout-sec <sec>`: таймаут одного запроса;
- `--max-retries <n>` и `--retry-backoff-sec <sec>`: retry-поведение;
- `--max-resume-cycles <n>`: количество дополнительных resume-циклов;
- `--run-dir <path>`: фиксированный каталог артефактов.

Артефакты в `run-dir`:
- `ci_verdict.json` и `ci_verdict.md`: итоговый verdict;
- `commands.log`: журнал шагов;
- `precheck.json`: результат precheck;
- `cmd/*.meta.json`: метаданные шагов;
- `cmd/*.out`, `cmd/*.err`: stdout/stderr каждого цикла.

Exit-коды:
- `0` — PASS;
- `1` — FAIL;
- `2` — INCONCLUSIVE (не удалось строго завершить за лимит resume-циклов).

## Core quick start

```bash
# локальный демо-поток
./scripts/demo_flow.sh --env .env

# полный readiness-прогон
python scripts/final_ready_run.py --env .env --data-dir data/demo
```
