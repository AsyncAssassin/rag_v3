# Scripts Map

Назначение скриптов разделено на три класса:

- `core`: ежедневный запуск, демо, воспроизводимый прогон.
- `ops/qa`: диагностика и проверка качества/устойчивости.
- `legacy/demo-external`: внешние/исторические сценарии, не обязательные для core-потока.

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
| `legacy/preflight_reviews_defense.sh` | legacy/demo-external | Preflight внешнего демо-стенда `RAG_over_reviews` | Только для внешнего стенда | Нет |
| `legacy/rehearsal_timeline_reviews.sh` | legacy/demo-external | Таймкод-репетиция внешней защиты | Только для внешнего стенда | Нет |
| `legacy/presentation_narration_ru.txt` | legacy/demo-external | Текст озвучки для внешнего сценария | Только с `--with-tts` | Нет |

## Core quick start

```bash
# локальный демо-поток
./scripts/demo_flow.sh --env .env

# полный readiness-прогон
python scripts/final_ready_run.py --env .env --data-dir data/demo
```
