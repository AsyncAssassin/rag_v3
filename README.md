# rag_v3

Продвинутая RAG-система поверх локальных документов с гибридным retrieval,
локальным rerank и web-интерфейсом на Streamlit.

## 1. Архитектура

Пайплайн:
1. **Ingestion / Indexing**
   - Поддерживаемые форматы: `.pdf`, `.txt`, `.csv`, `.md`.
   - Профили: `demo-fast`, `full-quality`.
   - Экстракция: Docling / PyMuPDF4LLM / Unstructured + fallback-policy.
2. **Retrieval**
   - Sparse: BM25.
   - Dense: embedding-поиск.
   - Fusion: RRF.
   - Дополнительно: source-diversity и year-aware boost.
3. **Rerank**
   - Cross-encoder rerank (`amberoad`, `bge_m3`, `jina_multilingual`).
4. **Generation + Guardrails**
   - Ответ через GigaChat.
   - Grounded refusal для out-of-domain сценариев.

Ключевые модули:
- `rag_system/pipeline.py` — оркестрация end-to-end.
- `rag_system/indexing.py` — построение/загрузка индекса.
- `rag_system/retrieval.py` — гибридный retrieval.
- `rag_system/rerankers/` — reranker-фабрика и ранжирование.
- `streamlit_app.py` — web UI.

## 2. Быстрый старт локально

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Обязательные переменные в `.env`:
- `GIGA_API_KEY`
- при необходимости `GIGA_SCOPE` (по умолчанию `GIGACHAT_API_B2B`)

Рекомендуемые demo-значения уже в `.env.example`:
- `DATA_DIR=data/demo`
- `INDEX_DIR=.rag_index`
- `RERANKER_CACHE_DIR=.rag_cache/rerankers`
- `DEFAULT_EXTRACTOR=pymupdf4llm`
- `INGEST_PROFILE=demo-fast`

## 3. Запуск фронта

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

В UI доступны поля/блоки, достаточные для защиты:
- параметры retrieval/rerank,
- `Selfcheck API`,
- `Preload rerankers`,
- индексация,
- вопрос/ответ,
- `Источники`,
- `Debug trace`,
- статусный блок `Готовность к запросу`.

## 4. Demo dataset для Cloud и smoke

В репозитории есть минимальный синтетический корпус:
- `data/demo/sber_2015_demo.md`
- `data/demo/sber_2024_demo.md`

Он используется как легкий и быстрый набор для:
- локального smoke,
- первого запуска на Streamlit Cloud.

## 5. Деплой на Streamlit Cloud

1. Открыть Streamlit Cloud и подключить репозиторий `AsyncAssassin/rag_v3`.
2. Указать:
   - Branch: `main`
   - Main file path: `streamlit_app.py`
3. В `Secrets` добавить минимум:
   - `GIGA_API_KEY="..."`
   - опционально `GIGA_SCOPE="GIGACHAT_API_B2B"`
4. После старта приложения выполнить в UI:
   - `Selfcheck API`
   - `Индексировать документы`
   - in-domain вопрос
   - out-of-domain вопрос

Fallback для ограничений Cloud:
- использовать `demo-fast` + `pymupdf4llm`;
- работать с `data/demo`.

## 6. Тестирование

### 6.1 Unit/contract

```bash
source .venv/bin/activate
pytest
```

### 6.2 CLI smoke

```bash
./scripts/demo_flow.sh --env .env
```

Ожидаемый контур:
- `selfcheck` OK,
- in-domain `ask` без refusal,
- out-of-domain `ask` с refusal,
- `quality-gate` по порогам.

### 6.3 UI smoke

```bash
python scripts/playwright_ui_demo.py --url http://localhost:8512 --output artifacts/ui_demo.json --screenshot artifacts/ui_demo.png
```

### 6.4 Release gate

Деплой считается валидным, если:
1. Проходит локальный smoke (`demo_flow.sh`),
2. Проходит cloud smoke (1 in-domain + 1 OOD запрос),
3. В UI корректно показываются `Источники` и `Debug trace`.

## 7. Карта скриптов

См. `scripts/README.md`.

Классы скриптов:
- `core` — основной рабочий контур,
- `ops/qa` — диагностические и quality-инструменты,
- `legacy/demo-external` — внешние/исторические сценарии.

## 8. Что не входит в core

В `scripts/legacy/` лежат сценарии, не обязательные для текущего
боевого потока `rag_v3`:
- `preflight_reviews_defense.sh`
- `rehearsal_timeline_reviews.sh`
- `presentation_narration_ru.txt`

Они сохранены как вспомогательные, но не участвуют в основном
локальном/Cloud запуске.
