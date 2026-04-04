# Demo Dataset

Папка `data/demo` содержит минимальный синтетический набор документов
для быстрого запуска и деплоя фронта в Streamlit Cloud.

Рекомендуемый сценарий:
1. `Selfcheck API`
2. `Индексировать документы` (Extractor: `pymupdf4llm`, Profile: `demo-fast`)
3. `Спросить` in-domain вопрос по 2015/2024
4. `Спросить` out-of-domain вопрос для проверки refusal
