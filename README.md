# Datathon ’26 · подготовка к 17–18 октября

Начни с [START_HERE.md](START_HERE.md). Окружение настроено на этом ноутбуке; кейс пока неизвестен.

```bash
cd '/home/zynqq/ai projects/datathon'
bash scripts/start_lab.sh
```

Jupyter напечатает локальную ссылку с токеном. Открой её в браузере. Kernel: **Datathon (Python 3.12)**. Интернет для установленных библиотек не нужен.

Учебный прогон:

```bash
bash scripts/run_practice.sh
```

Настоящий кейс:

```bash
cp configs/competition.example.yaml configs/competition.yaml
# Заполни data, target, features, categorical и split по заданию.
.venv/bin/datathon audit data/raw/train.csv --target ИМЯ_ЦЕЛИ
.venv/bin/datathon train --config configs/competition.yaml
```

Каждый прогон сохраняется в `reports/runs/<UTC-время>/`: конфигурация, SHA-256 файла данных, состав выборок, модель, качество Dummy и CatBoost, предсказания, важность признаков и график. Признаки указываются явно. Повторные попытки оптимизировать финальный holdout превращают его в ещё одну validation-выборку — для разработки смотри validation.

Файл на сдачу после согласования формата с организаторами:

```bash
.venv/bin/datathon predict \
  --run reports/runs/ИМЯ_ПРОГОНА \
  --data data/raw/test.csv --sample data/raw/sample_submission.csv \
  --output artifacts/submission.csv --prediction-column ИМЯ_КОЛОНКИ \
  --kind probability
```

`probability` — вероятность выбранного положительного класса; `label` — класс или численный прогноз для регрессии. Для multiclass-формата с несколькими колонками вероятностей потребуется адаптация. ID должны быть уникальны, множества ID входных данных и sample должны совпадать; порядок берётся из sample. Модель остаётся обученной только на train; переобучение на всех данных перед сдачей — отдельное осознанное действие после фиксации параметров и оценки.

Локальный просмотр экспериментов, если полезен:

```bash
.venv/bin/streamlit run scripts/run_report.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Проверка и восстановление:

```bash
.venv/bin/python scripts/doctor.py --gpu
.venv/bin/pytest -q
uv sync --locked --python 3.12  # восстановление с интернетом
uv sync --locked --offline    # на этом ноутбуке, пока сохранён uv cache
```

Python и окружение переносить копированием `.venv` между ОС не следует. На другой машине нужны uv, Python 3.12 и установка из `uv.lock` заранее. Точные версии сохранены в lock-файле; копия списка — `requirements.lock.txt`.

Набор поддерживает CSV/TSV/Parquet/XLSX и табличную классификацию/регрессию. Если кейс окажется про текст, изображения, поиск, рекомендации или чистую аналитику, сначала переформулируйте целевой результат; не подгоняйте задачу под этот baseline. Крупные CSV сначала исследуйте через DuckDB/Polars — pandas-аудит читает всю таблицу в RAM.

Презентации: `output/slides/forte-pitch-2min.pptx` и `output/slides/forte-pitch-full.pptx`. Резервные PDF находятся в `output/pdf/`. Это редактируемые шаблоны с явными полями для замены, без вымышленных результатов. Оформление вдохновлено Forte; шаблон не является официальным материалом банка.

Подробности: [план 24 часов](docs/PLAYBOOK.md), [разбор воркшопа](docs/WORKSHOP_REVIEW.md), [работа с ИИ](docs/AI_WORKFLOW.md), [подготовка питча](docs/PITCH.md), [карта требований](docs/REQUIREMENTS.md), [данные и проверка](docs/VALIDATION.md).
