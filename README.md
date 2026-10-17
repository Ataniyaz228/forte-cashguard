# CashGuard · ForteBank Datathon ’26

Локальный AI-сервис: загружаете видео кассы — получаете вердикт, события нарушений с таймкодом, кадрами-доказательствами, описанием и пунктом норматива РК. Результат — подсказка для проверки человеком, не юридическое заключение. Все модели работают на нашей машине, облачные AI-API не используются.

Подробно: [docs/SUBMISSION.md](docs/SUBMISSION.md) (архитектура, модели, нормы, типы, ограничения, метрики). Разбор данных: [docs/VIDEO_REVIEW.md](docs/VIDEO_REVIEW.md). Журнал экспериментов: [docs/EXPERIMENTS.csv](docs/EXPERIMENTS.csv).

## Запуск

```bash
uv sync                                         # Python 3.12, CUDA-сборка PyTorch
echo "CASHGUARD_API_KEYS=<ключ>" >> .env        # ключ для жюри, в git не попадает
bash deploy/serve_loop.sh                       # сервис на :8090 с автоперезапуском
cloudflared tunnel --config .build/empty-cloudflared.yml --url http://127.0.0.1:8090
```

Проверка без сервера: `.venv/bin/python -m datathon.cashguard analyze video.mov --no-images`.

## API (раздел 10 ТЗ)

```bash
curl -X POST <URL>/api/v1/analyze -H "X-API-Key: <ключ>" -F "file=@video.mov"
curl <URL>/api/v1/jobs/<job_id> -H "X-API-Key: <ключ>"
```

## Разработка

```bash
.venv/bin/pytest -q && .venv/bin/ruff check src scripts tests
.venv/bin/python scripts/dev_eval.py --probe          # оценка на обучающих видео (локально)
.venv/bin/python scripts/jury_check.py <URL> v1.mov v2.mov --parallel 5   # имитация жюри
.venv/bin/python scripts/robustness.py               # устойчивость к другому ракурсу
```

Данные дататона (`data/raw`, разметка `data/annotations`) в репозиторий не входят и не публикуются.
