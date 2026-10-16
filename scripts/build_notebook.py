"""Build a small, executable offline practice notebook."""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md(
        "# Datathon · стартовая точка\n\nСначала прочитайте START_HERE.md и brief. "
        "Этот notebook запускается на **учебных синтетических данных**, не на кейсе ForteBank. "
        "Результаты нельзя использовать как соревновательные. Kernel: Datathon (Python 3.12)."
    ),
    code(
        "from pathlib import Path\nimport os\nimport sys\n\n"
        "root = Path.cwd()\nif root.name == 'notebooks':\n    root = root.parent\n"
        "assert (root / 'pyproject.toml').exists(), 'Откройте notebook из папки datathon'\n"
        "os.chdir(root)\nprint('Project:', root)\nprint('Python:', sys.executable)"
    ),
    md(
        "## 1. Создать учебные данные\n\nДля настоящего кейса замените путь на data/raw и "
        "сначала проверьте правила подготовленного кода, API и внешних данных."
    ),
    code(
        "import subprocess\nsubprocess.run([sys.executable, 'scripts/make_synthetic.py'], check=True)"
    ),
    md(
        "## 2. Описательный аудит\n\nПолный аудит показывает схему и качество. Связи с target, "
        "сегменты и отбор признаков исследуйте только на train/development после разбиения."
    ),
    code(
        "from datathon.eda import audit\nfrom datathon.data import load_table\n\n"
        "DATA = 'data/synthetic/train.csv'\nreport = audit(DATA, target='target', output='reports/eda/notebook')\n"
        "frame = load_table(DATA)\nframe.shape, report"
    ),
    md(
        "## 3. Зафиксировать разбиение и доступные признаки\n\n"
        "Time — прогноз будущего. Group — новые сущности. Random — независимые объекты. "
        "В шаблоне используется group; одинаковый клиент не попадёт по обе стороны."
    ),
    code(
        "import yaml\nfrom datathon.splits import make_split\n\n"
        "CONFIG = 'configs/synthetic.yaml'\ncfg = yaml.safe_load(Path(CONFIG).read_text())\n"
        "train_idx, valid_idx, holdout_idx = make_split(frame, cfg)\n"
        "print({name: len(rows) for name, rows in [('train', train_idx), ('validation', valid_idx), ('holdout', holdout_idx)]})\n"
        "print('Features:', cfg['features'])"
    ),
    md(
        "## 4. Один воспроизводимый прогон\n\nEarly stopping и порог выбираются на validation; "
        "итоговые метрики — на отдельном holdout. Не используйте этот holdout для цикла оптимизации."
    ),
    code("from datathon.train import train\nrun = train(CONFIG)\nprint(run)"),
    code(
        "import json\nimport pandas as pd\n\nmetrics = json.loads((run / 'metrics.json').read_text())\n"
        "pd.DataFrame({'Dummy': metrics['baseline_holdout'], 'CatBoost': metrics['holdout']})"
    ),
    md(
        "## 5. Предсказания в порядке sample_submission\n\nID сверяются и выравниваются явно. "
        "Тип результата должен точно соответствовать формату организаторов."
    ),
    code(
        "from datathon.predict import predict\npredict(run, 'data/synthetic/test.csv', "
        "'data/synthetic/sample_submission.csv', 'artifacts/practice_submission.csv', 'target', "
        "kind='probability')\npd.read_csv('artifacts/practice_submission.csv').head()"
    ),
    md(
        "## 6. Перенос на настоящий кейс\n\nСкопируйте configs/competition.example.yaml. "
        "Заполните цель, allowlist признаков и split по brief. Метрику, название положительного класса "
        "и цену ошибки проверьте отдельно. Если задача не табличная, измените постановку и baseline. "
        "Следующий материал: docs/AI_WORKFLOW.md и docs/REQUIREMENTS.md."
    ),
]
nb.metadata = {
    "kernelspec": {
        "display_name": "Datathon (Python 3.12)",
        "language": "python",
        "name": "datathon",
    },
    "language_info": {"name": "python", "version": "3.12"},
}
Path("notebooks").mkdir(exist_ok=True)
nbf.write(nb, "notebooks/00_launchpad.ipynb")
print("Created notebooks/00_launchpad.ipynb")
