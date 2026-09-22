# JustID machine learning

Student project (Saxion / JustID): two BERTje classifiers that read a Dutch court ruling and assign **rechtsgebieden** (law areas) and **bijzondere kenmerken** (procedure features). Both tasks are multi-label. A label is kept if its sigmoid score is at least 0.5.

Input is ruling text only. Output is two `{label: score}` maps.

Weights live on the Hugging Face Hub, not in this repo:

- [`newnus/justid-rechtsgebieden`](https://huggingface.co/newnus/justid-rechtsgebieden)
- [`newnus/justid-bijzondere-kenmerken`](https://huggingface.co/newnus/justid-bijzondere-kenmerken)

## Results

Balanced `test.jsonl` is the number for the report. `natural_test.jsonl` is a second hold-out with the real year / area mix.

| Task | Split | Micro-F1 | Macro-F1 |
|---|---|---|---|
| Rechtsgebieden | test | 0.96 | 0.73 |
| Rechtsgebieden | natural_test | 0.96 | 0.75 |
| Bijzondere kenmerken | test | 0.92 | 0.76 |
| Bijzondere kenmerken | natural_test | 0.93 | 0.76 |

Micro is high because frequent labels (Strafrecht, Bestuursrecht, Hoger beroep, …) are easy. Macro is lower because rare labels barely appear in training.

On the rechtsgebieden test set, **Burgerlijk procesrecht** (22 rows) and **Materieel strafrecht** (35 rows) have F1 0.00. On bijzondere kenmerken, **Proceskostenveroordeling** (37 rows) is 0.00 and **Tussenuitspraak** is 0.16. Full per-label tables are in `results/`.

Natural_test is slightly higher than the balanced test. The models did not overfit the stratified sample.

## How to run

Python 3.12. GPU is optional (RTX 3050 / CUDA 12.1 was used for training).

```powershell
cd justid-machinelearning
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Predict one ruling

```powershell
.\.venv\Scripts\python.exe predict.py
```

Paste text, then a blank line. First run downloads the two Hub models if `models/rg` and `models/bk` are missing.

Notebooks: `.\.venv\Scripts\python.exe -m jupyter lab notebooks`, pick the `.venv` kernel, run `01_load` → `02_clean` → `03_train` → `04_predict`. Do not run the two `trainer.train()` cells unless you want a multi-hour job.

### API

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs

- `POST /predict-text` — JSON `{"text": "..."}`
- `POST /predict-document` — PDF upload
- `GET /health`

### Docker

Needs Docker Desktop. Uses CPU torch. Local `models/` is mounted when present; otherwise the container pulls from the Hub.

```powershell
docker compose up --build
```

Same docs URL as above. Stop with `Ctrl+C` or `docker compose down`.

## Data

The ruling texts (`data/processed/*.jsonl`) are not in git. Notebooks `01` and `02` expect those files locally. Kept labels are listed in `data/processed/labels.json`. After cleaning: **65 316** rulings. Labels with fewer than 100 training examples are dropped: 20 rechtsgebieden and 25 bijzondere kenmerken remain.

## Layout

```
app.py                 FastAPI
predict.py             shared inference
config.py              paths, Hub ids, threshold
notebooks/             01 load → 02 clean → 03 train → 04 eval + predict
results/               eval.json + per-label CSV
models/                local weights (gitignored)
```

## Limits

- BERTje truncates at 512 tokens. Long rulings lose the tail.
- Rare labels stay weak at threshold 0.5. Per-label thresholds were not searched.
- PDF upload needs a text layer. Scanned pages return 400.
