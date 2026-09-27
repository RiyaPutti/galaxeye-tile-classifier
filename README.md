# Offline Tile Classifier — thin slice

Part 2 of the GalaxEye take-home. One endpoint, end to end: upload a tile,
classify it with a model trained locally, store the result, return it.
Everything else is either stubbed or just not built — listed at the bottom.

## What's here

```
config.yaml / config.py   all paths/thresholds live here, nothing hardcoded elsewhere
features.py               turns a tile into a feature vector (color histogram + texture)
train.py                  trains a RandomForest on candidate_tiles/, saves artifacts/classifier.joblib
storage.py                SQLite — stores successes AND failed/corrupt uploads
main.py                   FastAPI app: POST /tiles/classify, GET /tiles, GET /health
test_main.py              pytest suite against the real API + real model (not mocked)
check_confidence.py       standalone script I used to check the confidence threshold is doing real work
requirements.txt
design note.docx          Part 1
part3 answers.docx        Part 3
```

## Setup

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Needs the dataset unzipped as a sibling folder to this one:
```
be-mlsys-assignment-dataset/
  candidate_tiles/<ClassName>/*.png
  eval_set/*.png
  eval_labels.csv
submission/        <- this folder
```
If your layout's different, the path is `training.candidate_tiles_dir` in
`config.yaml`.

## Train the classifier

```powershell
python train.py
```

Reads every class folder under `candidate_tiles/` (doesn't hardcode the
class list — reads it off the folder names, so adding a class later is
just adding a folder), extracts features, does an 85/15 train/validation
split, trains a RandomForest, and saves the model + label encoder to
`artifacts/classifier.joblib`. Takes a few seconds on CPU. No GPU, no
internet, no pretrained weights to download — that was the whole point of
picking hand-built features over a CNN.

On my own validation split: 83% accuracy. Weakest classes are Highway and
River (both around 0.6 f1) — they're both thin, linear features that are
genuinely hard to tell apart with color+texture at 64×64. Not tuned
further past that — the brief says accuracy isn't graded.

## Run the API

```powershell
uvicorn main:app --reload
```

Then, from a second terminal (same venv activated):

```powershell
curl.exe -X POST http://127.0.0.1:8000/tiles/classify -F "file=@../be-mlsys-assignment-dataset/eval_set/tile_001.png"
```

```json
{
  "id": 1,
  "filename": "tile_001.png",
  "predicted_label": "Forest",
  "confidence": 0.985,
  "class_scores": {
    "AnnualCrop": 0.005, "Forest": 0.985, "Highway": 0.0,
    "Industrial": 0.0, "Residential": 0.0, "River": 0.005, "SeaLake": 0.005
  },
  "needs_review": false
}
```

The full per-class probability distribution is stored along with the
winning label — if a prediction looks wrong later, this tells you whether
it was a close call or way off, without re-running anything. A tile that
fails to decode gets stored too, as a `status="failed"` row with an error
message, instead of just erroring out and losing the event.

Query what's been classified so far:
```powershell
curl.exe "http://127.0.0.1:8000/tiles?needs_review=true"
```

You can also just hit `http://127.0.0.1:8000/docs` in a browser while
`uvicorn` is running — FastAPI generates an interactive UI for free, no
extra code, good for a quick demo without typing curl commands.

## Checking the confidence threshold is actually doing something

```powershell
python check_confidence.py
```

Runs the model against GalaxEye's `eval_set` (never touched during
training) and compares accuracy on the flagged low-confidence tiles vs.
everything else. On my run: 83.8% overall (176/210), but only 47.1%
(16/34) on the tiles flagged `needs_review`. So the flag is real — it's
picking out the tiles that are close to a coin flip, not just a number
nobody looks at.

## Run the tests

```powershell
pytest test_main.py -v
```

Hits the real running model, not a mock — five tests: health check, a
real classification on an eval_set tile, rejecting a corrupt upload,
checking a stored result is actually queryable, and filtering by label.
5/5 pass for me.

## What's stubbed / not built

- **Batching.** The endpoint takes one tile per request. No directory
  watcher, no bulk upload.
- **A real query language.** `GET /tiles` only does equality filters on
  label and `needs_review`, plus a limit. No confidence-range queries, no
  date ranges, no spatial queries — even though these are satellite tiles
  and a location-based query is probably the first thing a real analyst
  would want. Depends on metadata I wasn't given with this dataset.
- **Retraining / feedback loop, monitoring, drift detection.** Reasoned
  about these in the design note, didn't build any of it.
- **Auth.** None at all. Fine for a single-operator offline deployment,
  not fine the moment that assumption changes.