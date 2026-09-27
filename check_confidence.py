"""
Checks whether the low-confidence flag is doing real work: are tiles below
the threshold actually wrong more often than tiles above it?

Run from submission/, with eval_set/ and eval_labels.csv present as
described in config.yaml.
"""
import csv
import glob

import joblib # type: ignore
from PIL import Image # type: ignore

from config import config
from features import extract_features

artifact = joblib.load(config.model_artifact_path)
clf, encoder = artifact["model"], artifact["label_encoder"]

eval_set_dir = config.candidate_tiles_dir.parent / "eval_set"
labels_csv = config.candidate_tiles_dir.parent / "eval_labels.csv"

true_labels = {}
with open(labels_csv) as f:
    for row in csv.DictReader(f):
        # adjust key names below if your CSV header differs
        true_labels[row["filename"]] = row["true_label"]

correct, total = 0, 0
low_conf_correct, low_conf_total = 0, 0

for path in sorted(glob.glob(str(eval_set_dir / "*.png"))):
    fname = path.split("\\")[-1].split("/")[-1]  # works on Windows or not
    with Image.open(path) as im:
        feats = extract_features(im).reshape(1, -1)
    proba = clf.predict_proba(feats)[0]
    idx = proba.argmax()
    pred = encoder.inverse_transform([idx])[0]
    conf = proba[idx]
    true = true_labels[fname]

    total += 1
    if pred == true:
        correct += 1
    if conf < config.low_confidence_threshold:
        low_conf_total += 1
        if pred == true:
            low_conf_correct += 1

print(f"Overall accuracy:  {correct}/{total} = {correct/total:.1%}")
if low_conf_total:
    print(f"Low-confidence (<{config.low_confidence_threshold}) accuracy: "
          f"{low_conf_correct}/{low_conf_total} = {low_conf_correct/low_conf_total:.1%}")
else:
    print("No tiles fell below the threshold on this eval set.")