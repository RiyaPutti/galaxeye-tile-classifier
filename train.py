import sys
import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from config import config
from features import extract_features


def load_dataset(root):
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No class folders found under {root}")

    X, y = [], []
    for class_dir in class_dirs:
        label = class_dir.name
        for img_path in sorted(class_dir.glob("*.png")):
            with Image.open(img_path) as im:
                X.append(extract_features(im))
            y.append(label)
    return np.array(X), np.array(y)


def main():
    print(f"Loading training tiles from {config.candidate_tiles_dir}")
    X, y = load_dataset(config.candidate_tiles_dir)
    print(f"Loaded {len(X)} tiles across {len(set(y))} classes: {sorted(set(y))}")

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_enc,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y_enc,
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=config.random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    val_pred = clf.predict(X_val)
    print("\nHeld-out validation report:")
    print(classification_report(y_val, val_pred, target_names=encoder.classes_))

    artifact = {
        "model": clf,
        "label_encoder": encoder,
        "feature_extractor_version": "v1",
    }
    config.model_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, config.model_artifact_path)
    print(f"\nSaved model artifact to {config.model_artifact_path}")


if __name__ == "__main__":
    sys.exit(main())