
"""Evaluate frozen MobileNetV3-Large embeddings on grouped fold 0."""

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from data import load_folds, OUTPUTS
from metrics import evaluate, summarise, log_result

SEED = 42
FOLD = 0
SCHEME = "grouped"


def main():
    # Load previously extracted MobileNetV3 embeddings.
    X = np.load(OUTPUTS / "X_mobilenetv3.npy")
    folds = load_folds()

    # Verify that embeddings match their original image order.
    order = pd.read_csv(
        OUTPUTS / "feature_order_mobilenetv3.csv"
    )["Filename"].tolist()

    assert order == folds["Filename"].tolist(), "Image order mismatch!"
    assert len(X) == len(folds)

    # Exclude the held-out test set.
    pool = folds.loc[~folds["is_test"]].copy()

    # Select train and validation indices using grouped fold 0.
    train_mask = (
        (~folds["is_test"]) &
        (folds["fold_grouped"] != FOLD)
    ).to_numpy()

    val_mask = (
        (~folds["is_test"]) &
        (folds["fold_grouped"] == FOLD)
    ).to_numpy()

    X_train = X[train_mask]
    y_train = folds.loc[train_mask, "Label"].to_numpy()

    X_val = X[val_mask]
    y_val = folds.loc[val_mask, "Label"].to_numpy()

    # Verify grouped splitting prevents session leakage.
    train_sessions = set(folds.loc[train_mask, "session"])
    val_sessions = set(folds.loc[val_mask, "session"])
    assert train_sessions.isdisjoint(val_sessions)

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print("Feature dimensions:", X_train.shape[1])
    print("Shared sessions:", len(train_sessions & val_sessions))

    # Fit the scaler only on training data to prevent leakage.
    classifier = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            max_iter=2000,
            class_weight=None,
            random_state=SEED,
        )),
    ])

    print("\nTraining logistic regression...")
    classifier.fit(X_train, y_train)

    # Evaluate validation data only.
    predictions = classifier.predict(X_val)
    metrics = evaluate(y_val, predictions)

    print("\n=== MobileNetV3 Frozen Embeddings ===")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(
        "Weed-to-negative rate: "
        f"{metrics['weed_to_negative_rate']:.4f}"
    )

    # Use the existing shared results logger.
    summary = summarise([metrics])

    log_result(
        model="mobilenetv3_large_frozen_logreg",
        scheme=SCHEME,
        params={
            "head": "logreg",
            "pretrained": "IMAGENET1K_V2",
            "frozen": True,
            "folds": [FOLD],
            "balanced": False,
            "augmentation": False,
        },
        summary=summary,
    )

    print("\nResult saved to results.csv")


if __name__ == "__main__":
    main()
