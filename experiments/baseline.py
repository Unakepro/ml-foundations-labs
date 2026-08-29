import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml_foundations.metrics import (
    balanced_accuracy_score,
    confusion_counts,
    f1_score,
    precision_score,
    recall_score,
    specificity_score,
)


def train_baseline() -> tuple:
    dataset = load_breast_cancer()

    X = dataset.data
    y = (dataset.target == 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    val_probabilities = cross_val_predict(
        pipe,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    pipe.fit(X_train, y_train)

    return pipe, y_train, val_probabilities, X_test, y_test


def apply_threshold(probabilities, threshold: float = 0.5):
    return (probabilities >= threshold).astype(int)


def evaluate_threshold(
    y_true, probabilities, threshold: float = 0.5
) -> dict[str, float | int]:
    y_pred = apply_threshold(probabilities, threshold)

    tn, fp, fn, tp = confusion_counts(y_true, y_pred)

    return {
        "threshold": threshold,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "specificity": specificity_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "cost": 10 * fn + fp,
    }


def main() -> None:
    pipe, y_train, val_probabilities, X_test, y_test = train_baseline()

    thresholds = np.linspace(0.01, 0.99, 99)

    results = [
        evaluate_threshold(y_train, val_probabilities, threshold)
        for threshold in thresholds
    ]

    best_cost = min(results, key=lambda result: result["cost"])

    test_probabilities = pipe.predict_proba(X_test)[:, 1]
    selected_threshold = float(best_cost["threshold"])

    test_baseline = evaluate_threshold(
        y_test,
        test_probabilities,
        threshold=0.5,
    )

    test_selected = evaluate_threshold(
        y_test,
        test_probabilities,
        threshold=selected_threshold,
    )

    print(test_selected, test_baseline)


if __name__ == "__main__":
    main()
