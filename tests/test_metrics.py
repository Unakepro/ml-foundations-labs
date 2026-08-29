import pytest
from sklearn.metrics import (
    balanced_accuracy_score as sklearn_balanced_accuracy_score,
)
from sklearn.metrics import (
    confusion_matrix as sklearn_confusion_matrix,
)
from sklearn.metrics import (
    f1_score as sklearn_f1_score,
)
from sklearn.metrics import (
    precision_score as sklearn_precision_score,
)
from sklearn.metrics import (
    recall_score as sklearn_recall_score,
)

from ml_foundations.metrics import (
    balanced_accuracy_score,
    confusion_counts,
    f1_score,
    precision_score,
    recall_score,
    specificity_score,
)


def test_metrics_with_perfect_predictions():
    y_true = [1, 0, 0, 1, 1]
    y_pred = [1, 0, 0, 1, 1]

    result = confusion_counts(y_true, y_pred)

    assert result == (2, 0, 0, 3)

    assert precision_score(y_true, y_pred) == pytest.approx(1.0)
    assert recall_score(y_true, y_pred) == pytest.approx(1.0)
    assert specificity_score(y_true, y_pred) == pytest.approx(1.0)
    assert f1_score(y_true, y_pred) == pytest.approx(1.0)
    assert balanced_accuracy_score(y_true, y_pred) == pytest.approx(1.0)


def test_metrics_with_all_predictions_negative():
    y_true = [1, 0, 0, 1, 1]
    y_pred = [0, 0, 0, 0, 0]

    result = confusion_counts(y_true, y_pred)

    assert result == (2, 0, 3, 0)

    assert precision_score(y_true, y_pred) == pytest.approx(0.0)
    assert recall_score(y_true, y_pred) == pytest.approx(0.0)
    assert specificity_score(y_true, y_pred) == pytest.approx(1.0)
    assert f1_score(y_true, y_pred) == pytest.approx(0.0)
    assert balanced_accuracy_score(y_true, y_pred) == pytest.approx(0.5)


def test_metrics_with_all_predictions_positive():
    y_true = [1, 0, 0, 1, 1]
    y_pred = [1, 1, 1, 1, 1]

    result = confusion_counts(y_true, y_pred)

    assert result == (0, 2, 0, 3)
    assert precision_score(y_true, y_pred) == pytest.approx(0.6)
    assert recall_score(y_true, y_pred) == pytest.approx(1.0)
    assert specificity_score(y_true, y_pred) == pytest.approx(0.0)
    assert f1_score(y_true, y_pred) == pytest.approx(0.75)
    assert balanced_accuracy_score(y_true, y_pred) == pytest.approx(0.5)


def test_metrics_with_imbalanced_classes():
    y_true = [1, 1, 1, 1, 0]
    y_pred = [0, 0, 1, 1, 0]

    result = confusion_counts(y_true, y_pred)

    assert result == (1, 0, 2, 2)

    assert precision_score(y_true, y_pred) == pytest.approx(1.0)
    assert recall_score(y_true, y_pred) == pytest.approx(0.5)
    assert specificity_score(y_true, y_pred) == pytest.approx(1.0)
    assert f1_score(y_true, y_pred) == pytest.approx(2 / 3)
    assert balanced_accuracy_score(y_true, y_pred) == pytest.approx(0.75)


def test_different_input_lengths():
    y_true = [1, 1, 1, 1, 0]
    y_pred = [0, 0, 1, 1, 0, 0]

    with pytest.raises(
        ValueError,
        match="y_true and y_pred must have the same length",
    ):
        confusion_counts(y_true, y_pred)


def test_precision_zero_denominator():
    y_true = [0, 0, 1, 1, 0]
    y_pred = [0, 0, 0, 0, 0]

    result = confusion_counts(y_true, y_pred)

    assert result == (3, 0, 2, 0)
    assert precision_score(y_true, y_pred) == 0.0


def test_recall_zero_denominator():
    y_true = [0, 0, 0, 0, 0]
    y_pred = [0, 0, 1, 1, 0]

    result = confusion_counts(y_true, y_pred)

    assert result == (3, 2, 0, 0)
    assert recall_score(y_true, y_pred) == 0.0


def test_specificity_zero_denominator():
    y_true = [1, 1, 1, 1, 1]
    y_pred = [1, 1, 0, 0, 1]

    result = confusion_counts(y_true, y_pred)

    assert result == (0, 0, 2, 3)
    assert specificity_score(y_true, y_pred) == 0.0


def test_f1_zero_denominator():
    y_true = [1, 0, 0, 1, 1]
    y_pred = [0, 0, 0, 0, 0]

    result = confusion_counts(y_true, y_pred)

    assert result == (2, 0, 3, 0)

    assert precision_score(y_true, y_pred) == 0.0
    assert recall_score(y_true, y_pred) == 0.0
    assert f1_score(y_true, y_pred) == 0.0


def test_metrics_match_sklearn():
    y_true = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 0, 0, 1, 1, 1, 1]

    expected_count = tuple(
        sklearn_confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    )

    assert confusion_counts(y_true, y_pred) == expected_count

    assert recall_score(y_true, y_pred) == pytest.approx(
        sklearn_recall_score(y_true, y_pred, zero_division=0)
    )
    assert precision_score(y_true, y_pred) == pytest.approx(
        sklearn_precision_score(y_true, y_pred, zero_division=0)
    )

    assert specificity_score(y_true, y_pred) == sklearn_recall_score(
        y_true, y_pred, pos_label=0, zero_division=0
    )

    assert f1_score(y_true, y_pred) == pytest.approx(
        sklearn_f1_score(y_true, y_pred, zero_division=0)
    )

    assert balanced_accuracy_score(y_true, y_pred) == pytest.approx(
        sklearn_balanced_accuracy_score(y_true, y_pred)
    )
