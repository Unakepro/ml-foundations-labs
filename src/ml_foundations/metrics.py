def confusion_counts(y_true, y_pred) -> tuple[int, int, int, int]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    tn = fp = fn = tp = 0

    for true, pred in zip(y_true, y_pred, strict=True):
        if true not in (0, 1) or pred not in (0, 1):
            raise ValueError("Labels must be either 0 or 1")

        if true != pred:
            if true == 1:
                fn += 1
            else:
                fp += 1
        else:
            if true == 1:
                tp += 1
            else:
                tn += 1

    return tn, fp, fn, tp


def precision_score(y_true, y_pred) -> float:
    _, fp, _, tp = confusion_counts(y_true, y_pred)

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)


def recall_score(y_true, y_pred) -> float:
    _, _, fn, tp = confusion_counts(y_true, y_pred)

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)


def specificity_score(y_true, y_pred) -> float:
    tn, fp, _, _ = confusion_counts(y_true, y_pred)

    if tn + fp == 0:
        return 0.0

    return tn / (tn + fp)


def f1_score(y_true, y_pred) -> float:
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    if precision + recall == 0:
        return 0.0

    return (2 * precision * recall) / (precision + recall)


def balanced_accuracy_score(y_true, y_pred) -> float:
    recall = recall_score(y_true, y_pred)
    specificity = specificity_score(y_true, y_pred)

    return (recall + specificity) / 2
