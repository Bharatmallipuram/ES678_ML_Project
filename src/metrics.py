"""Evaluation metrics used throughout the project."""
import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score, roc_curve)

C_FP, C_FN = 1, 5                      # a missed default costs 5x a false alarm
COST_THRESHOLD = C_FP / (C_FP + C_FN)  # = 1/6, optimal if probabilities are calibrated


def recall_at_fpr(y, scores, max_fpr=0.10):
    """Share of defaulters caught while flagging at most max_fpr of non-defaulters."""
    fpr, tpr, _ = roc_curve(y, scores)
    return tpr[fpr <= max_fpr].max()


def expected_cost(y, prob, threshold, c_fp=C_FP, c_fn=C_FN):
    """Average cost per customer of flagging everyone with prob >= threshold."""
    y = np.asarray(y)
    flag = np.asarray(prob) >= threshold
    fp = np.sum(flag & (y == 0))
    fn = np.sum(~flag & (y == 1))
    return (c_fp * fp + c_fn * fn) / len(y)


def evaluate(y, prob, threshold=COST_THRESHOLD):
    y, prob = np.asarray(y), np.asarray(prob)
    return {
        "pr_auc": average_precision_score(y, prob),
        "roc_auc": roc_auc_score(y, prob) if len(np.unique(prob)) > 1 else 0.5,
        "recall_at_10fpr": recall_at_fpr(y, prob),
        "cost_per_customer": expected_cost(y, prob, threshold),
        "brier": brier_score_loss(y, prob),
        "accuracy_at_0.5": np.mean((prob >= 0.5) == y),
    }