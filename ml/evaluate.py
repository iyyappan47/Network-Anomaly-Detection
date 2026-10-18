import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

def evaluate_model_performance(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray):
    """
    Computes genuine evaluation metrics on the held-out test dataset.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    auc = None
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.0

    cm = confusion_matrix(y_true, y_pred).tolist()
    # cm format: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])

    # Per-class metrics
    report = classification_report(y_true, y_pred, target_names=["Normal", "Attack"], output_dict=True, zero_division=0)

    # Class support counts
    normal_support = int(np.sum(y_true == 0))
    attack_support = int(np.sum(y_true == 1))

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4) if auc is not None else None,
        "confusion_matrix": {
            "matrix": cm,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        },
        "classification_report": {
            "normal": {
                "precision": round(report["Normal"]["precision"], 4),
                "recall": round(report["Normal"]["recall"], 4),
                "f1_score": round(report["Normal"]["f1-score"], 4),
                "support": normal_support
            },
            "attack": {
                "precision": round(report["Attack"]["precision"], 4),
                "recall": round(report["Attack"]["recall"], 4),
                "f1_score": round(report["Attack"]["f1-score"], 4),
                "support": attack_support
            },
            "macro_avg": {
                "precision": round(report["macro avg"]["precision"], 4),
                "recall": round(report["macro avg"]["recall"], 4),
                "f1_score": round(report["macro avg"]["f1-score"], 4),
                "support": len(y_true)
            },
            "weighted_avg": {
                "precision": round(report["weighted avg"]["precision"], 4),
                "recall": round(report["weighted avg"]["recall"], 4),
                "f1_score": round(report["weighted avg"]["f1-score"], 4),
                "support": len(y_true)
            }
        },
        "total_test_samples": len(y_true)
    }
