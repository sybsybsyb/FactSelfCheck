import numpy as np
from sklearn.metrics import auc, precision_recall_curve


def auc_pr(y_true: np.ndarray | list, y_pred: np.ndarray | list) -> float:
    p, r, _ = precision_recall_curve(y_true, y_pred)
    return auc(r, p)
