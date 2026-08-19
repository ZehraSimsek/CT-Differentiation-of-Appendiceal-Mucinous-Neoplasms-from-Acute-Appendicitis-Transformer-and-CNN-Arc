"""
utils/metrics.py — Medical Evaluation Metrics (Clinical-Grade)
===============================================================
Computes the full clinical metric suite required for a **Red Flag**
binary classifier using ``sklearn.metrics``.
Reported metrics
----------------
+-------------------+----------------------------------------------------------+
| Metric            | Clinical rationale                                       |
+===================+==========================================================+
| AUC-ROC           | Threshold-independent discriminative power; the primary  |
|                   | checkpoint selection metric.                             |
+-------------------+----------------------------------------------------------+
| Accuracy          | Baseline sanity check (insufficient alone for imbalanced |
|                   | datasets).                                               |
+-------------------+----------------------------------------------------------+
| Balanced Accuracy | (Sensitivity + Specificity) / 2 — fairer for imbalanced  |
|                   | datasets than plain accuracy.                            |
+-------------------+----------------------------------------------------------+
| Sensitivity       | TP / (TP + FN) — **the most critical metric**.  A missed |
| (Recall)          | tumour (false negative) can be life-threatening.         |
+-------------------+----------------------------------------------------------+
| Specificity       | TN / (TN + FP) — complementary to sensitivity; tracks   |
|                   | unnecessary interventions on healthy patients.           |
+-------------------+----------------------------------------------------------+
| Precision         | TP / (TP + FP) — measures the cost of false alarms.     |
+-------------------+----------------------------------------------------------+
| NPV               | TN / (TN + FN) — probability that a negative prediction |
|                   | is truly negative (no missed tumour).                    |
+-------------------+----------------------------------------------------------+
| F1-Score          | Harmonic mean of precision & recall — single summary     |
|                   | that balances both error types.                          |
+-------------------+----------------------------------------------------------+
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) for binary classification."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        if i == 0:
            in_bin = in_bin | (y_prob == bin_lower)
        prob_in_bin = in_bin.mean()
        if prob_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return float(ece)
def permutation_test_auc(y_true: np.ndarray, y_prob: np.ndarray, n_permutations: int = 1000) -> float:
    """Computes P-value for AUC via permutation testing."""
    if len(np.unique(y_true)) < 2:
        return 1.0
    try:
        actual_auc = roc_auc_score(y_true, y_prob)
        count = 0
        y_true_shuffled = y_true.copy()
        for _ in range(n_permutations):
            np.random.shuffle(y_true_shuffled)
            if roc_auc_score(y_true_shuffled, y_prob) >= actual_auc:
                count += 1
        p_value = (count + 1) / (n_permutations + 1)
        return float(p_value)
    except ValueError:
        return 1.0
def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Find the optimal decision threshold using Youden's J Statistic.
    J = Sensitivity + Specificity - 1
    The threshold that maximises J provides the best trade-off between
    sensitivity and specificity on the ROC curve.
    Parameters
    ----------
    y_true : ndarray, shape (N,)
        Ground-truth binary labels in {0, 1}.
    y_prob : ndarray, shape (N,)
        Predicted probabilities in [0, 1].
    Returns
    -------
    float
        Optimal threshold value.  Falls back to 0.5 if ROC computation
        fails (e.g., only one class present).
    """
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return float(thresholds[best_idx])
    except (ValueError, IndexError):
        return 0.5
@dataclass
class MetricResult:
    """Container for a single evaluation epoch's clinical metrics."""
    accuracy: float
    f1: float
    auc_roc: float
    sensitivity: float  
    precision: float
    specificity: float
    npv: float
    balanced_accuracy: float
    brier_score: float = 0.0
    ece: float = 0.0
    p_value_auc: float = 1.0
    optimal_threshold: float = 0.5
    opt_sensitivity: float = 0.0
    opt_specificity: float = 0.0
    opt_precision: float = 0.0
    opt_f1: float = 0.0
    opt_npv: float = 0.0
    opt_accuracy: float = 0.0
    opt_balanced_accuracy: float = 0.0
    def to_dict(self) -> dict[str, float]:
        """Serialise to a plain dict (useful for logging / JSON export)."""
        return {
            "auc_roc": self.auc_roc,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "precision": self.precision,
            "npv": self.npv,
            "f1": self.f1,
            "optimal_threshold": self.optimal_threshold,
            "opt_sensitivity": self.opt_sensitivity,
            "opt_f1": self.opt_f1,
            "opt_specificity": self.opt_specificity,
            "opt_precision": self.opt_precision,
            "opt_npv": self.opt_npv,
            "opt_accuracy": self.opt_accuracy,
            "opt_balanced_accuracy": self.opt_balanced_accuracy,
            "brier_score": self.brier_score,
            "ece": self.ece,
            "p_value_auc": self.p_value_auc,
        }
    def pretty_print(self, epoch: int | None = None) -> None:
        """Print a formatted dual-threshold metrics table to stdout."""
        header = f"  Epoch {epoch} " if epoch is not None else "  "
        print("\n" + "=" * 70)
        print(f"{header}-- Clinical Evaluation Metrics")
        print("=" * 70)
        print(f"  {'AUC-ROC':<22s} {self.auc_roc * 100:.2f}%")
        print("-" * 70)
        print(f"  {'':22s} {'Thresh=0.50':>12s}  {'Optimal':>12s} (J={self.optimal_threshold:.3f})")
        print("-" * 70)
        print(f"  {'Accuracy':<22s} {self.accuracy * 100:>11.2f}%  {self.opt_accuracy * 100:>11.2f}%")
        print(f"  {'Balanced Accuracy':<22s} {self.balanced_accuracy * 100:>11.2f}%  {self.opt_balanced_accuracy * 100:>11.2f}%")
        print(f"  {'Sensitivity (+)':<22s} {self.sensitivity * 100:>11.2f}%  {self.opt_sensitivity * 100:>11.2f}%   <- Red Flag KPI")
        print(f"  {'Specificity':<22s} {self.specificity * 100:>11.2f}%  {self.opt_specificity * 100:>11.2f}%")
        print(f"  {'Precision':<22s} {self.precision * 100:>11.2f}%  {self.opt_precision * 100:>11.2f}%")
        print(f"  {'NPV':<22s} {self.npv * 100:>11.2f}%  {self.opt_npv * 100:>11.2f}%")
        print(f"  {'F1-Score':<22s} {self.f1 * 100:>11.2f}%  {self.opt_f1 * 100:>11.2f}%")
        print("-" * 70)
        print(f"  {'Brier Score':<22s} {self.brier_score:.4f}")
        print(f"  {'ECE':<22s} {self.ece:.4f}")
        print(f"  {'AUC P-Value':<22s} {self.p_value_auc:.4e}")
        print("=" * 70 + "\n")
_EPS = 1e-7  
def _compute_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """
    Compute all 7 threshold-dependent clinical metrics at a given threshold.
    Returns a dict with keys: accuracy, balanced_accuracy, sensitivity,
    specificity, precision, npv, f1.
    """
    y_pred = (y_prob >= threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0.0)
    precision = precision_score(y_true, y_pred, zero_division=0.0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0.0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    specificity = float(tn) / (float(tn + fp) + _EPS)
    npv = float(tn) / (float(tn + fn) + _EPS)
    balanced_accuracy = (sensitivity + specificity) / 2.0
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "f1": f1,
    }
def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    y_pred: np.ndarray | None = None,
) -> MetricResult:
    """
    Compute the full clinical metric suite at both the default threshold
    and the Youden's J optimal threshold.
    Parameters
    ----------
    y_true : ndarray, shape (N,)
        Ground-truth binary labels in {0, 1}.
    y_prob : ndarray, shape (N,)
        Predicted probabilities (post-softmax) in [0, 1].
    threshold : float
        Default decision boundary (typically 0.5).
    y_pred : ndarray | None
        If provided, used for the default-threshold metrics instead of
        thresholding y_prob.  Ignored for optimal-threshold computation.
    Returns
    -------
    MetricResult
        Dataclass holding all computed metrics at both thresholds.
    """
    try:
        auc_roc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc_roc = 0.0
    if y_pred is not None:
        default_metrics = _compute_at_threshold(y_true, y_prob, threshold)
        default_metrics_from_pred = {}
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp = cm[0, 0], cm[0, 1]
        fn, tp = cm[1, 0], cm[1, 1]
        default_metrics["accuracy"] = accuracy_score(y_true, y_pred)
        default_metrics["f1"] = f1_score(y_true, y_pred, zero_division=0.0)
        default_metrics["precision"] = precision_score(y_true, y_pred, zero_division=0.0)
        default_metrics["sensitivity"] = recall_score(y_true, y_pred, zero_division=0.0)
        default_metrics["specificity"] = float(tn) / (float(tn + fp) + _EPS)
        default_metrics["npv"] = float(tn) / (float(tn + fn) + _EPS)
        default_metrics["balanced_accuracy"] = (
            default_metrics["sensitivity"] + default_metrics["specificity"]
        ) / 2.0
    else:
        default_metrics = _compute_at_threshold(y_true, y_prob, threshold)
    optimal_threshold = find_optimal_threshold(y_true, y_prob)
    opt_metrics = _compute_at_threshold(y_true, y_prob, optimal_threshold)
    return MetricResult(
        accuracy=default_metrics["accuracy"],
        f1=default_metrics["f1"],
        auc_roc=auc_roc,
        sensitivity=default_metrics["sensitivity"],
        precision=default_metrics["precision"],
        specificity=default_metrics["specificity"],
        npv=default_metrics["npv"],
        balanced_accuracy=default_metrics["balanced_accuracy"],
        optimal_threshold=optimal_threshold,
        opt_sensitivity=opt_metrics["sensitivity"],
        opt_specificity=opt_metrics["specificity"],
        opt_precision=opt_metrics["precision"],
        opt_f1=opt_metrics["f1"],
        opt_npv=opt_metrics["npv"],
        opt_accuracy=opt_metrics["accuracy"],
        opt_balanced_accuracy=opt_metrics["balanced_accuracy"],
        brier_score=brier_score_loss(y_true, y_prob),
        ece=expected_calibration_error(y_true, y_prob),
        p_value_auc=permutation_test_auc(y_true, y_prob),
    )
