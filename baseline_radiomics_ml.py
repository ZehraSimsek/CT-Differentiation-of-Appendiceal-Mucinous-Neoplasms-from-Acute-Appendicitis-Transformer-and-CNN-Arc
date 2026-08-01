"""
Radiomics-lite + Klasik ML Baseline — Appendisit/Müsinöz Sınıflandırma
==========================================================================
Q1 hakemlerinin standart sorusu: "Derin öğrenme, basit el-yapımı özniteliklere
göre gerçekten daha mı iyi?" Bu script o karşılaştırmayı sağlar.

Öznitelikler (pyradiomics kurulu değil; numpy/scipy/skimage ile hafif ama
meşru bir "radiomics-lite" seti):
  - First-order: mean, std, min/max, percentile'lar, skewness, kurtosis, entropy
  - GLCM doku öznitelikleri (contrast, homogeneity, energy, correlation, ASM) —
    orta 8 aksiyel kesitte hesaplanıp ortalanır
  - H5 metadata öznitelikleri (roi_volume_fraction, n_valid_slices_used,
    z_coverage_fraction, pixel_spacing) — zaten preprocessing'de hesaplanmış,
    klinik olarak anlamlı (lezyon boyutu/ekstresi ile ilişkili)

Model: LogisticRegression + RandomForest, AYNI 5-fold split + external test +
Q1 protokolü (Youden/0.5/90%-Sens + %95 bootstrap CI + kalibrasyon) ile.

Çalıştırma:
    cd segformer && python baseline_radiomics_ml.py
Çıktılar:
    experiments/baseline_logreg/, experiments/baseline_rf/
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from scipy import stats as sp_stats
from skimage.feature import graycomatrix, graycoprops
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from shared_utils import (
    SHARED_CONFIG, compute_binary_metrics, find_youden_threshold, compute_bootstrap_ci,
    print_full_metrics_table, plot_confusion_matrix, plot_roc_pr, plot_calibration_curve,
    setup_file_logging, roc_curve,
)

DATA_ROOT = Path(r"/home/zera/Downloads/Appendiks varyasyon3 DS-20260713T105239Z-2-001/Appendiks varyasyon3 DS")
DATAS_DIR = DATA_ROOT / "segformer" / "datas"
EXPERIMENTS_DIR = DATA_ROOT / "segformer" / "experiments"
FEATURES_CACHE = DATA_ROOT / "segformer" / "radiomics_lite_features.csv"

N_BOOTSTRAP = SHARED_CONFIG["n_bootstrap"]
GLCM_DISTANCES = [1, 2]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_LEVELS = 32  # hız için 32 gri seviyeye quantize


# ============================================================
# Öznitelik çıkarımı
# ============================================================
def _glcm_features(slice2d):
    img = (np.clip(slice2d, 0, 1) * (GLCM_LEVELS - 1)).astype(np.uint8)
    glcm = graycomatrix(img, distances=GLCM_DISTANCES, angles=GLCM_ANGLES,
                         levels=GLCM_LEVELS, symmetric=True, normed=True)
    feats = {}
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        feats[f"glcm_{prop}"] = float(graycoprops(glcm, prop).mean())
    return feats


def extract_features(h5_path):
    with h5py.File(h5_path, "r") as f:
        vol = f["image"][()].astype(np.float32)
        attrs = dict(f.attrs)

    if vol.ndim == 4:
        vol = vol[..., 0]  # [D,H,W,1] -> [D,H,W]

    flat = vol.flatten()
    feats = {
        "fo_mean": float(np.mean(flat)), "fo_std": float(np.std(flat)),
        "fo_min": float(np.min(flat)), "fo_max": float(np.max(flat)),
        "fo_median": float(np.median(flat)),
        "fo_p10": float(np.percentile(flat, 10)), "fo_p25": float(np.percentile(flat, 25)),
        "fo_p75": float(np.percentile(flat, 75)), "fo_p90": float(np.percentile(flat, 90)),
        "fo_skewness": float(sp_stats.skew(flat)), "fo_kurtosis": float(sp_stats.kurtosis(flat)),
        "fo_entropy": float(sp_stats.entropy(np.histogram(flat, bins=32, range=(0, 1))[0] + 1e-8)),
    }

    D = vol.shape[0]
    mid = D // 2
    slice_idxs = range(max(0, mid - 4), min(D, mid + 4))
    glcm_list = [_glcm_features(vol[i]) for i in slice_idxs]
    for key in glcm_list[0]:
        feats[key] = float(np.mean([g[key] for g in glcm_list]))

    # Preprocessing meta-öznitelikleri (klinik olarak anlamlı: lezyon boyutu/kapsamı)
    def _scalar(attr_key, default=np.nan):
        v = attrs.get(attr_key, default)
        try:
            return float(np.asarray(v).reshape(-1)[0])
        except Exception:
            return default

    feats["meta_roi_volume_fraction"] = _scalar("roi_volume_fraction")
    feats["meta_n_valid_slices_used"] = _scalar("n_valid_slices_used")
    feats["meta_z_coverage_fraction"] = _scalar("z_coverage_fraction")
    feats["meta_pixel_spacing_x"] = _scalar("pixel_spacing_x")
    feats["meta_slice_spacing"] = _scalar("slice_spacing")

    return feats


def build_feature_cache():
    if FEATURES_CACHE.exists():
        print(f"Özellik cache bulundu, yeniden hesaplanmıyor: {FEATURES_CACHE}")
        return pd.read_csv(FEATURES_CACHE)

    all_csvs = list(DATAS_DIR.glob("fold_*_train.csv")) + list(DATAS_DIR.glob("fold_*_val.csv")) + \
        [DATAS_DIR / "external_test_set.csv"]
    manifest = pd.concat([pd.read_csv(c) for c in all_csvs], ignore_index=True).drop_duplicates("patient_id")
    print(f"Özellik çıkarımı: {len(manifest)} benzersiz hasta...")

    rows = []
    for i, row in manifest.iterrows():
        feats = extract_features(row["h5_path"])
        feats["patient_id"] = row["patient_id"]
        feats["label"] = row["label"]
        rows.append(feats)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(manifest)} tamamlandı")

    feat_df = pd.DataFrame(rows)
    feat_df.to_csv(FEATURES_CACHE, index=False)
    print(f"Özellikler kaydedildi: {FEATURES_CACHE} ({feat_df.shape[1] - 2} öznitelik)")
    return feat_df


FEATURE_COLS = None  # build_feature_cache sonrası doldurulur


# ============================================================
# Tek fold eğitimi (sklearn modeli)
# ============================================================
def run_one_fold_sklearn(model_name, pipeline_factory, train_df, val_df, feat_df, fold_idx, output_dir):
    fold_dir = Path(output_dir) / f"fold_{fold_idx:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_feat = feat_df.merge(train_df[["patient_id"]], on="patient_id")
    val_feat = feat_df.merge(val_df[["patient_id"]], on="patient_id")

    X_train = train_feat[FEATURE_COLS].values
    y_train = train_feat["label"].values
    X_val = val_feat[FEATURE_COLS].values
    y_val = val_feat["label"].values

    pipeline = pipeline_factory()
    pipeline.fit(X_train, y_train)

    val_prob = pipeline.predict_proba(X_val)[:, 1]
    opt_thr, _ = find_youden_threshold(y_val, val_prob)
    m_y, cm_y, _ = compute_binary_metrics(y_val, val_prob, threshold=opt_thr)
    ci_y = compute_bootstrap_ci(y_val, val_prob, threshold=opt_thr, n_bootstraps=N_BOOTSTRAP)

    print_full_metrics_table(m_y, ci_y, f"{model_name} Fold {fold_idx}", f"Youden {opt_thr:.3f}")
    plot_confusion_matrix(cm_y, f"{model_name} Fold {fold_idx} @Youden {opt_thr:.3f}",
                           save_path=fold_dir / "cm_youden.png")

    import joblib
    joblib.dump(pipeline, fold_dir / "model.joblib")

    val_pred_df = pd.DataFrame({"patient_id": val_feat["patient_id"].values, "label": y_val,
                                 "prob_mucinous": val_prob, "fold": fold_idx, "youden_threshold": opt_thr})
    val_pred_df.to_csv(fold_dir / "val_predictions.csv", index=False)

    return m_y, ci_y, val_pred_df


def evaluate_external_test_sklearn(model_name, output_dir, feat_df, test_df, n_folds=5):
    import joblib
    test_feat = feat_df.merge(test_df[["patient_id"]], on="patient_id")
    X_test = test_feat[FEATURE_COLS].values
    yt = test_feat["label"].values

    test_dir = Path(output_dir) / "external_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    all_probs = []
    rows = []
    for fold_idx in range(1, n_folds + 1):
        model_path = Path(output_dir) / f"fold_{fold_idx:02d}" / "model.joblib"
        if not model_path.exists():
            continue
        pipeline = joblib.load(model_path)
        fold_prob = pipeline.predict_proba(X_test)[:, 1]
        all_probs.append(fold_prob)

        opt_thr_fold, _ = find_youden_threshold(yt, fold_prob)
        m_fold, _, _ = compute_binary_metrics(yt, fold_prob, threshold=opt_thr_fold)
        m_fold["fold"] = f"Fold {fold_idx}"
        m_fold["threshold_used"] = opt_thr_fold
        rows.append(m_fold)

    ens = np.mean(all_probs, axis=0)
    fpr, tpr, thrs = roc_curve(yt, ens)

    def _threshold_row(thr, name):
        m, cm, _ = compute_binary_metrics(yt, ens, threshold=thr)
        ci = compute_bootstrap_ci(yt, ens, threshold=thr, n_bootstraps=N_BOOTSTRAP)
        row = dict(m)
        row.update(ci)
        row["fold"] = name
        row["threshold_used"] = thr
        rows.append(row)
        return m, cm, ci

    m_05, cm_05, ci_05 = _threshold_row(0.5, "Ensemble (@0.5)")
    youden_thr, _ = find_youden_threshold(yt, ens)
    m_y, cm_y, ci_y = _threshold_row(youden_thr, "Ensemble (@Youden)")
    valid_idx = np.where(tpr >= 0.90)[0]
    high_sens_thr = float(thrs[valid_idx[0]]) if len(valid_idx) > 0 else youden_thr
    m_s, cm_s, ci_s = _threshold_row(high_sens_thr, "Ensemble (90+ Sens)")

    summary = pd.DataFrame(rows)
    cols = ["fold"] + [c for c in summary.columns if c != "fold"]
    summary = summary[cols]
    summary.to_csv(test_dir / "q1_external_test_metrics.csv", index=False)

    plot_confusion_matrix(cm_y, f"{model_name} External Test @Youden {youden_thr:.3f}",
                           save_path=test_dir / "cm_youden.png")
    plot_roc_pr(yt, ens, title_prefix=f"{model_name}_external_test", save_dir=test_dir, opt_threshold=youden_thr)
    cal = plot_calibration_curve(yt, ens, f"{model_name} External Test Kalibrasyonu",
                                  save_path=test_dir / "calibration.png")

    ens_probs_df = pd.DataFrame({"patient_id": test_feat["patient_id"].values, "label": yt, "prob_mucinous": ens})
    ens_probs_df.to_csv(test_dir / "ensemble_probs.csv", index=False)

    print(f"\n{model_name} — EXTERNAL TEST ENSEMBLE @Youden ({youden_thr:.3f}):")
    print_full_metrics_table(m_y, ci_y, model_name, f"Youden {youden_thr:.3f}")
    print(f"Kalibrasyon: Brier={cal['brier_score']:.4f} | ECE={cal['ece']:.4f}")

    return summary


def main():
    global FEATURE_COLS
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    feat_df = build_feature_cache()
    FEATURE_COLS = [c for c in feat_df.columns if c not in ("patient_id", "label")]
    print(f"Kullanılan öznitelik sayısı: {len(FEATURE_COLS)}")

    test_df = pd.read_csv(DATAS_DIR / "external_test_set.csv")

    model_configs = {
        "baseline_logreg": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)),
        ]),
        "baseline_rf": lambda: Pipeline([
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=5, class_weight="balanced",
                                            random_state=42, n_jobs=-1)),
        ]),
    }

    for model_name, pipeline_factory in model_configs.items():
        output_dir = EXPERIMENTS_DIR / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = setup_file_logging(output_dir / "train_log.txt")

        print("=" * 80)
        print(f"{model_name} — Radiomics-lite baseline")
        print("=" * 80)

        for fold_idx in range(1, SHARED_CONFIG["n_splits"] + 1):
            train_df = pd.read_csv(DATAS_DIR / f"fold_{fold_idx}_train.csv")
            val_df = pd.read_csv(DATAS_DIR / f"fold_{fold_idx}_val.csv")
            print(f"\n--- Fold {fold_idx}/{SHARED_CONFIG['n_splits']} ---")
            run_one_fold_sklearn(model_name, pipeline_factory, train_df, val_df, feat_df, fold_idx, output_dir)

        evaluate_external_test_sklearn(model_name, output_dir, feat_df, test_df,
                                        n_folds=SHARED_CONFIG["n_splits"])

        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()


if __name__ == "__main__":
    main()
