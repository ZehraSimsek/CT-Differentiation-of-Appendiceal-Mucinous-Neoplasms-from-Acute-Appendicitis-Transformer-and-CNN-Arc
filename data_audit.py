"""
Veri Denetimi (Data Audit) — Q1 makale öncesi zorunlu kontroller
====================================================================
- H5 shape/label tutarlılığı (klasör etiketi vs H5 içi etiket)
- Intensity istatistikleri (normalizasyon sonrası aykırı değer var mı?)
- Fold'lar arası / train-val-test arası patient_id leakage kontrolü
- Sınıf dengesi (her fold + genel)
- Eksik/bozuk dosya taraması

Çalıştırma:
    cd segformer && python data_audit.py
Çıktı:
    data_audit_report.csv, data_audit_summary.txt (segformer/ altında)
"""
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import h5py

DATAS_DIR = Path(__file__).parent / "datas"
REPORT_CSV = Path(__file__).parent / "data_audit_report.csv"
SUMMARY_TXT = Path(__file__).parent / "data_audit_summary.txt"


def load_all_splits():
    splits = {}
    for f in sorted(DATAS_DIR.glob("fold_*_train.csv")):
        fold = int(f.stem.split("_")[1])
        splits[(fold, "train")] = pd.read_csv(f)
    for f in sorted(DATAS_DIR.glob("fold_*_val.csv")):
        fold = int(f.stem.split("_")[1])
        splits[(fold, "val")] = pd.read_csv(f)
    splits[("external", "test")] = pd.read_csv(DATAS_DIR / "external_test_set.csv")
    return splits


def audit_h5(row):
    result = {
        "patient_id": row["patient_id"], "csv_label": row["label"], "h5_path": row["h5_path"],
        "exists": False, "image_shape": None, "h5_label": None, "label_match": None,
        "intensity_min": None, "intensity_max": None, "intensity_mean": None, "intensity_std": None,
        "n_nan": None, "n_inf": None, "error": None,
    }
    p = Path(row["h5_path"])
    if not p.exists():
        result["error"] = "DOSYA YOK"
        return result
    result["exists"] = True
    try:
        with h5py.File(p, "r") as f:
            if "image" not in f:
                result["error"] = "image dataset yok"
                return result
            img = f["image"][()]
            result["image_shape"] = str(img.shape)
            if "label" in f:
                h5_label = int(np.array(f["label"]).reshape(-1)[0])
                result["h5_label"] = h5_label
                result["label_match"] = (h5_label == int(row["label"]))
            img = img.astype(np.float32)
            result["intensity_min"] = float(np.min(img))
            result["intensity_max"] = float(np.max(img))
            result["intensity_mean"] = float(np.mean(img))
            result["intensity_std"] = float(np.std(img))
            result["n_nan"] = int(np.isnan(img).sum())
            result["n_inf"] = int(np.isinf(img).sum())
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def main():
    splits = load_all_splits()

    # ---- 1) Tüm benzersiz hastaları topla ----
    all_rows = []
    membership = defaultdict(list)  # patient_id -> [(fold/external, split_name), ...]
    for (fold, split_name), df in splits.items():
        for _, row in df.iterrows():
            membership[row["patient_id"]].append(f"fold{fold}_{split_name}" if fold != "external" else "external_test")
        all_rows.append(df.assign(_fold=fold, _split=split_name))

    combined = pd.concat(all_rows, ignore_index=True)
    unique_patients = combined.drop_duplicates("patient_id")[["patient_id", "h5_path", "label"]]

    print("=" * 80)
    print(f"Toplam kayıt (fold+split satırları, hasta tekrarlanabilir): {len(combined)}")
    print(f"Benzersiz hasta sayısı: {len(unique_patients)}")
    print("=" * 80)

    # ---- 2) Leakage kontrolü: bir hasta aynı fold içinde hem train hem val'de mi? ----
    leakage_rows = []
    for pid, memberships in membership.items():
        fold_train_val = [m for m in memberships if m != "external_test"]
        folds_seen = set(m.split("_")[0] for m in fold_train_val)
        for fold in folds_seen:
            in_train = f"{fold}_train" in fold_train_val
            in_val = f"{fold}_val" in fold_train_val
            if in_train and in_val:
                leakage_rows.append({"patient_id": pid, "issue": f"{fold} train+val aynı anda"})
        if "external_test" in memberships:
            other = [m for m in memberships if m != "external_test"]
            if other:
                leakage_rows.append({"patient_id": pid, "issue": f"external_test + {other} — TEST LEAKAGE"})

    if leakage_rows:
        print(f"\n[UYARI] {len(leakage_rows)} olası leakage kaydı bulundu:")
        for r in leakage_rows[:20]:
            print(f"  - {r['patient_id']}: {r['issue']}")
    else:
        print("\n[OK] Leakage kontrolü temiz: hiçbir hasta train+val aynı fold'da değil, external test train/val ile örtüşmüyor.")

    # ---- 3) Fold bazında sınıf dengesi ----
    print("\nFold bazında sınıf dağılımı:")
    for (fold, split_name), df in sorted(splits.items(), key=lambda kv: str(kv[0])):
        counts = df["label"].value_counts().sort_index().to_dict()
        print(f"  {fold}/{split_name}: n={len(df)} | label_counts={counts}")

    # ---- 4) H5 dosya bazlı denetim (shape/label/intensity) ----
    print(f"\nH5 dosyaları denetleniyor ({len(unique_patients)} benzersiz hasta)...")
    audit_results = [audit_h5(row) for _, row in unique_patients.iterrows()]
    audit_df = pd.DataFrame(audit_results)
    audit_df.to_csv(REPORT_CSV, index=False)

    n_missing = (~audit_df["exists"]).sum()
    n_errors = audit_df["error"].notna().sum()
    n_label_mismatch = (audit_df["label_match"] == False).sum()  # noqa: E712
    shape_counts = audit_df["image_shape"].value_counts()

    print(f"\n  Eksik/okunamayan dosya: {n_missing}")
    print(f"  Hata veren dosya: {n_errors}")
    print(f"  H5-içi label ile CSV label uyuşmazlığı: {n_label_mismatch}")
    print(f"  Shape dağılımı:\n{shape_counts.to_string()}")

    valid = audit_df[audit_df["error"].isna()]
    if len(valid):
        print("\n  Intensity istatistikleri (tüm hastalar, ham H5 değerleri):")
        print(f"    min:  mean={valid['intensity_min'].mean():.3f}  range=[{valid['intensity_min'].min():.3f}, {valid['intensity_min'].max():.3f}]")
        print(f"    max:  mean={valid['intensity_max'].mean():.3f}  range=[{valid['intensity_max'].min():.3f}, {valid['intensity_max'].max():.3f}]")
        print(f"    mean: mean={valid['intensity_mean'].mean():.3f}  std_across_patients={valid['intensity_mean'].std():.3f}")
        print(f"    NaN toplam: {valid['n_nan'].sum()}  | Inf toplam: {valid['n_inf'].sum()}")

    # ---- 5) Özet dosyaya yaz ----
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(f"Benzersiz hasta: {len(unique_patients)}\n")
        f.write(f"Leakage kaydı: {len(leakage_rows)}\n")
        f.write(f"Eksik dosya: {n_missing}\n")
        f.write(f"Label uyuşmazlığı: {n_label_mismatch}\n")
        f.write(f"Shape dağılımı:\n{shape_counts.to_string()}\n")

    print(f"\nDetaylı rapor: {REPORT_CSV}")
    print(f"Özet: {SUMMARY_TXT}")

    return audit_df, leakage_rows


if __name__ == "__main__":
    main()
