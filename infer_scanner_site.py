"""
Tarayıcı/Merkez (Scanner/Site) Çıkarımı — DICOM UID kökü + seri açıklaması kümeleme
=========================================================================================
Hastane/kurum bilgisi metadata'da doğrudan yok, ama DICOM series_instance_uid'nin kök
kısmı (organizasyon + cihaz alt-ağacı) pratikte cihaz/kuruma özgüdür — aynı kökü paylaşan
hastalar neredeyse kesin aynı fiziksel tarayıcıdan/merkezden gelir. series_description
metni (dil/format tarzı) bu kümelemeyi doğrulamak için ikincil sinyal olarak kullanılır.

Bu, Q1 makale için önemli bir ek kontrol: çok-merkezli veri olup olmadığını, ve varsa
sınıf dağılımının merkezler arasında dengesiz olup olmadığını (confound riski) gösterir.

Çalıştırma:
    cd segformer && python infer_scanner_site.py
Çıktı:
    scanner_site_clusters.csv (segformer/ altında)
"""
from pathlib import Path
import pandas as pd

DATA_ROOT = Path(__file__).parent.parent
OUT_CSV = Path(__file__).parent / "scanner_site_clusters.csv"


def device_root(uid, depth=7):
    parts = str(uid).split(".")
    return ".".join(parts[:depth])


def main():
    app = pd.read_csv(DATA_ROOT / "metadata_processed_appendicitis.csv", encoding="utf-8")
    muc = pd.read_csv(DATA_ROOT / "metadata_processed_musinoz.csv", encoding="utf-8")
    df = pd.concat([app, muc], ignore_index=True)
    print(f"Toplam kayıt: {len(df)} (appendicitis={len(app)}, musinoz={len(muc)})")

    df["device_root"] = df["selected_series_uid"].apply(device_root)

    cluster_summary = (
        df.groupby("device_root")
        .agg(
            n_total=("patient_id", "count"),
            n_appendicitis=("label", lambda s: int((s == 0).sum())),
            n_mucinous=("label", lambda s: int((s == 1).sum())),
            example_series_desc=("series_description", lambda s: s.value_counts().index[0] if s.notna().any() else "(bos)"),
            pixel_spacing_mode=("pixel_spacing_x", lambda s: round(s.mode().iloc[0], 3) if len(s.mode()) else None),
            slice_spacing_mode=("slice_spacing", lambda s: s.mode().iloc[0] if len(s.mode()) else None),
        )
        .sort_values("n_total", ascending=False)
        .reset_index()
    )
    cluster_summary["pct_mucinous"] = (
        cluster_summary["n_mucinous"] / cluster_summary["n_total"] * 100
    ).round(1)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\n" + "=" * 100)
    print(f"Tespit edilen olası tarayıcı/merkez kümesi sayısı: {len(cluster_summary)}")
    print("=" * 100)
    print(cluster_summary.to_string(index=False))

    df[["patient_id", "label_name", "device_root", "series_description",
        "pixel_spacing_x", "slice_spacing", "selected_series_uid"]].to_csv(OUT_CSV, index=False)
    print(f"\nHasta bazlı küme ataması kaydedildi: {OUT_CSV}")

    # Genel sınıf oranı ile kıyasla — büyük sapma varsa merkez/sınıf confound riski var demektir
    overall_pct_mucinous = (df["label"] == 1).mean() * 100
    print(f"\nGenel müsinöz oranı: {overall_pct_mucinous:.1f}%")
    print("Bir kümenin oranı bundan çok saparsa (örn. tamamı tek sınıf), o merkez sınıfla")
    print("karışmış (confounded) demektir — model gerçek patolojiyi değil tarayıcı imzasını öğrenebilir.")

    return cluster_summary


if __name__ == "__main__":
    main()
