"""
Duplicate/Leakage Düzeltmesi — data_audit.py'nin bulduğu DICOM series_instance_uid
çakışmalarına karşı cerrahi (minimal) düzeltme.
=====================================================================================
Bulgular (bkz. data_audit.py + series_instance_uid taraması):
  1. iman_tugce: aynı UID hem Appendisit(0) hem Musinoz(1) klasöründe zıt etiketle.
     -> Kullanıcı kararı: DOKUNULMADI (ikisi de olduğu yerde kalıyor).
  2. irmak_unek_fatma / irmak_unek_fatma2: aynı UID, aynı etiket (0), iki ayrı dosya adı.
     Fold 1 ve Fold 2'de aynı hasta train+val'de aynı anda (leakage).
     -> "irmak_unek_fatma2" tüm split dosyalarından çıkarılır (kanonik: "irmak_unek_fatma").
  3. subasi_murat / subasi_murat2: aynı UID, aynı etiket (0). "subasi_murat" external test
     setinde, "subasi_murat2" ise 5-fold CV havuzunda (train+val) -> test/train leakage.
     -> "subasi_murat2" tüm CV split dosyalarından çıkarılır (kanonik: "subasi_murat",
        sadece external test setinde kalır, hiç CV'ye girmez — zaten doğru rolü buydu).

Bu script satır SİLER, dosya/H5 SİLMEZ (geri dönüşü kolay, ham veri dokunulmamış kalır).
Orijinal CSV'ler segformer/datas_backup_before_dedup_fix/ altında yedeklendi.

Çalıştırma:
    cd segformer && python fix_duplicate_leakage.py
"""
from pathlib import Path
import pandas as pd

DATAS_DIR = Path(__file__).parent / "datas"

PATIENTS_TO_REMOVE = ["irmak_unek_fatma2", "subasi_murat2"]


def fix_file(path: Path):
    df = pd.read_csv(path)
    before = len(df)
    df = df[~df["patient_id"].isin(PATIENTS_TO_REMOVE)].reset_index(drop=True)
    after = len(df)
    if before != after:
        df.to_csv(path, index=False)
        print(f"  {path.name}: {before} -> {after} satır ({before - after} kaldırıldı)")
    else:
        print(f"  {path.name}: değişiklik yok ({before} satır)")


def main():
    print("Duplicate leakage düzeltmesi uygulanıyor...")
    print(f"Kaldırılacak duplicate kayıtlar: {PATIENTS_TO_REMOVE}")
    print("(iman_tugce'ye dokunulmuyor — kullanıcı kararıyla olduğu gibi bırakıldı)\n")

    all_csvs = sorted(DATAS_DIR.glob("fold_*_train.csv")) + \
        sorted(DATAS_DIR.glob("fold_*_val.csv")) + \
        [DATAS_DIR / "external_test_set.csv"]

    for csv_path in all_csvs:
        fix_file(csv_path)

    print("\nTamamlandı. Doğrulamak için: python data_audit.py")


if __name__ == "__main__":
    main()
