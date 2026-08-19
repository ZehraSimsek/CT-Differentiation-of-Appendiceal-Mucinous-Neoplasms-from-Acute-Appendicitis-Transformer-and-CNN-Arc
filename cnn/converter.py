import os
import pandas as pd
def convert_csv_paths(csv_path: str, output_path: str, use_relative: bool = True):
    if not os.path.exists(csv_path):
        print(f"UYARI: Dosya bulunamadı, atlanıyor: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    base_dir = os.path.abspath(os.getcwd())
    def transform_path(path):
        filename = os.path.basename(path)
        matching = df.loc[df['h5_path'] == path, 'label_name']
        if len(matching) > 0:
            label_name = matching.iloc[0]
        else:
            label_name = "Appendisit" if "appendisit" in path.lower() or "apandisit" in path.lower() else "Musinoz"
        folder = "Appendisit" if label_name == "Appendisit" else "Musinoz"
        subfolder = "apandisit_128" if label_name == "Appendisit" else "musinoz_128"
        if use_relative:
            return os.path.join("datas", folder, subfolder, filename)
        else:
            return os.path.join(base_dir, "datas", folder, subfolder, filename)
    df['h5_path'] = df['h5_path'].apply(transform_path)
    df.to_csv(output_path, index=False)
    print(f"Başarılı: {output_path} oluşturuldu.")
if __name__ == "__main__":
    print(f"Mevcut Çalışma Dizini: {os.getcwd()}")
    convert_csv_paths('datas/external_test_set.csv', 'datas/external_test_set_fixed.csv')
    for i in range(1, 6):
        for split in ['train', 'val']:
            in_file = os.path.join('datas', f'fold_{i}_{split}.csv')
            out_file = os.path.join('datas', f'fold_{i}_{split}_fixed.csv')
            convert_csv_paths(in_file, out_file)
