#!/bin/bash
set -e

PROJ="/home/zera/Downloads/Appendiks varyasyon3 DS-20260713T105239Z-2-001/Appendiks varyasyon3 DS"
Q1_DIR="$PROJ/segformer/q1_final"
EXP_DIR="$PROJ/segformer/experiments_q1_128"

cd "$Q1_DIR"

# Eski deney sonuçlarını yedekle (varsa)
if [ -d "$EXP_DIR" ]; then
    cp -r "$EXP_DIR" "$PROJ/segformer/experiments_q1_128_backup_$(date +%Y%m%d_%H%M%S)"
fi

# SegFormer3D MAE encoder'ını sil (mimari değişti, uyumsuz olabilir).
# MAE-Tiny encoder mimarisi aynı kaldı; varsa zaman kazanmak için kullan.
rm -f "$EXP_DIR/segformer3d_mae/segformer3d_mae_encoder.pt"

# Eğitimleri sırayla çalıştır
echo "[1/4] MAE-TinyTransformer3D eğitimi başlıyor..."
python train_mae_tinytransformer.py | tee log_mae_tinytransformer.out

echo "[2/4] Attention-SwinUNETR eğitimi başlıyor..."
python train_attention_swinunetr.py | tee log_attention_swinunetr.out

echo "[3/4] SwinUNETR Linear-Probe eğitimi başlıyor..."
python train_swinunetr_linearprobe.py | tee log_swinunetr_lp.out

echo "[4/4] SegFormer3D-MSCA eğitimi başlıyor..."
python train_segformer3d.py | tee log_segformer3d.out

# Tüm modeller bittikten sonra Q1 master tablosu üret
echo "Q1 Master karşılaştırma tablosu üretiliyor..."
python generate_q1_final_table.py

echo "Tüm işlemler tamamlandı. Çıktılar: $EXP_DIR"
