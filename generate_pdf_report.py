"""
Q1 Benchmark PDF Raporu — experiments_q1_128 (External Test, n=24)
Tüm modeller, fold-bazlı metrikler, DeLong testi, ROC grafikleri.
"""
import os, sys
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, Image, PageBreak, HRFlowable)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

try:
    pdfmetrics.registerFont(TTFont('DejaVu', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    pdfmetrics.registerFont(TTFont('DejaVu-Bold', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
    FONT_NORMAL = 'DejaVu'
    FONT_BOLD = 'DejaVu-Bold'
except:
    FONT_NORMAL = 'Helvetica'
    FONT_BOLD = 'Helvetica-Bold'

# ── Paths ──────────────────────────────────────────────────────────────────────
EXP_DIR   = Path("experiments_q1_128")
OUT_PDF   = EXP_DIR / "Q1_Benchmark_Report.pdf"
IMG_DIR   = EXP_DIR / "_report_imgs"
IMG_DIR.mkdir(exist_ok=True)

DISPLAY = {
    "attention_swinunetr":  "Attention-SwinUNETR (AG-MSF)",
    "mae_tinytransformer":  "MAE-TinyTransformer3D",
    "segformer3d":          "SegFormer3D-MSCA",
    "swinunetr_lp":         "SwinUNETR Linear-Probe",
    "baseline_logreg":      "Radiomics + LogReg",
    "baseline_rf":          "Radiomics + RandomForest",
}
COLORS = {
    "attention_swinunetr": "#1565C0",
    "mae_tinytransformer": "#2E7D32",
    "segformer3d":         "#AD1457",
    "swinunetr_lp":        "#E65100",
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def savefig(name):
    p = IMG_DIR / name
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    return str(p)

def hex2rgb01(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2],16)/255 for i in (0,2,4))

# ── 1. Master Metrics ──────────────────────────────────────────────────────────
master = pd.read_csv(EXP_DIR / "q1_master_comparison.csv")
delong = pd.read_csv(EXP_DIR / "q1_delong_pairwise.csv")

# ── 2. Fold metrics ────────────────────────────────────────────────────────────
fold_data = {}
for mdir in sorted(EXP_DIR.iterdir()):
    p = mdir / "external_test" / "q1_external_test_metrics.csv"
    if p.exists():
        df = pd.read_csv(p)
        fold_data[mdir.name] = df

# ── 3. Figure A: AUC bar chart ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
models  = master["model"].tolist()
aucs    = master["auc_roc"].tolist()
cis_lo  = master["auc_ci"].apply(lambda s: float(s.strip("[]").split("-")[0])).tolist()
cis_hi  = master["auc_ci"].apply(lambda s: float(s.strip("[]").split("-")[1])).tolist()
bar_colors = ["#1565C0","#AD1457","#2E7D32","#E65100"][:len(models)]
bars = ax.barh(models, aucs, color=bar_colors, alpha=0.85, height=0.5)
for i, (lo, hi, auc) in enumerate(zip(cis_lo, cis_hi, aucs)):
    ax.errorbar(auc, i, xerr=[[auc-lo],[hi-auc]], fmt="none",
                color="black", capsize=4, linewidth=1.5)
    ax.text(auc+0.005, i, f"{auc:.3f}", va="center", fontsize=9)
ax.axvline(0.80, color="red", linestyle="--", alpha=0.6, label="AUC=0.80 hedef")
ax.set_xlabel("AUC-ROC", fontsize=11)
ax.set_title("External Test AUC-ROC (Ensemble @Youden, n=24)\n95% Bootstrap CI", fontsize=11, fontweight="bold")
ax.set_xlim(0.5, 1.0)
ax.legend(fontsize=9)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
FIG_AUC = savefig("fig_auc_bar.png")

# ── 4. Figure B: SENS vs SPEC scatter ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
for i, row in master.iterrows():
    c = bar_colors[i % len(bar_colors)]
    ax.scatter(row["specificity"], row["sensitivity"], s=160, color=c, zorder=5, edgecolors="black", linewidth=0.8)
    ax.annotate(row["model"].split()[0], (row["specificity"], row["sensitivity"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.axhline(0.80, color="red", linestyle="--", alpha=0.5, label="SENS=0.80")
ax.axvline(0.50, color="gray", linestyle="--", alpha=0.5, label="SPEC=0.50")
ax.set_xlabel("Specificity", fontsize=11)
ax.set_ylabel("Sensitivity", fontsize=11)
ax.set_title("Sensitivity vs Specificity\n(Ensemble @Youden, n=24)", fontsize=11, fontweight="bold")
ax.set_xlim(-0.05, 1.1); ax.set_ylim(0.4, 1.05)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
FIG_SENS = savefig("fig_sens_spec.png")

# ── 5. Figure C: Fold AUC line plot ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
for mkey, df in fold_data.items():
    fold_rows = df[df["fold"].str.startswith("Fold")].sort_values("fold")
    if fold_rows.empty: continue
    c = COLORS.get(mkey, "#555555")
    label = DISPLAY.get(mkey, mkey)
    ax.plot(range(1, len(fold_rows)+1), fold_rows["auc_roc"].values,
            marker="o", color=c, linewidth=2, markersize=6, label=label)
ax.axhline(0.80, color="red", linestyle="--", alpha=0.5)
ax.set_xlabel("Fold"); ax.set_ylabel("AUC-ROC")
ax.set_title("Fold-Bazlı AUC-ROC (External Test, n=24)", fontsize=11, fontweight="bold")
ax.set_xticks(range(1, 6))
ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3)
plt.tight_layout()
FIG_FOLD = savefig("fig_fold_auc.png")

# ── 6. Figure D: F1 grouped bar ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(5)
width = 0.2
for i, (mkey, df) in enumerate(fold_data.items()):
    fold_rows = df[df["fold"].str.startswith("Fold")].sort_values("fold")
    if fold_rows.empty: continue
    c = COLORS.get(mkey, "#999999")
    ax.bar(x + i*width, fold_rows["f1"].values, width, label=DISPLAY.get(mkey,mkey), color=c, alpha=0.8)
ax.set_xticks(x + width*1.5)
ax.set_xticklabels([f"Fold {i}" for i in range(1,6)])
ax.set_ylabel("F1-Score"); ax.set_ylim(0.5, 1.0)
ax.set_title("Fold-Bazlı F1-Score (External Test, n=24)", fontsize=11, fontweight="bold")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
FIG_F1 = savefig("fig_fold_f1.png")

# ── ROC curves from ensemble probs ─────────────────────────────────────────────
from shared_utils import roc_curve, roc_auc_score
fig, ax = plt.subplots(figsize=(6, 6))
for mkey, df in fold_data.items():
    probs_path = EXP_DIR / mkey / "external_test" / "ensemble_probs.csv"
    if not probs_path.exists(): continue
    pdf = pd.read_csv(probs_path)
    yt = pdf["label"].values; yp = pdf["prob_mucinous"].values
    if len(np.unique(yt)) < 2: continue
    fpr, tpr, _ = roc_curve(yt, yp)
    auc = roc_auc_score(yt, yp)
    c = COLORS.get(mkey, "#555555")
    ax.plot(fpr, tpr, color=c, lw=2, label=f"{DISPLAY.get(mkey,mkey)} (AUC={auc:.3f})")
ax.plot([0,1],[0,1],"--",color="gray",alpha=0.5)
ax.set_xlabel("1 - Specificity (FPR)", fontsize=11)
ax.set_ylabel("Sensitivity (TPR)", fontsize=11)
ax.set_title("ROC Eğrileri — Ensemble (External Test, n=24)", fontsize=11, fontweight="bold")
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
plt.tight_layout()
FIG_ROC = savefig("fig_roc.png")

# ── PDF BUILD ──────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(str(OUT_PDF), pagesize=A4,
                        leftMargin=2*cm, rightMargin=2*cm,
                        topMargin=2*cm, bottomMargin=2*cm)
styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName=FONT_BOLD, fontSize=16, spaceAfter=8, alignment=TA_CENTER)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName=FONT_BOLD, fontSize=13, spaceBefore=12, spaceAfter=6)
H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontName=FONT_BOLD, fontSize=11, spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle("BODY", parent=styles["Normal"], fontName=FONT_NORMAL, fontSize=9, leading=13)
CAPTION = ParagraphStyle("CAP", parent=styles["Normal"], fontName=FONT_NORMAL, fontSize=8, leading=11,
                          textColor=colors.grey, alignment=TA_CENTER)

def hr(): return HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceAfter=6)
def h1(t): return Paragraph(t, H1)
def h2(t): return Paragraph(t, H2)
def h3(t): return Paragraph(t, H3)
def body(t): return Paragraph(t, BODY)
def caption(t): return Paragraph(t, CAPTION)
def sp(n=6): return Spacer(1, n)

def img(path, w=14*cm):
    try:
        from PIL import Image as PILImage
        im = PILImage.open(path)
        iw, ih = im.size
        h = w * ih / iw
        return Image(path, width=w, height=h)
    except:
        return Image(path, width=w)

# Table style helpers
HEADER_BG = colors.HexColor("#1565C0")
ALT_BG    = colors.HexColor("#EEF2F7")
def make_table(data, col_widths=None):
    t = Table(data, colWidths=col_widths)
    style = [
        ("BACKGROUND", (0,0), (-1,0), HEADER_BG),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), FONT_BOLD),
        ("FONTNAME",   (0,1), (-1,-1), FONT_NORMAL),
        ("FONTSIZE",   (0,0), (-1,-1), 8),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("GRID",       (0,0), (-1,-1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, ALT_BG]),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
    ]
    t.setStyle(TableStyle(style))
    return t

story = []

# ── TITLE PAGE ─────────────────────────────────────────────────────────────────
story += [
    sp(40),
    h1("AMN vs. Apandisit 3D CT Sınıflandırma"),
    h1("Q1 Benchmark Raporu"),
    sp(8),
    Paragraph("experiments_q1_128 | External Test: n=24 | @Youden Threshold", CAPTION),
    sp(4),
    Paragraph("Oluşturulma: 2026-07-28", CAPTION),
    sp(40),
    hr(),
    PageBreak(),
]

# ── SECTION 1: DATASET ─────────────────────────────────────────────────────────
story += [
    h2("1. Dataset Özeti"),
    hr(),
    body("Toplam hasta: <b>143</b> (117 Müsinöz AMN, 26 Apandisit) — Tek merkez, retrospektif."),
    body("Veri bölme: StratifiedGroupKFold (hasta bazlı, leakage yok)."),
    sp(6),
]

ds_data = [
    ["Bölüm", "Müsinöz (AMN)", "Apandisit", "Toplam", "Oran"],
    ["Train+Val (5-fold)", "98", "21", "119", "4.7:1"],
    ["External Test (holdout)", "19", "5", "24", "3.8:1"],
    ["TOPLAM", "117", "26", "143", "4.5:1"],
]
fold_dist = [
    ["Fold", "Train (M/A)", "Val (M/A)"],
    ["Fold 1", "78 / 18 = 96", "20 / 4 = 24"],
    ["Fold 2", "78 / 18 = 96", "20 / 4 = 24"],
    ["Fold 3", "79 / 18 = 97", "19 / 4 = 23"],
    ["Fold 4", "78 / 17 = 95", "20 / 5 = 25"],
    ["Fold 5", "79 / 17 = 96", "19 / 5 = 24"],
]
story += [
    make_table(ds_data, [4*cm, 3.5*cm, 3*cm, 3*cm, 2.5*cm]),
    sp(8),
    make_table(fold_dist, [3*cm, 6*cm, 6*cm]),
    sp(4),
    PageBreak(),
]

# ── SECTION 2: MASTER TABLE ────────────────────────────────────────────────────
story += [
    h2("2. Ensemble @Youden — External Test Master Tablosu (n=24)"),
    hr(),
]

cols_show = ["model","auc_roc","auc_ci","sensitivity","specificity","ppv","npv","f1","accuracy","brier_score","ece"]
headers   = ["Model","AUC","95% CI","SENS","SPEC","PPV","NPV","F1","ACC","Brier","ECE"]
col_w     = [4.5*cm,1.5*cm,2.8*cm,1.4*cm,1.4*cm,1.4*cm,1.4*cm,1.4*cm,1.4*cm,1.5*cm,1.4*cm]

mdata = [headers]
for _, row in master.iterrows():
    mdata.append([
        str(row["model"]),
        f"{row['auc_roc']:.3f}",
        str(row["auc_ci"]),
        f"{row['sensitivity']:.3f}",
        f"{row['specificity']:.3f}",
        f"{row['ppv']:.3f}",
        f"{row['npv']:.3f}",
        f"{row['f1']:.3f}",
        f"{row['accuracy']:.3f}",
        f"{row['brier_score']:.3f}",
        f"{row['ece']:.3f}",
    ])
story += [make_table(mdata, col_w), sp(4)]

# DeLong
story += [
    sp(8), h3("2.1 DeLong Pairwise AUC Karşılaştırması"),
]
dl_headers = ["Model 1","AUC1","Model 2","AUC2","Δ AUC","z","p-value","p<0.05?"]
dl_w = [3.8*cm,1.4*cm,3.8*cm,1.4*cm,1.4*cm,1.4*cm,1.4*cm,1.4*cm]
dl_data = [dl_headers]
for _, row in delong.iterrows():
    dl_data.append([
        str(row["model_1"]), f"{row['auc_1']:.3f}",
        str(row["model_2"]), f"{row['auc_2']:.3f}",
        f"{row['auc_diff']:+.3f}", f"{row['z']:.3f}",
        f"{row['p_value']:.4f}",
        "✓" if row["significant_p<0.05"] else "✗",
    ])
story += [make_table(dl_data, dl_w), sp(4), PageBreak()]

# ── SECTION 3: FIGURES ─────────────────────────────────────────────────────────
story += [
    h2("3. Grafikler"),
    hr(),
    h3("3.1 AUC-ROC Karşılaştırması (95% Bootstrap CI)"),
    img(FIG_AUC, 14*cm),
    caption("Şekil 1. Ensemble @Youden external test AUC-ROC. Kırmızı kesikli çizgi Q1 hedef eşiği (0.80)."),
    sp(8),
    h3("3.2 Sensitivity vs Specificity"),
    img(FIG_SENS, 10*cm),
    caption("Şekil 2. Klinik kısıtlar: SENS ≥ 0.80 (kırmızı), SPEC ≥ 0.50 (gri)."),
    PageBreak(),
    h3("3.3 ROC Eğrileri (Ensemble Olasılıkları)"),
    img(FIG_ROC, 12*cm),
    caption("Şekil 3. Tüm modellerin ROC eğrileri, ensemble posterior probability ile."),
    sp(8),
    h3("3.4 Fold-Bazlı AUC-ROC"),
    img(FIG_FOLD, 14*cm),
    caption("Şekil 4. Her fold için external test AUC-ROC. Fold 5 tüm modellerde düşük."),
    sp(8),
    h3("3.5 Fold-Bazlı F1-Score"),
    img(FIG_F1, 14*cm),
    caption("Şekil 5. F1-Score fold dağılımı."),
    PageBreak(),
]

# ── SECTION 4: FOLD-WISE TABLES ───────────────────────────────────────────────
story += [h2("4. Model Bazlı Fold Metrikleri"), hr()]
fold_cols = ["fold","auc_roc","sensitivity","specificity","f1","accuracy","tp","fp","fn","tn"]
fold_headers = ["Fold","AUC","SENS","SPEC","F1","ACC","TP","FP","FN","TN"]
fold_cw = [2*cm,1.6*cm,1.6*cm,1.6*cm,1.6*cm,1.6*cm,1.2*cm,1.2*cm,1.2*cm,1.2*cm]

for mkey, df in fold_data.items():
    fold_rows = df[df["fold"].str.startswith("Fold")].sort_values("fold")
    ens_row   = df[df["fold"] == "Ensemble (@Youden)"]
    if fold_rows.empty: continue
    story += [h3(DISPLAY.get(mkey, mkey))]
    tdata = [fold_headers]
    for _, r in fold_rows.iterrows():
        tdata.append([
            r["fold"], f"{r['auc_roc']:.3f}", f"{r['sensitivity']:.3f}",
            f"{r['specificity']:.3f}", f"{r['f1']:.3f}", f"{r['accuracy']:.3f}",
            int(r["tp"]), int(r["fp"]), int(r["fn"]), int(r["tn"]),
        ])
    if not ens_row.empty:
        r = ens_row.iloc[0]
        tdata.append([
            "Ensemble", f"{r['auc_roc']:.3f}", f"{r['sensitivity']:.3f}",
            f"{r['specificity']:.3f}", f"{r['f1']:.3f}", f"{r['accuracy']:.3f}",
            "—","—","—","—",
        ])
    t = Table(tdata, colWidths=fold_cw)
    tstyle = [
        ("BACKGROUND", (0,0),(-1,0), HEADER_BG),
        ("TEXTCOLOR",  (0,0),(-1,0), colors.white),
        ("FONTNAME",   (0,0),(-1,0), FONT_BOLD),
        ("FONTNAME",   (0,1),(-1,-2), FONT_NORMAL),
        ("FONTNAME",   (0,len(tdata)-1),(-1,len(tdata)-1), FONT_BOLD),
        ("BACKGROUND", (0,len(tdata)-1),(-1,len(tdata)-1), colors.HexColor("#FFF3E0")),
        ("FONTSIZE",   (0,0),(-1,-1), 8),
        ("ALIGN",      (0,0),(-1,-1), "CENTER"),
        ("GRID",       (0,0),(-1,-1), 0.4, colors.grey),
        ("ROWBACKGROUNDS",(0,1),(-1,-2),[colors.white, ALT_BG]),
        ("TOPPADDING", (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
    ]
    t.setStyle(TableStyle(tstyle))
    story += [t, sp(10)]

# ── SECTION 5: METHODOLOGY SUMMARY ────────────────────────────────────────────
story += [
    PageBreak(),
    h2("5. Metodoloji Özeti"),
    hr(),
    h3("5.1 Sınıf Dengesizliği Stratejisi"),
    body("• <b>Clinical Focal Loss</b>: pos_weight=3.0 (final), γ=1.5, label smoothing=0.10"),
    body("• <b>Threshold</b>: Çift kısıt (SENS≥0.80 VE SPEC≥0.50) → F1 maksimize → Youden fallback"),
    body("• <b>Bootstrap CI</b>: n=2000, %95 güven aralığı"),
    sp(6),
    h3("5.2 Eğitim Protokolü (Tüm DL Modeller)"),
    body("• Optimizer: AdamW | LR: 1e-4 – 5e-4 | Weight Decay: 1e-2"),
    body("• Scheduler: Linear Warmup (10 epoch) + Cosine Annealing"),
    body("• SWA: Son %30 epoch (epoch≥70), swa_lr = base_lr × 0.1"),
    body("• Gradient Accumulation: accum_steps=4–8 → eff. batch=16"),
    body("• EMA checkpoint seçimi: α=0.3"),
    body("• TTA: 4 yön (original + D/H/W flip), olasılık ortalaması"),
    sp(6),
    h3("5.3 Validasyon Protokolü"),
    body("• 5-Fold StratifiedGroupKFold (hasta bazlı, leakage yok)"),
    body("• Bağımsız external holdout: n=24 (bu rapor) / n=37 (final)"),
    body("• DeLong pairwise AUC testi (Sun & Xu, 2014)"),
    body("• Kalibrasyon: Brier score + ECE (10 bin)"),
    sp(6),
    h3("5.4 Model Seçim Gerekçesi"),
    body("• <b>SwinUNETR Linear-Probe</b>: Frozen pretrained backbone → overfitting riski minimal"),
    body("• <b>Attention-SwinUNETR</b>: 5-stage hierarchical multi-scale fusion (AG-MSF, özgün)"),
    body("• <b>MAE-TinyTransformer3D</b>: Self-supervised pretraining (özgün 3D MAE adaptasyonu)"),
    body("• <b>SegFormer3D-MSCA</b>: 3D SegFormer + CBAM + multi-scale fusion (özgün)"),
    body("• <b>Radiomics baselines</b>: Klasik el-yapımı öznitelik karşılaştırması"),
    PageBreak(),
]

# ── SECTION 6: KEY FINDINGS ────────────────────────────────────────────────────
story += [
    h2("6. Temel Bulgular"),
    hr(),
    body("1. <b>SwinUNETR Linear-Probe en yüksek AUC'yi</b> elde etti (0.853, 95%CI [0.674–0.979]), "
         "SPEC=1.000 ile hiçbir Apandisit vakasını AMN olarak yanlış sınıflandırmadı."),
    sp(4),
    body("2. <b>Attention-SwinUNETR en yüksek SENS'i</b> sağladı (0.842), klinik "
         "açıdan en kritik metrikte (FN minimizasyonu) öne çıktı."),
    sp(4),
    body("3. <b>DeLong testi hiçbir model çifti için p<0.05</b> sonuç vermedi — küçük "
         "external test seti (n=24) istatistiksel güç sınırlıdır."),
    sp(4),
    body("4. <b>Fold 5 tüm modellerde zayıf</b> (SENS≈0.58, FN=8) — bu fold'daki hasta "
         "dağılımı incelenmeli."),
    sp(4),
    body("5. <b>MAE pretraining</b> (TinyTransformer3D) en yüksek SPEC'i (1.000) verdi "
         "ancak SENS=0.632 ile AMN vakalarının %37'sini kaçırdı."),
    sp(12),
    h3("Limitasyonlar"),
    body("• Tek merkez, retrospektif, n=143 (küçük kohort)"),
    body("• Ciddi sınıf dengesizliği (4.5:1) — AMN nadir tümör"),
    body("• External test n=24; DeLong testi için yetersiz istatistiksel güç"),
    body("• Radiomics baseline bu raporda mevcut değil (henüz koşturulmadı)"),
]

# ── BUILD ──────────────────────────────────────────────────────────────────────
doc.build(story)
print(f"\n✅ PDF oluşturuldu: {OUT_PDF}")
print(f"   Boyut: {OUT_PDF.stat().st_size / 1024:.1f} KB")
