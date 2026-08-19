"""
run_all_multirun_2d.py
======================
Grup A — 4 Transformer modeli, 3 run, 5 fold (toplam 60 model eğitimi).
YENİ 2D veriseti üzerinde çalışır (apandisit_2d + musinoz_2d, toplam 244 hasta).
Tüm çıktılar → experiments_multirun_2d/<model_adı>/run_XX/
Çalıştırma:
    cd segformer
    nohup /home/zera/venvs/torch/bin/python transformermodels/run_all_multirun_2d.py \
        > experiments_multirun_2d/master.log 2>&1 &
"""
import sys, os, json, time
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("final"))
sys.path.insert(0, os.path.abspath("transformermodels"))
sys.path.insert(0, os.path.abspath("transformermodels/final"))
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import confusion_matrix
from shared_utils import (
    SHARED_CONFIG, AppendixH5Dataset,
    ClinicalFocalLoss, pos_weight_from_labels,
    get_warmup_cosine_scheduler, train_one_epoch,
    evaluate_model, evaluate_external_test_ensemble,
    find_youden_threshold, compute_binary_metrics,
    compute_bootstrap_ci, print_full_metrics_table,
    plot_confusion_matrix, plot_roc_pr,
    clinical_composite, setup_file_logging,
)
from torch.utils.data import DataLoader
N_RUNS    = 3
RUN_SEEDS = [42, 123, 456]
DATA_ROOT = Path("/home/zera/Downloads/Appendiks varyasyon3 DS-20260713T105239Z-2-001/Appendiks varyasyon3 DS")
DATAS_DIR = DATA_ROOT / "segformer" / "datas_2d"        
OUT_ROOT  = DATA_ROOT / "segformer" / "experiments_multirun_2d"  
OUT_ROOT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
def run_one_fold_clean(train_df, val_df, fold_idx, config, fold_dir, builder, model_tag):
    from sklearn.metrics import f1_score as _f1
    fold_dir = Path(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_ds = AppendixH5Dataset(train_df, augment=True,  config=config)
    val_ds   = AppendixH5Dataset(val_df,   augment=False, config=config)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,
                               num_workers=config["num_workers"], pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"], shuffle=False,
                               num_workers=config["num_workers"], pin_memory=True)
    model = builder().to(DEVICE)
    if hasattr(model, "_load_pretrained"):
        model._load_pretrained()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = get_warmup_cosine_scheduler(optimizer, config.get("warmup_epochs", 10), config["n_epochs"])
    criterion = ClinicalFocalLoss(
        pos_weight=pos_weight_from_labels(train_df["label"].values),
        gamma=config.get("focal_gamma", 2.0),
        smoothing=config.get("label_smoothing", 0.05),
    )
    best_score, patience_cnt = -10.0, 0
    ema_score, ema_alpha = None, config.get("ema_alpha", 0.3)
    min_save = config.get("min_epochs_before_save", 3)
    history  = []
    for epoch in range(1, config["n_epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE,
                                     config.get("mixup_alpha", 0.0),
                                     accum_steps=config.get("accum_steps", 1))
        val_loss, val_auc, _, _, pred_df = evaluate_model(
            model, val_loader, criterion, DEVICE, use_tta=False)
        scheduler.step()
        yt  = pred_df["label"].values
        yp  = pred_df["prob_mucinous"].values
        thr = float(pred_df["_threshold_used"].iloc[0]) if "_threshold_used" in pred_df.columns else 0.5
        yhat = (yp >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, yhat, labels=[0, 1]).ravel()
        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        f1   = float(_f1(yt, yhat, zero_division=0))
        comp = clinical_composite(sens, val_auc, spec, f1=f1)
        min_sens = config.get("min_sens_floor", 0.80)
        min_spec = config.get("min_spec_floor", 0.50)
        meets    = (sens >= min_sens - 1e-5) and (spec >= min_spec - 1e-5)
        raw      = comp if meets else (comp - 2.0)
        ema_score = raw if ema_score is None else (ema_alpha * raw + (1 - ema_alpha) * ema_score)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                         "val_auc": val_auc, "sens": sens, "spec": spec,
                         "f1": f1, "composite": comp, "ema": ema_score})
        print(f"[{model_tag} | fold {fold_idx} | ep {epoch:03d}] "
              f"train={train_loss:.4f} val={val_loss:.4f} auc={val_auc:.4f} "
              f"SENS={sens:.3f} SPEC={spec:.3f} COMP={comp:.4f} EMA={ema_score:.4f}")
        if epoch >= min_save and ema_score > best_score:
            best_score = ema_score
            patience_cnt = 0
            torch.save({"model_state_dict": model.state_dict(),
                        "val_auc": val_auc, "ema_score": ema_score}, fold_dir / "best_model.pt")
            pred_df.to_csv(fold_dir / "best_val_predictions.csv", index=False)
            tag = "★ DUAL" if meets else "☆ FALLBACK"
            print(f"    {tag} (SENS={sens:.3f} SPEC={spec:.3f} comp={comp:.4f})")
        elif epoch >= min_save:
            patience_cnt += 1
            if patience_cnt >= config["patience"]:
                print(f"  Early stop @ {epoch} (best EMA={best_score:.4f})")
                break
    pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False)
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    best_pred = pd.read_csv(fold_dir / "best_val_predictions.csv")
    yt, yp = best_pred["label"].values, best_pred["prob_mucinous"].values
    opt_thr, _ = find_youden_threshold(yt, yp)
    m, cm, _   = compute_binary_metrics(yt, yp, threshold=opt_thr)
    ci         = compute_bootstrap_ci(yt, yp, threshold=opt_thr, n_bootstraps=config["n_bootstrap"])
    print_full_metrics_table(m, ci, f"{model_tag} Fold {fold_idx}", f"Youden {opt_thr:.3f}")
    plot_confusion_matrix(cm, f"{model_tag} Fold {fold_idx}", save_path=fold_dir / "cm_youden.png")
    best_pred["fold"] = fold_idx
    best_pred["youden_threshold"] = opt_thr
    return m, ci, best_pred
def run_model_multirun(model_tag, builder, config, pretrain_fn=None):
    base_dir = OUT_ROOT / model_tag.lower().replace("-", "_").replace(" ", "_")
    base_dir.mkdir(parents=True, exist_ok=True)
    setup_file_logging(base_dir / "train_log.txt")
    test_csv = DATAS_DIR / "external_test_set.csv"
    mae_encoder_path = None
    if pretrain_fn is not None:
        all_csvs   = list(DATAS_DIR.glob("fold_*_train.csv")) + list(DATAS_DIR.glob("fold_*_val.csv"))
        manifest   = pd.concat([pd.read_csv(c) for c in all_csvs]).drop_duplicates("patient_id")
        mae_encoder_path = pretrain_fn(manifest, config, base_dir)
    run_ext_aucs = []
    for run_idx, seed in enumerate(RUN_SEEDS, start=1):
        print(f"\n{'#'*80}")
        print(f"  {model_tag}  |  RUN {run_idx}/{N_RUNS}  |  seed={seed}")
        print(f"{'#'*80}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        run_dir = base_dir / f"run_{run_idx:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        if mae_encoder_path is not None:
            import transformermodels.final.train_mae_tinytransformer as _mae_mod
            _mae_enc_path = mae_encoder_path
            def _mae_builder_with_encoder():
                m = _mae_mod.build_model().to(DEVICE)
                if _mae_enc_path.exists():
                    m.encoder.load_state_dict(
                        torch.load(_mae_enc_path, map_location=DEVICE, weights_only=False))
                return m
            effective_builder = _mae_builder_with_encoder
        else:
            effective_builder = builder
        all_preds = []
        for fold_idx in range(1, config["n_splits"] + 1):
            train_df = pd.read_csv(DATAS_DIR / f"fold_{fold_idx}_train.csv")
            val_df   = pd.read_csv(DATAS_DIR / f"fold_{fold_idx}_val.csv")
            print(f"\n{'='*70}\n{model_tag} | Run {run_idx} | Fold {fold_idx}\n{'='*70}")
            fold_dir = run_dir / f"fold_{fold_idx:02d}"
            _, _, pred = run_one_fold_clean(
                train_df, val_df, fold_idx, config, fold_dir, effective_builder, model_tag)
            all_preds.append(pred)
        oof = pd.concat(all_preds, ignore_index=True)
        yt, yp = oof["label"].values, oof["prob_mucinous"].values
        opt_thr, _ = find_youden_threshold(yt, yp)
        m_oof, cm_oof, _ = compute_binary_metrics(yt, yp, threshold=opt_thr)
        ci_oof = compute_bootstrap_ci(yt, yp, threshold=opt_thr, n_bootstraps=config["n_bootstrap"])
        agg_dir = run_dir / "aggregate_oof"
        agg_dir.mkdir(exist_ok=True)
        oof.to_csv(agg_dir / "oof_predictions.csv", index=False)
        plot_confusion_matrix(cm_oof, f"{model_tag} Run{run_idx} OOF",
                               save_path=agg_dir / "agg_cm.png")
        plot_roc_pr(yt, yp, f"{model_tag}_run{run_idx}_oof", agg_dir, opt_threshold=opt_thr)
        print(f"\n{model_tag} Run {run_idx} — OOF Aggregate:")
        print_full_metrics_table(m_oof, ci_oof, f"{model_tag} Run{run_idx}", f"Youden {opt_thr:.3f}")
        ext_summary = evaluate_external_test_ensemble(
            model_builder=effective_builder,
            base_dir=run_dir,
            config=config,
            test_csv_path=test_csv,
            model_display_name=f"{model_tag} (Run {run_idx})",
            n_folds=config["n_splits"],
        )
        if ext_summary is not None:
            rows = ext_summary[ext_summary["fold"].str.contains("Youden", na=False)]
            if len(rows):
                run_ext_aucs.append(float(rows["auc_roc"].values[0]))
    print(f"\n{'='*80}")
    print(f"  {model_tag} — MULTI-RUN FINAL ÖZET (External Test @Youden)")
    print(f"{'='*80}")
    if run_ext_aucs:
        arr = np.array(run_ext_aucs)
        summary = {"model": model_tag, "runs": run_ext_aucs,
                   "mean_auc": float(arr.mean()), "std_auc": float(arr.std())}
        print(f"  AUC = {arr.mean():.3f} ± {arr.std():.3f}  ({N_RUNS} runs × 5-fold)")
        with open(base_dir / "multirun_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    print(f"{'='*80}\n")
    return run_ext_aucs
def main():
    t_start = time.time()
    import hashlib
    expected_hashes = {
        "fold_1_train.csv": "42be48706af91948ebe9d2fc365c352b",
        "fold_1_val.csv":   "6c7fa00e50a58d990652b0a95545fa5c",
        "fold_2_train.csv": "ed4f6067f96738ec6938bf9ea7298561",
        "fold_2_val.csv":   "1dc27d335efb4116a51d0b586e7cf76b",
        "fold_3_train.csv": "1a40015eb991dd5fd49c9da2ba19cc53",
        "fold_3_val.csv":   "90ce95f96a9edbd736eefe014066abad",
        "fold_4_train.csv": "97066eb3557220f7018024ae01f0583d",
        "fold_4_val.csv":   "83eb8971e7eb5d66758cfa90e3ad6ea5",
        "fold_5_train.csv": "d138f47b966ef928bdfe9fb2eed46674",
        "fold_5_val.csv":   "8a757810ea687441b045dd2d15dca48b",
    }
    print("=== FOLD BÜTÜNLÜK KONTROLÜ (2D Dataset) ===")
    for fname, h_exp in expected_hashes.items():
        h_got = hashlib.md5((DATAS_DIR / fname).read_bytes()).hexdigest()
        ok = h_got == h_exp
        print(f"  {'✅' if ok else '❌'} {fname}")
        if not ok:
            raise RuntimeError(f"FOLD DEĞİŞMİŞ: {fname}. Eğitimi durdurun!")
    print("  Fold atamaları doğrulandı.\n")
    import transformermodels.final.train_swinunetr_linearprobe as swin_lp
    CONFIG_SWIN_LP = dict(SHARED_CONFIG)
    CONFIG_SWIN_LP.update({
        "model_name": "swinunetr_lp", "feature_size": 48,
        "n_epochs": 100, "patience": 25, "lr": 1e-3,
        "weight_decay": 5e-3, "warmup_epochs": 5,
        "focal_gamma": 1.0, "label_smoothing": 0.05,
        "batch_size": 2, "accum_steps": 8,
        "min_epochs_before_save": 3, "ema_alpha": 0.3,
    })
    def _build_swin_lp():
        m = swin_lp.build_model().to(DEVICE)
        m = swin_lp.load_pretrained_swin(m)
        return m
    run_model_multirun("SwinUNETR-LP", _build_swin_lp, CONFIG_SWIN_LP)
    import transformermodels.final.train_attention_swinunetr as attn_swin
    CONFIG_ATTN = dict(SHARED_CONFIG)
    CONFIG_ATTN.update({
        "model_name": "attention_swinunetr",
        "n_epochs": 100, "patience": 25, "lr": 1e-4,
        "weight_decay": 5e-3, "warmup_epochs": 10,
        "mixup_alpha": 0.05, "accum_steps": 4,
        "focal_gamma": 1.0, "label_smoothing": 0.05,
        "batch_size": 2, "min_epochs_before_save": 3, "ema_alpha": 0.3,
    })
    run_model_multirun("AG-MSF", attn_swin.build_model, CONFIG_ATTN)
    import transformermodels.final.train_mae_tinytransformer as mae_tiny
    CONFIG_MAE = dict(SHARED_CONFIG)
    CONFIG_MAE.update({
        "model_name": "mae_tinytransformer", "embed_dim": 192,
        "n_epochs": 100, "patience": 20, "lr": 2e-4,
        "weight_decay": 1e-2, "warmup_epochs": 10,
        "mae_mask_ratio": 0.75, "mae_epochs": 80,
        "mixup_alpha": 0.4, "accum_steps": 8,
        "batch_size": 2, "min_epochs_before_save": 3, "ema_alpha": 0.3,
    })
    run_model_multirun("MAE-Tiny3D", mae_tiny.build_model, CONFIG_MAE,
                       pretrain_fn=mae_tiny.run_mae_pretraining)
    import transformermodels.final.train_segformer3d as segformer3d
    CONFIG_SEG = dict(SHARED_CONFIG)
    CONFIG_SEG.update({
        "model_name": "segformer3d",
        "n_epochs": 100, "patience": 25, "lr": 3e-4,
        "weight_decay": 1e-2, "warmup_epochs": 10,
        "mixup_alpha": 0.1, "accum_steps": 8,
        "focal_gamma": 1.0, "label_smoothing": 0.05,
        "batch_size": 2, "min_epochs_before_save": 3, "ema_alpha": 0.3,
    })
    run_model_multirun("SegFormer3D-MSCA", segformer3d.build_model, CONFIG_SEG)
    elapsed = (time.time() - t_start) / 3600
    print(f"\n{'#'*80}")
    print(f"  TÜM MODELLERİN EĞİTİMİ TAMAMLANDI — {elapsed:.1f} saat")
    print(f"  Çıktılar: {OUT_ROOT}")
    print(f"{'#'*80}")
if __name__ == "__main__":
    main()
