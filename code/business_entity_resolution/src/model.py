"""Stage 10-11: LightGBM Training + F0.5 Threshold Tuning.

Handles:
  1. Label generation — joins features with ground truth to create match labels
  2. Train/val split — GroupShuffleSplit by s1_entity_id (no data leakage)
  3. LightGBM binary classifier training
  4. F0.5 threshold optimization on validation set
  5. Model checkpoint save/load
  6. Prediction on new data

Memory management strategy:
  - Features are loaded in 2M-row chunks as float32 (halves memory vs float64)
  - After the train/val split, pandas DataFrames are freed BEFORE converting
    to numpy arrays — this prevents holding both simultaneously (~19 GB saved)
  - Only a lightweight val_meta (3 columns) is kept during threshold tuning
  - LightGBM therefore sees ~12 GB peak instead of the ~30 GB that caused OOM

Usage:
    python -m src.model
"""
import gc
import os
import time
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    precision_score, recall_score,
    classification_report, roc_auc_score,
)

from . import config


# ═══════════════════════════════════════════════════════════════════
# LABEL GENERATION
# ═══════════════════════════════════════════════════════════════════

def load_ground_truth_pairs(gt_path):
    """Load ground truth as a set of (s1_id, cand_id) tuples."""
    gt_pairs = set()
    with open(gt_path, 'r', encoding='utf-8') as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split('\t')
            s1_id = parts[0].strip()
            if len(parts) > 1 and parts[1].strip():
                for cid in parts[1].split(','):
                    cid = cid.strip()
                    if cid:
                        gt_pairs.add((s1_id, cid))
    return gt_pairs


def generate_labels(features_path, gt_path):
    """Attach binary labels to feature rows using ground truth.

    Loads in 2M-row chunks as float32 to keep memory manageable at 135M rows.
    Label generation uses zip() — avoids the 18+ min apply(axis=1) Python loop.
    """
    print(f"  Loading ground truth...")
    gt_pairs = load_ground_truth_pairs(gt_path)
    print(f"  Ground truth: {len(gt_pairs):,} true pairs")

    print(f"  Loading features from {features_path.name} in chunks...")

    # float32 for feature columns halves memory vs float64 default
    dtypes = {'s1_entity_id': str, 'cand_entity_id': str}
    for col in config.FEATURE_COLS:
        dtypes[col] = np.float32

    chunksize = 2_000_000
    reader = pd.read_csv(
        features_path, sep='\t', encoding='utf-8',
        dtype=dtypes, chunksize=chunksize,
        usecols=['s1_entity_id', 'cand_entity_id'] + config.FEATURE_COLS,
        keep_default_na=False,
    )

    chunk_list = []
    total_rows = 0
    t0 = time.time()

    for i, chunk in enumerate(reader):
        # Vectorized label generation via zip — ~50x faster than apply(axis=1)
        # apply(axis=1) on 135M rows takes 18+ minutes; this takes ~30 seconds
        pair_tuples = list(zip(chunk['s1_entity_id'], chunk['cand_entity_id']))
        chunk['label'] = np.array(
            [1 if p in gt_pairs else 0 for p in pair_tuples],
            dtype=np.int8
        )
        chunk_list.append(chunk)
        total_rows += len(chunk)
        print(f"    Loaded {total_rows:,} rows... ({(time.time()-t0)/60:.1f} min)")

    print(f"  Concatenating {len(chunk_list)} chunks into RAM...")
    df = pd.concat(chunk_list, ignore_index=True)
    del chunk_list
    gc.collect()

    print(f"  Feature rows: {len(df):,}")
    n_pos = df['label'].sum()
    n_neg = len(df) - n_pos
    print(f"  Labels: {n_pos:,} positive ({n_pos/len(df)*100:.2f}%), "
          f"{n_neg:,} negative ({n_neg/len(df)*100:.2f}%)")

    return df


# ═══════════════════════════════════════════════════════════════════
# TRAIN / VALIDATION SPLIT
# ═══════════════════════════════════════════════════════════════════

def split_train_val(df, val_ratio=None, random_state=42):
    """Split data using GroupShuffleSplit on s1_entity_id — no data leakage."""
    if val_ratio is None:
        val_ratio = config.VAL_RATIO

    groups = df['s1_entity_id'].values
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
    train_idx, val_idx = next(splitter.split(df, groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)

    train_s1 = set(train_df['s1_entity_id'].unique())
    val_s1   = set(val_df['s1_entity_id'].unique())
    overlap  = train_s1 & val_s1
    assert len(overlap) == 0, f"Data leakage! {len(overlap)} S1 entities in both splits."

    print(f"\n  Train/Val Split (GroupShuffleSplit on s1_entity_id):")
    print(f"    Train: {len(train_df):,} pairs, {len(train_s1):,} S1 entities, "
          f"{train_df['label'].sum():,} positives ({train_df['label'].mean()*100:.2f}%)")
    print(f"    Val:   {len(val_df):,} pairs, {len(val_s1):,} S1 entities, "
          f"{val_df['label'].sum():,} positives ({val_df['label'].mean()*100:.2f}%)")
    print(f"    Leakage check: PASS")

    return train_df, val_df


# ═══════════════════════════════════════════════════════════════════
# FEATURE COLUMNS
# ═══════════════════════════════════════════════════════════════════

def get_feature_columns(df):
    """Return feature columns present in the DataFrame."""
    available = []
    for col in config.FEATURE_COLS:
        if col in df.columns:
            available.append(col)
        else:
            print(f"  WARNING: Feature '{col}' not found — skipping.")
    return available


# ═══════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════

def train_lightgbm(X_train, y_train, X_val, y_val, feature_cols, n_pos, n_neg):
    """Train LightGBM binary classifier.

    Accepts numpy arrays directly so the caller can free pandas DataFrames
    before this function runs — critical for fitting within 50 GB RAM.

    Args:
        X_train / y_train: numpy arrays from the training split
        X_val / y_val:     numpy arrays from the validation split
        feature_cols:      list of feature names (for importance printout)
        n_pos / n_neg:     positive/negative counts in training set

    Returns:
        trained LGBMClassifier
    """
    print(f"\n{'='*60}")
    print(f"  TRAINING LightGBM")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Train samples: {len(X_train):,}")
    print(f"  Val samples:   {len(X_val):,}")
    print(f"{'='*60}")

    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"  Class imbalance: {n_neg:,} neg / {n_pos:,} pos = {scale_pos_weight:.1f}x")

    params = config.LGBM_PARAMS.copy()
    params['scale_pos_weight'] = scale_pos_weight
    n_estimators = params.pop('n_estimators', 800)

    model = lgb.LGBMClassifier(n_estimators=n_estimators, **params)

    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='binary_logloss',
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )
    elapsed = time.time() - t0

    print(f"\n  Training complete in {elapsed:.1f}s")
    print(f"  Best iteration: {model.best_iteration_}")
    print(f"  Best val logloss: {model.best_score_['valid_0']['binary_logloss']:.6f}")

    importance = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: -x[1]
    )
    print(f"\n  Feature Importance (gain):")
    for fname, imp in importance:
        print(f"    {fname:30s}: {imp:>6d}")

    return model


# ═══════════════════════════════════════════════════════════════════
# THRESHOLD TUNING (F0.5)
# ═══════════════════════════════════════════════════════════════════

def per_entity_f05(val_meta, y_pred, beta=0.5):
    """Macro-average F_beta per S1 entity — matches the challenge metric exactly.

    Args:
        val_meta: DataFrame with columns [s1_entity_id, label] only
                  (caller should strip feature columns before calling to save RAM)
        y_pred:   binary predictions array (same length as val_meta)
    """
    val = pd.DataFrame({
        's1_entity_id': val_meta['s1_entity_id'].values,
        'label':        val_meta['label'].values,
        'pred':         y_pred,
    })

    tp       = ((val['label'] == 1) & (val['pred'] == 1)).groupby(val['s1_entity_id']).sum()
    pred_pos = (val['pred']  == 1).groupby(val['s1_entity_id']).sum()
    true_pos = (val['label'] == 1).groupby(val['s1_entity_id']).sum()

    precision = (tp / pred_pos).fillna(0.0)
    recall    = (tp / true_pos).fillna(0.0)

    f_num   = (1 + beta**2) * precision * recall
    f_den   = (beta**2 * precision) + recall
    f_score = (f_num / f_den).fillna(0.0)

    # Correct singletons: true_pos == 0 AND pred_pos == 0 → perfect score
    perfect_mask = (true_pos == 0) & (pred_pos == 0)
    f_score[perfect_mask] = 1.0

    return float(f_score.mean())


def tune_threshold(model, X_val, y_val, val_meta, feature_cols, beta=0.5):
    """Find the probability threshold maximising F-beta on the validation set.

    Args:
        model:       trained LGBMClassifier
        X_val:       numpy feature array for validation (float32)
        y_val:       numpy label array for validation (int8)
        val_meta:    lightweight DataFrame with [s1_entity_id, label] columns
                     used for per-entity F0.5 computation
        feature_cols: feature names (informational only)
        beta:        beta for F-beta (0.5 = precision-weighted)

    Returns:
        (best_threshold, best_f_beta, results_list)
    """
    print(f"\n{'='*60}")
    print(f"  THRESHOLD TUNING (F{beta})")
    print(f"{'='*60}")

    y_prob = model.predict_proba(X_val)[:, 1]

    thresholds = np.arange(
        config.THRESHOLD_SEARCH_MIN,
        config.THRESHOLD_SEARCH_MAX + config.THRESHOLD_SEARCH_STEP,
        config.THRESHOLD_SEARCH_STEP,
    )

    best_threshold = 0.5
    best_fbeta     = 0.0
    results        = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if y_pred.sum() == 0:
            continue

        prec = precision_score(y_val, y_pred, zero_division=0)
        rec  = recall_score(y_val, y_pred, zero_division=0)
        fb   = per_entity_f05(val_meta, y_pred, beta=beta)

        results.append({
            'threshold':     round(float(t), 3),
            'precision':     round(prec, 4),
            'recall':        round(rec,  4),
            f'f{beta}':      round(fb,   4),
            'predicted_pos': int(y_pred.sum()),
        })

        if fb > best_fbeta:
            best_fbeta     = fb
            best_threshold = float(t)

    print(f"\n  Threshold search results (±0.05 around best):")
    print(f"  {'Threshold':>10s} {'Precision':>10s} {'Recall':>10s} "
          f"{'F0.5':>10s} {'Pred_Pos':>10s}")
    for r in results:
        if abs(r['threshold'] - best_threshold) <= 0.05:
            marker = " <-- BEST" if abs(r['threshold'] - best_threshold) < 0.005 else ""
            print(f"  {r['threshold']:>10.3f} {r['precision']:>10.4f} "
                  f"{r['recall']:>10.4f} {r[f'f{beta}']:>10.4f} "
                  f"{r['predicted_pos']:>10d}{marker}")

    print(f"\n  Best threshold: {best_threshold:.3f}")
    print(f"  Best F{beta}:      {best_fbeta:.4f}")

    y_pred_best = (y_prob >= best_threshold).astype(int)
    print(f"\n  Classification Report @ threshold={best_threshold:.3f}:")
    print(classification_report(y_val, y_pred_best, target_names=['No Match', 'Match']))

    try:
        auc = roc_auc_score(y_val, y_prob)
        print(f"  AUC-ROC: {auc:.4f}")
    except ValueError:
        pass

    return best_threshold, best_fbeta, results


# ═══════════════════════════════════════════════════════════════════
# SAVE / LOAD
# ═══════════════════════════════════════════════════════════════════

def save_model(model, threshold, feature_cols, metrics=None):
    """Save model weights + metadata to config.MODEL_DIR."""
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    model_path = config.MODEL_DIR / "lgbm_model.joblib"
    meta_path  = config.MODEL_DIR / "model_metadata.json"

    joblib.dump(model, model_path)

    metadata = {
        'threshold':      float(threshold),
        'feature_cols':   feature_cols,
        'n_features':     len(feature_cols),
        'best_iteration': model.best_iteration_,
        'metrics':        metrics or {},
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Model saved to {model_path}")
    print(f"  Metadata saved to {meta_path}")
    print(f"  Threshold: {threshold:.3f}")


def load_model():
    """Load trained model and metadata. Returns (model, threshold, feature_cols)."""
    model_path = config.MODEL_DIR / "lgbm_model.joblib"
    meta_path  = config.MODEL_DIR / "model_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"No saved model at {model_path}. Train first.")

    model = joblib.load(model_path)
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    print(f"  Loaded model from {model_path}")
    print(f"  Threshold: {metadata['threshold']:.3f}")
    print(f"  Features: {metadata['n_features']}")

    return model, metadata['threshold'], metadata['feature_cols']


# ═══════════════════════════════════════════════════════════════════
# PREDICTION
# ═══════════════════════════════════════════════════════════════════

def predict(model, df, feature_cols, threshold):
    """Run prediction on a features DataFrame."""
    X    = df[feature_cols].values
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    df = df.copy()
    df['match_prob'] = probs
    df['match_pred'] = preds

    n_matches = preds.sum()
    print(f"  Predictions: {n_matches:,} matches out of {len(df):,} pairs "
          f"({n_matches/len(df)*100:.2f}%)")

    return df


def generate_submission(pred_df, all_test_s1_ids, output_dir):
    """Write matching_results.tsv and candidate_pairs.tsv to output_dir.

    all_test_s1_ids must be the FULL set of S1 entity_ids from test_source1.tsv
    so singletons and zero-candidate entities still get a row — the validator
    rejects any submission missing a Source 1 entity.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- matching_results.tsv ---
    matches = pred_df[pred_df['match_pred'] == 1]
    match_groups = defaultdict(list)
    # zip iteration is 50-100x faster than iterrows() at millions of rows
    for s1_id, cand_id in zip(matches['s1_entity_id'], matches['cand_entity_id']):
        match_groups[s1_id].append(cand_id)

    results_path = output_dir / "matching_results.tsv"
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(f"{config.SUBMIT_COL_S1_ID}\t{config.SUBMIT_COL_MATCHED}\n")
        for s1_id in sorted(all_test_s1_ids):
            matched = ','.join(sorted(set(match_groups.get(s1_id, []))))
            f.write(f"{s1_id}\t{matched}\n")
    print(f"  matching_results.tsv: {len(all_test_s1_ids):,} rows "
          f"({len(match_groups):,} with matches)")

    # --- candidate_pairs.tsv ---
    cand_groups = defaultdict(list)
    for s1_id, cand_id in zip(pred_df['s1_entity_id'], pred_df['cand_entity_id']):
        cand_groups[s1_id].append(cand_id)

    cand_path = output_dir / "candidate_pairs.tsv"
    with open(cand_path, 'w', encoding='utf-8') as f:
        f.write(f"{config.SUBMIT_COL_S1_ID}\t{config.SUBMIT_COL_CANDS}\n")
        for s1_id in sorted(all_test_s1_ids):
            cands = ','.join(sorted(set(cand_groups.get(s1_id, []))))
            f.write(f"{s1_id}\t{cands}\n")
    print(f"  candidate_pairs.tsv: {len(all_test_s1_ids):,} rows")


# ═══════════════════════════════════════════════════════════════════
# PIPELINE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def run_training():
    """Full training pipeline: label → split → train → tune → save.

    Memory checkpoints (50 GB pod, 135M pairs, 19 float32 features):
      After generate_labels + concat:      ~14 GB  (df)
      After split, del df:                 ~14 GB  (train_df + val_df)
      After extracting arrays, del DFs:    ~12 GB  (X_train + X_val + val_meta)
      During LightGBM fit:                 ~20-25 GB  (arrays + LightGBM internals)
      After del X_train:                   ~12-15 GB  (LightGBM model + X_val + val_meta)
    """
    print("=" * 60)
    print("  MODEL TRAINING")
    print("=" * 60)

    feat_dir      = config.DATA_ROOT / "features"
    features_path = feat_dir / "train_features_full.tsv"

    if not features_path.exists():
        features_path = feat_dir / "train_features.tsv"
        if not features_path.exists():
            print(f"  ERROR: No features file found. Run --features first.")
            return
        print(f"  NOTE: Using traditional features only (no E5).")

    # ── Step 1: Generate labels ──────────────────────────────────
    print(f"\n  Step 1: Generate labels")
    df = generate_labels(features_path, config.TRAIN_GT)

    feature_cols = get_feature_columns(df)
    print(f"\n  Using {len(feature_cols)} features: {feature_cols}")

    # ── Step 2: Train/val split ───────────────────────────────────
    print(f"\n  Step 2: Train/Val split")
    train_df, val_df = split_train_val(df)

    # Save scalar stats before freeing DataFrames
    n_train       = len(train_df)
    n_val         = len(val_df)
    train_pos_rate = float(train_df['label'].mean())
    val_pos_rate   = float(val_df['label'].mean())

    del df
    gc.collect()

    # ── Step 3: Extract numpy arrays, free pandas DataFrames ──────
    # CRITICAL: train_df (~10.7 GB) and val_df (~2.7 GB) must be freed
    # BEFORE LightGBM allocates its internal quantized dataset.
    # Holding both pandas + numpy simultaneously (~19 GB) + LightGBM
    # internal memory (~4-8 GB) exceeds the 50 GB pod limit.
    print(f"\n  Step 3: Extracting arrays and freeing DataFrames...")
    X_train  = train_df[feature_cols].values.astype(np.float32)
    y_train  = train_df['label'].values.astype(np.int8)
    n_pos    = int(y_train.sum())
    n_neg    = int((y_train == 0).sum())
    del train_df
    gc.collect()

    X_val    = val_df[feature_cols].values.astype(np.float32)
    y_val    = val_df['label'].values.astype(np.int8)
    # Only keep the 2 columns per_entity_f05 actually needs
    val_meta = val_df[['s1_entity_id', 'label']].copy()
    del val_df
    gc.collect()
    print(f"  DataFrames freed. Peak memory now ~12 GB (arrays only).")

    # ── Step 4: Train LightGBM ────────────────────────────────────
    print(f"\n  Step 4: Train LightGBM")
    model = train_lightgbm(X_train, y_train, X_val, y_val, feature_cols, n_pos, n_neg)

    # Free training arrays — val arrays still needed for threshold tuning
    del X_train, y_train
    gc.collect()

    # ── Step 5: Tune threshold ────────────────────────────────────
    print(f"\n  Step 5: Tune threshold (F0.5)")
    best_threshold, best_fbeta, threshold_results = tune_threshold(
        model, X_val, y_val, val_meta, feature_cols
    )

    # ── Step 6: Save ─────────────────────────────────────────────
    metrics = {
        'best_f0.5':      float(best_fbeta),
        'threshold':      float(best_threshold),
        'n_train':        n_train,
        'n_val':          n_val,
        'train_pos_rate': train_pos_rate,
        'val_pos_rate':   val_pos_rate,
    }
    save_model(model, best_threshold, feature_cols, metrics)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Best F0.5: {best_fbeta:.4f} @ threshold {best_threshold:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    run_training()