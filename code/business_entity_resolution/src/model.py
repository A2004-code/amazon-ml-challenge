"""Stage 10-11: LightGBM Training + F0.5 Threshold Tuning.

Handles:
  1. Label generation — joins features with ground truth to create match labels
  2. Train/val split — GroupShuffleSplit by s1_entity_id (no data leakage)
  3. LightGBM binary classifier training
  4. F0.5 threshold optimization on validation set
  5. Model checkpoint save/load
  6. Prediction on new data

Usage:
    python -m src.model
"""
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
    precision_recall_curve, precision_score, recall_score,
    f1_score, fbeta_score, classification_report, roc_auc_score,
)

from . import config


# ═══════════════════════════════════════════════════════════════════
# LABEL GENERATION
# ═══════════════════════════════════════════════════════════════════

def load_ground_truth_pairs(gt_path):
    """Load ground truth as a set of (s1_id, cand_id) tuples.
    
    Returns:
        set of (s1_id, cand_id) tuples representing true matches
    """
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
    
    Args:
        features_path: path to features TSV (train_features_full.tsv)
        gt_path: path to ground truth TSV
    
    Returns:
        DataFrame with all features + 'label' column
    """
    print(f"  Loading ground truth...")
    gt_pairs = load_ground_truth_pairs(gt_path)
    print(f"  Ground truth: {len(gt_pairs):,} true pairs")
    
    print(f"  Loading features from {features_path.name}...")
    df = pd.read_csv(
        features_path, sep='\t', encoding='utf-8',
        dtype={'s1_entity_id': str, 'cand_entity_id': str},
    )
    print(f"  Feature rows: {len(df):,}")
    
    # Generate labels
    df['label'] = df.apply(
        lambda row: 1 if (row['s1_entity_id'], row['cand_entity_id']) in gt_pairs else 0,
        axis=1
    )
    
    n_pos = df['label'].sum()
    n_neg = len(df) - n_pos
    print(f"  Labels: {n_pos:,} positive ({n_pos/len(df)*100:.2f}%), "
          f"{n_neg:,} negative ({n_neg/len(df)*100:.2f}%)")
    
    return df


# ═══════════════════════════════════════════════════════════════════
# TRAIN / VALIDATION SPLIT
# ═══════════════════════════════════════════════════════════════════

def split_train_val(df, val_ratio=None, random_state=42):
    """Split data into train/val using GroupShuffleSplit on s1_entity_id.
    
    Critical: all pairs for a given S1 entity go entirely into train OR val.
    This prevents data leakage where the model memorizes S1 entity patterns.
    
    Args:
        df: DataFrame with features + labels
        val_ratio: fraction for validation
        random_state: reproducibility seed
    
    Returns:
        (train_df, val_df)
    """
    if val_ratio is None:
        val_ratio = config.VAL_RATIO
    
    groups = df['s1_entity_id'].values
    
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
    train_idx, val_idx = next(splitter.split(df, groups=groups))
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    
    # Verify no S1 entity leaks across splits
    train_s1 = set(train_df['s1_entity_id'].unique())
    val_s1 = set(val_df['s1_entity_id'].unique())
    overlap = train_s1 & val_s1
    assert len(overlap) == 0, f"Data leakage! {len(overlap)} S1 entities in both splits."
    
    print(f"\n  Train/Val Split (GroupShuffleSplit on s1_entity_id):")
    print(f"    Train: {len(train_df):,} pairs, {len(train_s1):,} S1 entities, "
          f"{train_df['label'].sum():,} positives ({train_df['label'].mean()*100:.2f}%)")
    print(f"    Val:   {len(val_df):,} pairs, {len(val_s1):,} S1 entities, "
          f"{val_df['label'].sum():,} positives ({val_df['label'].mean()*100:.2f}%)")
    print(f"    Leakage check: {'PASS' if len(overlap) == 0 else 'FAIL'}")
    
    return train_df, val_df


# ═══════════════════════════════════════════════════════════════════
# FEATURE COLUMNS
# ═══════════════════════════════════════════════════════════════════

def get_feature_columns(df):
    """Get the feature columns present in the DataFrame.
    
    Handles the case where E5 features may not be computed yet.
    """
    available = []
    for col in config.FEATURE_COLS:
        if col in df.columns:
            available.append(col)
        else:
            print(f"  WARNING: Feature '{col}' not found in data — skipping.")
    return available


# ═══════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════

def train_lightgbm(train_df, val_df, feature_cols):
    """Train LightGBM binary classifier.
    
    Args:
        train_df: training DataFrame with features + labels
        val_df: validation DataFrame
        feature_cols: list of feature column names
    
    Returns:
        trained LGBMClassifier model
    """
    print(f"\n{'='*60}")
    print(f"  TRAINING LightGBM")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Train samples: {len(train_df):,}")
    print(f"  Val samples: {len(val_df):,}")
    print(f"{'='*60}")
    
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['label'].values
    
    # Handle class imbalance via scale_pos_weight
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"  Class imbalance: {n_neg:,} neg / {n_pos:,} pos = {scale_pos_weight:.1f}x")
    
    params = config.LGBM_PARAMS.copy()
    params['scale_pos_weight'] = scale_pos_weight
    
    # Extract n_estimators since LGBMClassifier uses it separately
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
    
    # Feature importance
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

def fbeta_at_threshold(y_true, y_prob, threshold, beta=0.5):
    """Compute F-beta score at a given probability threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    if y_pred.sum() == 0:
        return 0.0
    return fbeta_score(y_true, y_pred, beta=beta, zero_division=0)


def tune_threshold(model, val_df, feature_cols, beta=0.5):
    """Find the probability threshold that maximizes F-beta on validation set.
    
    Args:
        model: trained LGBMClassifier
        val_df: validation DataFrame
        feature_cols: feature column names
        beta: beta for F-beta score (0.5 = precision-weighted)
    
    Returns:
        (best_threshold, best_f_beta, results_dict)
    """
    print(f"\n{'='*60}")
    print(f"  THRESHOLD TUNING (F{beta})")
    print(f"{'='*60}")
    
    X_val = val_df[feature_cols].values
    y_val = val_df['label'].values
    
    # Get predicted probabilities
    y_prob = model.predict_proba(X_val)[:, 1]
    
    # Grid search over thresholds
    thresholds = np.arange(
        config.THRESHOLD_SEARCH_MIN,
        config.THRESHOLD_SEARCH_MAX + config.THRESHOLD_SEARCH_STEP,
        config.THRESHOLD_SEARCH_STEP
    )
    
    best_threshold = 0.5
    best_fbeta = 0.0
    results = []
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if y_pred.sum() == 0:
            continue
        
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        fb = fbeta_score(y_val, y_pred, beta=beta, zero_division=0)
        
        results.append({
            'threshold': round(t, 3),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            f'f{beta}': round(fb, 4),
            'predicted_pos': int(y_pred.sum()),
        })
        
        if fb > best_fbeta:
            best_fbeta = fb
            best_threshold = t
    
    # Print top results around the best threshold
    print(f"\n  Threshold search results (around best):")
    print(f"  {'Threshold':>10s} {'Precision':>10s} {'Recall':>10s} {'F0.5':>10s} {'Pred_Pos':>10s}")
    for r in results:
        if abs(r['threshold'] - best_threshold) <= 0.05:
            marker = " <-- BEST" if abs(r['threshold'] - best_threshold) < 0.005 else ""
            print(f"  {r['threshold']:>10.3f} {r['precision']:>10.4f} "
                  f"{r['recall']:>10.4f} {r[f'f{beta}']:>10.4f} "
                  f"{r['predicted_pos']:>10d}{marker}")
    
    print(f"\n  Best threshold: {best_threshold:.3f}")
    print(f"  Best F{beta}:      {best_fbeta:.4f}")
    
    # Full classification report at best threshold
    y_pred_best = (y_prob >= best_threshold).astype(int)
    print(f"\n  Classification Report @ threshold={best_threshold:.3f}:")
    print(classification_report(y_val, y_pred_best, target_names=['No Match', 'Match']))
    
    # AUC-ROC
    try:
        auc = roc_auc_score(y_val, y_prob)
        print(f"  AUC-ROC: {auc:.4f}")
    except ValueError:
        auc = None
    
    return best_threshold, best_fbeta, results


# ═══════════════════════════════════════════════════════════════════
# SAVE / LOAD
# ═══════════════════════════════════════════════════════════════════

def save_model(model, threshold, feature_cols, metrics=None):
    """Save trained model, threshold, and metadata.
    
    Saves to config.MODEL_DIR:
      - lgbm_model.joblib      (model weights)
      - model_metadata.json    (threshold, feature_cols, metrics)
    """
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    model_path = config.MODEL_DIR / "lgbm_model.joblib"
    meta_path = config.MODEL_DIR / "model_metadata.json"
    
    # Save model
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata = {
        'threshold': float(threshold),
        'feature_cols': feature_cols,
        'n_features': len(feature_cols),
        'best_iteration': model.best_iteration_,
        'metrics': metrics or {},
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n  Model saved to {model_path}")
    print(f"  Metadata saved to {meta_path}")
    print(f"  Threshold: {threshold:.3f}")


def load_model():
    """Load trained model and metadata.
    
    Returns:
        (model, threshold, feature_cols)
    """
    model_path = config.MODEL_DIR / "lgbm_model.joblib"
    meta_path = config.MODEL_DIR / "model_metadata.json"
    
    if not model_path.exists():
        raise FileNotFoundError(f"No saved model at {model_path}. Train first.")
    
    model = joblib.load(model_path)
    
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    
    threshold = metadata['threshold']
    feature_cols = metadata['feature_cols']
    
    print(f"  Loaded model from {model_path}")
    print(f"  Threshold: {threshold:.3f}")
    print(f"  Features: {len(feature_cols)}")
    
    return model, threshold, feature_cols


# ═══════════════════════════════════════════════════════════════════
# PREDICTION
# ═══════════════════════════════════════════════════════════════════

def predict(model, df, feature_cols, threshold):
    """Run prediction on a features DataFrame.
    
    Args:
        model: trained LGBMClassifier
        df: DataFrame with feature columns
        feature_cols: list of feature column names
        threshold: probability threshold for match decision
    
    Returns:
        DataFrame with added 'match_prob' and 'match_pred' columns
    """
    X = df[feature_cols].values
    
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    df = df.copy()
    df['match_prob'] = probs
    df['match_pred'] = preds
    
    n_matches = preds.sum()
    print(f"  Predictions: {n_matches:,} matches out of {len(df):,} pairs "
          f"({n_matches/len(df)*100:.2f}%)")
    
    return df


def generate_submission(pred_df, output_dir):
    """Generate submission files from prediction results.
    
    Writes:
      - matching_results.tsv:  s1_entity_id -> comma-separated matched IDs
      - candidate_pairs.tsv:   all candidate pairs considered
    
    Args:
        pred_df: DataFrame with predictions (match_pred, s1_entity_id, cand_entity_id)
        output_dir: directory to write output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # matching_results.tsv: only matched pairs, grouped by S1 entity
    matches = pred_df[pred_df['match_pred'] == 1]
    
    match_groups = defaultdict(list)
    for _, row in matches.iterrows():
        match_groups[row['s1_entity_id']].append(row['cand_entity_id'])
    
    results_path = output_dir / "matching_results.tsv"
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("s1_entity_id\tmatched_entity_ids\n")
        for s1_id in sorted(match_groups.keys()):
            matched = ','.join(match_groups[s1_id])
            f.write(f"{s1_id}\t{matched}\n")
    
    print(f"  matching_results.tsv: {len(match_groups):,} S1 entities with matches")
    
    # candidate_pairs.tsv: all candidate pairs with scores
    cand_path = output_dir / "candidate_pairs.tsv"
    cand_cols = ['s1_entity_id', 'cand_entity_id', 'cand_source', 'country', 'match_prob', 'match_pred']
    available_cols = [c for c in cand_cols if c in pred_df.columns]
    pred_df[available_cols].to_csv(cand_path, sep='\t', index=False, encoding='utf-8')
    
    print(f"  candidate_pairs.tsv: {len(pred_df):,} total pairs")
    print(f"  Output directory: {output_dir}")


# ═══════════════════════════════════════════════════════════════════
# PIPELINE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def run_training():
    """Full training pipeline: label -> split -> train -> tune -> save."""
    print("=" * 60)
    print("  MODEL TRAINING")
    print("=" * 60)
    
    feat_dir = config.DATA_ROOT / "features"
    features_path = feat_dir / "train_features_full.tsv"
    
    # Fallback: if E5 features not computed yet, use traditional only
    if not features_path.exists():
        features_path = feat_dir / "train_features.tsv"
        if not features_path.exists():
            print(f"  ERROR: No features file found. Run --features first.")
            return
        print(f"  NOTE: Using traditional features only (no E5). "
              f"Run --embed for full features.")
    
    # 1. Generate labels
    print(f"\n  Step 1: Generate labels")
    df = generate_labels(features_path, config.TRAIN_GT)
    
    # 2. Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"\n  Using {len(feature_cols)} features: {feature_cols}")
    
    # 3. Train/val split
    print(f"\n  Step 2: Train/Val split")
    train_df, val_df = split_train_val(df)
    
    # 4. Train model
    print(f"\n  Step 3: Train LightGBM")
    model = train_lightgbm(train_df, val_df, feature_cols)
    
    # 5. Tune threshold
    print(f"\n  Step 4: Tune threshold (F0.5)")
    best_threshold, best_fbeta, threshold_results = tune_threshold(
        model, val_df, feature_cols
    )
    
    # 6. Save model
    metrics = {
        'best_f0.5': float(best_fbeta),
        'threshold': float(best_threshold),
        'n_train': len(train_df),
        'n_val': len(val_df),
        'train_pos_rate': float(train_df['label'].mean()),
        'val_pos_rate': float(val_df['label'].mean()),
    }
    save_model(model, best_threshold, feature_cols, metrics)
    
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Best F0.5: {best_fbeta:.4f} @ threshold {best_threshold:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    run_training()
