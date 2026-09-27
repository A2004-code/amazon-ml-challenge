import pandas as pd
import json
import joblib
from pathlib import Path
from src import config
from src.model import load_ground_truth_pairs

def run_diagnostics():
    print("Loading Validation Split...")
    
    # 1. We need the original ground truth pairs
    gt_pairs = load_ground_truth_pairs(config.TRAIN_GT)
    
    # 2. We need the validation features
    from src.model import generate_labels, split_train_val, get_feature_columns
    feat_dir = config.DATA_ROOT / "features"
    features_path = feat_dir / "train_features_full.tsv"
    
    df = generate_labels(features_path, config.TRAIN_GT)
    _, val_df = split_train_val(df)
    
    val_meta = val_df[['s1_entity_id', 'cand_entity_id', 'label']].copy()
    feature_cols = get_feature_columns(val_df)
    
    print("Loading Trained LightGBM Model...")
    model_path = config.MODEL_DIR / "lgbm_model.joblib"
    meta_path = config.MODEL_DIR / "model_metadata.json"
    
    model = joblib.load(model_path)
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    threshold = metadata['threshold']
    
    print("Predicting on Validation Set...")
    X_val = val_df[feature_cols].values
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    val_meta['pred'] = y_pred
    
    # --- DIAGNOSTIC 1: BLOCKING RECALL ---
    val_s1_entities = set(val_meta['s1_entity_id'].unique())
    
    total_true_pairs_in_val = sum(1 for (s1, cand) in gt_pairs if s1 in val_s1_entities)
    survived_true_pairs = val_meta['label'].sum()
    
    blocking_recall = survived_true_pairs / total_true_pairs_in_val if total_true_pairs_in_val > 0 else 0
    print(f"\n--- DIAGNOSTIC 1: BLOCKING RECALL ---")
    print(f"Total True Pairs for Val S1 Entities: {total_true_pairs_in_val:,}")
    print(f"True Pairs Survived Blocking:         {survived_true_pairs:,}")
    print(f"BLOCKING RECALL:                      {blocking_recall:.2%} (Limit of F0.5)")
    
    # --- DIAGNOSTIC 2: FALSE POSITIVE RATE ---
    predicted_positives = val_meta['pred'].sum()
    false_positives = len(val_meta[(val_meta['pred'] == 1) & (val_meta['label'] == 0)])
    
    fp_rate = false_positives / predicted_positives if predicted_positives > 0 else 0
    print(f"\n--- DIAGNOSTIC 2: FALSE POSITIVES ---")
    print(f"Predicted Matches: {predicted_positives:,}")
    print(f"False Positives:   {false_positives:,}")
    print(f"FALSE POS RATE:    {fp_rate:.2%}")
    
    # --- DIAGNOSTIC 3: COUNTRY BREAKDOWN ---
    print("\n--- DIAGNOSTIC 3: COUNTRY BREAKDOWN ---")
    print("Loading raw source data for country codes...")
    raw_s1 = pd.read_csv(config.TRAIN_S1, sep='\t', usecols=['entity_id', 'country'])
    
    val_with_country = pd.merge(val_meta, raw_s1, left_on='s1_entity_id', right_on='entity_id', how='left')
    
    country_stats = []
    for country, group in val_with_country.groupby('country'):
        c_survived = group['label'].sum()
        c_pred_pos = group['pred'].sum()
        c_tp = len(group[(group['label'] == 1) & (group['pred'] == 1)])
        
        c_precision = c_tp / c_pred_pos if c_pred_pos > 0 else 0
        c_recall = c_tp / c_survived if c_survived > 0 else 0
        
        if c_precision == 0 and c_recall == 0:
            c_f05 = 0
        else:
            c_f05 = 1.25 * (c_precision * c_recall) / (0.25 * c_precision + c_recall)
            
        country_stats.append({
            'Country': country,
            'S1 Entities': group['s1_entity_id'].nunique(),
            'Precision': round(c_precision, 4),
            'Recall (of blocked)': round(c_recall, 4),
            'Approx F0.5': round(c_f05, 4)
        })
        
    stats_df = pd.DataFrame(country_stats).sort_values('Approx F0.5')
    print(stats_df.to_string(index=False))

if __name__ == '__main__':
    run_diagnostics()
