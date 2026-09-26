"""End-to-end pipeline orchestration: training and inference.

Provides two high-level functions:
  - run_train_pipeline(): EDA -> Clean -> Normalize -> Block -> Features -> E5 -> Train
  - run_test_pipeline():  Clean -> Normalize -> Block -> Features -> E5 -> Predict -> Submit

Each stage checks if its output already exists and can optionally skip.

Usage:
    python run.py --all                # full train pipeline
    python run.py --predict-all        # full test/inference pipeline
"""
import os
import time
from pathlib import Path

from . import config


def run_train_pipeline(skip_existing=True):
    """Run the complete training pipeline end-to-end.
    
    Stages:
      1. EDA
      2. Data Cleaning
      3. Normalization
      4. Blocking + Recall Check
      5. Traditional Feature Engineering
      6. E5 Embeddings
      7. LightGBM Training + Threshold Tuning
    """
    print("=" * 70)
    print("  FULL TRAINING PIPELINE")
    print("=" * 70)
    t0 = time.time()
    
    clean_dir = config.DATA_ROOT / "cleaned"
    norm_dir = config.DATA_ROOT / "normalized"
    block_dir = config.DATA_ROOT / "blocked"
    feat_dir = config.DATA_ROOT / "features"
    
    # ── Stage 1: EDA ──────────────────────────────────────────────
    print("\n[1/7] EDA")
    from .eda import run_eda
    run_eda()
    
    # ── Stage 2: Data Cleaning ────────────────────────────────────
    print("\n[2/7] DATA CLEANING")
    s1_clean = clean_dir / "train_source1_clean.tsv"
    if skip_existing and s1_clean.exists():
        print(f"  SKIP: Cleaned files already exist at {clean_dir}")
    else:
        from .cleaning import run_cleaning
        run_cleaning()
    
    # ── Stage 3: Normalization ────────────────────────────────────
    print("\n[3/7] NORMALIZATION")
    s1_norm = norm_dir / "train_source1_norm.tsv"
    if skip_existing and s1_norm.exists():
        print(f"  SKIP: Normalized files already exist at {norm_dir}")
    else:
        from .normalization import run_normalization
        run_normalization()
    
    # ── Stage 4: Blocking + Recall Check ──────────────────────────
    print("\n[4/7] BLOCKING")
    pairs_path = block_dir / "train_candidate_pairs.tsv"
    if skip_existing and pairs_path.exists():
        print(f"  SKIP: Candidate pairs already exist at {pairs_path}")
    else:
        from .blocking import run_blocking
        run_blocking(mode='train')
    
    # ── Stage 5: Traditional Features ─────────────────────────────
    print("\n[5/7] FEATURE ENGINEERING")
    trad_feat_path = feat_dir / "train_features.tsv"
    if skip_existing and trad_feat_path.exists():
        print(f"  SKIP: Traditional features already exist at {trad_feat_path}")
    else:
        from .features import run_features
        run_features(mode='train')
    
    # ── Stage 6: E5 Embeddings ────────────────────────────────────
    print("\n[6/7] E5 EMBEDDINGS")
    full_feat_path = feat_dir / "train_features_full.tsv"
    if skip_existing and full_feat_path.exists():
        print(f"  SKIP: Full features already exist at {full_feat_path}")
    else:
        from .embeddings import run_embeddings
        run_embeddings(mode='train')
    
    # ── Stage 7: Training ─────────────────────────────────────────
    print("\n[7/7] MODEL TRAINING")
    model_path = config.MODEL_DIR / "lgbm_model.joblib"
    if skip_existing and model_path.exists():
        print(f"  SKIP: Trained model already exists at {model_path}")
    else:
        from .model import run_training
        run_training()
    
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  TRAINING PIPELINE COMPLETE ({elapsed:.0f}s / {elapsed/60:.1f}min)")
    print("=" * 70)


def run_test_pipeline(skip_existing=True):
    """Run the complete test/inference pipeline.
    
    Requires a trained model (run train pipeline first).
    
    Stages:
      1. Clean test data
      2. Normalize test data
      3. Block test data
      4. Traditional features on test pairs
      5. E5 embeddings on test pairs
      6. Predict with trained model
      7. Generate submission files
    """
    print("=" * 70)
    print("  FULL TEST/INFERENCE PIPELINE")
    print("=" * 70)
    t0 = time.time()
    
    # Verify trained model exists
    model_path = config.MODEL_DIR / "lgbm_model.joblib"
    if not model_path.exists():
        print(f"  ERROR: No trained model found at {model_path}")
        print(f"  Run the training pipeline first: python run.py --all")
        return
    
    clean_dir = config.DATA_ROOT / "cleaned"
    norm_dir = config.DATA_ROOT / "normalized"
    block_dir = config.DATA_ROOT / "blocked"
    feat_dir = config.DATA_ROOT / "features"
    
    # ── Stage 1: Clean test data ──────────────────────────────────
    print("\n[1/7] CLEANING TEST DATA")
    test_clean = clean_dir / "test_source1_clean.tsv"
    if skip_existing and test_clean.exists():
        print(f"  SKIP: Cleaned test files already exist")
    else:
        from .cleaning import clean_source_file, deduplicate_across_chunks
        test_files = [
            (config.TEST_S1, clean_dir / "test_source1_clean.tsv", "Test S1"),
            (config.TEST_S2, clean_dir / "test_source2_clean.tsv", "Test S2"),
            (config.TEST_S3, clean_dir / "test_source3_clean.tsv", "Test S3"),
        ]
        for inp, out, name in test_files:
            stats = clean_source_file(inp, out, name)
            if stats:
                deduplicate_across_chunks(out, name)
    
    # ── Stage 2: Normalize test data ──────────────────────────────
    print("\n[2/7] NORMALIZING TEST DATA")
    test_norm = norm_dir / "test_source1_norm.tsv"
    if skip_existing and test_norm.exists():
        print(f"  SKIP: Normalized test files already exist")
    else:
        from .normalization import normalize_source_file
        os.makedirs(norm_dir, exist_ok=True)
        test_norm_files = [
            (clean_dir / "test_source1_clean.tsv", norm_dir / "test_source1_norm.tsv", "Test S1"),
            (clean_dir / "test_source2_clean.tsv", norm_dir / "test_source2_norm.tsv", "Test S2"),
            (clean_dir / "test_source3_clean.tsv", norm_dir / "test_source3_norm.tsv", "Test S3"),
        ]
        for inp, out, name in test_norm_files:
            normalize_source_file(inp, out, name)
    
    # ── Stage 3: Block test data ──────────────────────────────────
    print("\n[3/7] BLOCKING TEST DATA")
    test_pairs = block_dir / "test_candidate_pairs.tsv"
    if skip_existing and test_pairs.exists():
        print(f"  SKIP: Test candidate pairs already exist")
    else:
        from .blocking import run_blocking
        run_blocking(mode='test')
    
    # ── Stage 4: Traditional features ─────────────────────────────
    print("\n[4/7] TEST FEATURE ENGINEERING")
    test_feat = feat_dir / "test_features.tsv"
    if skip_existing and test_feat.exists():
        print(f"  SKIP: Test features already exist")
    else:
        from .features import run_features
        run_features(mode='test')
    
    # ── Stage 5: E5 embeddings ────────────────────────────────────
    print("\n[5/7] TEST E5 EMBEDDINGS")
    test_feat_full = feat_dir / "test_features_full.tsv"
    if skip_existing and test_feat_full.exists():
        print(f"  SKIP: Test full features already exist")
    else:
        from .embeddings import run_embeddings
        run_embeddings(mode='test')
    
    # ── Stage 6: Predict ──────────────────────────────────────────
    print("\n[6/7] PREDICTION")
    import pandas as pd
    from .model import load_model, predict, generate_submission
    
    model, threshold, feature_cols = load_model()
    
    feat_path = test_feat_full if test_feat_full.exists() else test_feat
    print(f"  Loading test features from {feat_path.name}")
    test_df = pd.read_csv(
        feat_path, sep='\t', encoding='utf-8',
        dtype={'s1_entity_id': str, 'cand_entity_id': str}
    )
    
    pred_df = predict(model, test_df, feature_cols, threshold)
    
    # ── Stage 7: Generate submission ──────────────────────────────
    print("\n[7/7] GENERATING SUBMISSION FILES")
    generate_submission(pred_df, config.OUTPUT_DIR)
    
    # Also copy candidate_pairs to output/
    import shutil
    if test_pairs.exists():
        shutil.copy2(test_pairs, config.OUTPUT_DIR / "candidate_pairs.tsv")
        print(f"  Copied blocking candidate_pairs.tsv to output/")
    
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  TEST PIPELINE COMPLETE ({elapsed:.0f}s / {elapsed/60:.1f}min)")
    print(f"  Submission files in: {config.OUTPUT_DIR}")
    print("=" * 70)
