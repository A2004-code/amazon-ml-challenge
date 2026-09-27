#!/usr/bin/env python3
"""CLI entry point for Business Entity Resolution.

Usage:
    # Individual stages (train mode)
    python run.py --eda
    python run.py --clean
    python run.py --normalize
    python run.py --block
    python run.py --features
    python run.py --embed
    python run.py --train
    
    # Full pipelines
    python run.py --all              # full training pipeline (stages 1-7)
    python run.py --predict-all      # full test/inference pipeline
    python run.py --predict          # predict only (requires trained model + test features)
    
    # Combo
    python run.py --all --predict-all  # train then predict
"""
import sys
import os
import argparse

# Add package directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config


def main():
    parser = argparse.ArgumentParser(
        description="Business Entity Resolution Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --eda              # EDA only
  python run.py --clean --normalize  # clean + normalize
  python run.py --all              # full training pipeline
  python run.py --predict-all      # full test pipeline
  python run.py --all --predict-all  # end-to-end
        """
    )
    
    # Individual stages
    parser.add_argument('--eda', action='store_true', help="Run exploratory data analysis")
    parser.add_argument('--clean', action='store_true', help="Run data cleaning")
    parser.add_argument('--normalize', action='store_true', help="Run normalization")
    parser.add_argument('--block', action='store_true', help="Run blocking")
    parser.add_argument('--features', action='store_true', help="Compute traditional features")
    parser.add_argument('--embed', action='store_true', help="Compute E5 embedding features")
    parser.add_argument('--train', action='store_true', help="Train LightGBM model")
    parser.add_argument('--predict', action='store_true', help="Predict (requires model + test features)")
    
    # Full pipelines
    parser.add_argument('--all', action='store_true', help="Full training pipeline (stages 1-7)")
    parser.add_argument('--predict-all', action='store_true', help="Full test/inference pipeline")
    
    # Options
    parser.add_argument('--mode', choices=['train', 'test'], default='train',
                        help="Mode for individual stages (default: train)")
    parser.add_argument('--force', action='store_true',
                        help="Force recompute even if output exists")
    
    args = parser.parse_args()

    # If nothing specified, show help
    if not any([args.eda, args.clean, args.normalize, args.block,
                args.features, args.embed, args.train, args.predict,
                args.all, args.predict_all]):
        parser.print_help()
        return

    # ── Full Training Pipeline ────────────────────────────────────
    if args.all:
        from src.pipeline import run_train_pipeline
        run_train_pipeline(skip_existing=not args.force)
    else:
        # ── Individual stages ─────────────────────────────────────
        if args.eda:
            from src.eda import run_eda
            run_eda()

        if args.clean:
            from src.cleaning import run_cleaning
            run_cleaning()

        if args.normalize:
            from src.normalization import run_normalization
            run_normalization()

        if args.block:
            from src.blocking import run_blocking
            run_blocking(mode=args.mode)

        if args.features:
            from src.features import run_features
            run_features(mode=args.mode)

        if args.embed:
            from src.embeddings import run_embeddings
            run_embeddings(mode=args.mode)

        if args.train:
            from src.model import run_training
            run_training()

        if args.predict:
            from src.model import load_model, predict, generate_submission
            import pandas as pd
            model, threshold, feature_cols = load_model()
            feat_dir = config.DATA_ROOT / "features"
            feat_path = feat_dir / "test_features_full.tsv"
            if not feat_path.exists():
                feat_path = feat_dir / "test_features.tsv"
            if not feat_path.exists():
                print(f"ERROR: No test features found at {feat_dir}")
                print("Run: python run.py --block --features --embed --mode test")
                return
            
            print(f"  Loading test features from {feat_path.name} in chunks to prevent memory crash...")
            chunksize = 5_000_000
            reader = pd.read_csv(feat_path, sep='\t', encoding='utf-8',
                                 dtype={'s1_entity_id': str, 'cand_entity_id': str},
                                 chunksize=chunksize)
            
            preds_list = []
            all_s1_ids = set()
            processed = 0
            
            for chunk in reader:
                chunk_pred = predict(model, chunk, feature_cols, threshold)
                # Keep only what we need for submission to save RAM
                preds_list.append(chunk_pred[['s1_entity_id', 'cand_entity_id', 'match_pred']])
                all_s1_ids.update(chunk['s1_entity_id'].unique())
                
                processed += len(chunk)
                print(f"    Predicted {processed:,} pairs...")
                
            print("  Concatenating all predictions...")
            pred_df = pd.concat(preds_list, ignore_index=True)
            
            print("  Generating Kaggle submission files...")
            generate_submission(pred_df, all_s1_ids, config.OUTPUT_DIR)

    # ── Full Test Pipeline ────────────────────────────────────────
    if args.predict_all:
        from src.pipeline import run_test_pipeline
        run_test_pipeline(skip_existing=not args.force)


if __name__ == '__main__':
    main()
