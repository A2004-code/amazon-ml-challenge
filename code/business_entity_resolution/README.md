# Business Entity Resolution

## Overview
Entity resolution pipeline for matching business entities across three data sources (S1, S2, S3). Uses a combination of token-based blocking, string similarity features, multilingual E5 embeddings, and LightGBM classification.

## Requirements
- Python 3.10+
- GPU (NVIDIA L4 recommended) for E5 embeddings
- ~32GB RAM minimum (62GB recommended for full dataset)

## Setup

```bash
cd code/business_entity_resolution
pip install -r requirements.txt
```

## Quick Start — Full Pipeline

```bash
# Train on training data (all 7 stages)
python run.py --all

# Run inference on test data (generates submission files)
python run.py --predict-all

# Or end-to-end in one command
python run.py --all --predict-all
```

## Step-by-Step Execution

```bash
# Stage 1: Exploratory Data Analysis
python run.py --eda

# Stage 2: Data Cleaning
python run.py --clean

# Stage 3: Text Normalization
python run.py --normalize

# Stage 4: Blocking + Recall Check
python run.py --block

# Stage 5: Traditional Feature Engineering (CPU)
python run.py --features

# Stage 6: E5 Embedding Features (GPU)
python run.py --embed

# Stage 7: LightGBM Training + F0.5 Threshold Tuning
python run.py --train

# Stage 8: Prediction on Test Data
python run.py --block --features --embed --mode test
python run.py --predict
```

## Pipeline Architecture

```
Raw Data (S1, S2, S3)
    |
    v
[Cleaning] -> Strip whitespace, ZWNJ, dedup, handle nulls
    |
    v
[Normalization] -> Lowercase, accent strip, expand legal suffixes
    |                Produces: norm_name, core_name, norm_address, postal_code
    v
[Blocking] -> 3-channel inverted index (name tokens + addr tokens + postal)
    |           Country-partitioned, max_postings cap, recall checkpoint
    v
Candidate Pairs (~100 per S1 entity)
    |
    +---> [Traditional Features] 17 features (rapidfuzz, CPU)
    |
    +---> [E5 Embeddings] 2 features (multilingual-e5-base, GPU)
    |
    v
[LightGBM] -> Binary classifier, GroupShuffleSplit validation
    |
    v
[Threshold Tuning] -> Optimize F0.5
    |
    v
matching_results.tsv + candidate_pairs.tsv
```

## Output

After running the full pipeline, submission files are in `output/`:
- `matching_results.tsv` — final entity matches
- `candidate_pairs.tsv` — all candidate pairs with match probabilities

## Intermediate Data

All intermediate files are stored under `amazon_dataset/student_resource/dataset/`:
- `cleaned/` — cleaned TSVs
- `normalized/` — normalized TSVs with derived columns
- `blocked/` — candidate pair TSVs
- `features/` — feature matrices

## CLI Reference

```
python run.py --help

Options:
  --eda           Run EDA
  --clean         Data cleaning
  --normalize     Text normalization
  --block         Blocking (candidate generation)
  --features      Traditional feature engineering
  --embed         E5 embedding features
  --train         Train LightGBM
  --predict       Predict (requires model + test features)
  --all           Full training pipeline
  --predict-all   Full test/inference pipeline
  --mode {train,test}  Mode for individual stages
  --force         Recompute even if output exists
```
