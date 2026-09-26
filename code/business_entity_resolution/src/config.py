"""Configuration: paths, hyperparameters, constants.

Supports both local (Windows) and RunPod (Linux) environments.
Set ER_PROJECT_ROOT env var to override auto-detection.
"""
import os
from pathlib import Path

# ── Project Root ──────────────────────────────────────────────────
# Auto-detect: config.py is at code/business_entity_resolution/src/config.py
# Project root is 4 levels up (amazon_ml_challange/)
PROJECT_ROOT = Path(os.environ.get(
    'ER_PROJECT_ROOT',
    Path(__file__).resolve().parent.parent.parent.parent
))

# ── Data Paths ────────────────────────────────────────────────────
DATA_ROOT = PROJECT_ROOT / "amazon_dataset" / "student_resource" / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_DIR = PROJECT_ROOT / "code" / "business_entity_resolution" / "models"

# Training data
TRAIN_S1 = DATA_ROOT / "train" / "train_source1.tsv"
TRAIN_S2 = DATA_ROOT / "train" / "train_source2.tsv"
TRAIN_S3 = DATA_ROOT / "train" / "train_source3.tsv"
TRAIN_GT = DATA_ROOT / "train" / "train_ground_truth.tsv"

# Test data
TEST_S1 = DATA_ROOT / "test" / "test_source1.tsv"
TEST_S2 = DATA_ROOT / "test" / "test_source2.tsv"
TEST_S3 = DATA_ROOT / "test" / "test_source3.tsv"

# Validation script
VALIDATOR = PROJECT_ROOT / "amazon_dataset" / "student_resource" / "utils" / "validate_submission.py"

# ── Column Names ──────────────────────────────────────────────────
COL_ENTITY_ID = "entity_id"
COL_NAME = "business_name"
COL_ADDRESS = "business_address"
COL_COUNTRY = "country"

# ── Cleaning Constants ────────────────────────────────────────────
# Zero-width characters to strip during cleaning
ZERO_WIDTH_CHARS = [
    '\u200c',  # Zero-Width Non-Joiner (ZWNJ)
    '\u200b',  # Zero-Width Space
    '\u200d',  # Zero-Width Joiner (ZWJ)
    '\ufeff',  # BOM / Zero-Width No-Break Space
    '\u200e',  # Left-to-Right Mark
    '\u200f',  # Right-to-Left Mark
]

# ── Blocking Hyperparameters ──────────────────────────────────────
# Max posting list size: tokens appearing in more candidates than this
# are too common to be discriminative — skip them during lookup.
BLOCKING_MAX_POSTINGS = 5_000

# Minimum shared tokens to consider a candidate pair
BLOCKING_MIN_NAME_OVERLAP = 1

# Hard cap on candidates per S1 entity (take top-K by overlap score)
BLOCKING_MAX_CANDIDATES = 100

# Minimum token length to include in the inverted index
BLOCKING_MIN_TOKEN_LEN = 2

# Tokens to skip during blocking (too common, not discriminative)
BLOCKING_STOPWORDS = {
    'and', 'the', 'of', 'or', 'for', 'in', 'at', 'to', 'on', 'by',
    'an', 'is', 'it', 'no', 'as', 'do',
    # Common legal/business words (after expansion)
    'private', 'limited', 'incorporated', 'corporation', 'company',
    'llc', 'llp', 'plc', 'sarl', 'sas', 'sa',
    # Common address words
    'road', 'street', 'avenue', 'drive', 'lane', 'floor', 'building',
    'apartment', 'suite', 'number', 'near', 'opposite',
    'district', 'block', 'sector', 'plot', 'new', 'old',
    # Common Indian words in normalized form
    'nagar', 'marg', 'colony', 'vihar', 'puri', 'pur',
    'market', 'bazaar', 'chowk',
    # Common French
    'rue', 'les', 'des', 'sur', 'par', 'avec', 'dans',
    'cedex', 'bp',
}

# ── E5 Embedding Hyperparameters ──────────────────────────────────
E5_MODEL_NAME = 'intfloat/multilingual-e5-base'
E5_BATCH_SIZE = 512           # encoding batch size on GPU
E5_MAX_SEQ_LENGTH = 128       # max tokens per input (names are short)
E5_EMBEDDING_DIM = 768        # output dimension for e5-base
E5_CACHE_DIR = MODEL_DIR / "embeddings_cache"  # cache .npy files

# ── Feature Engineering ───────────────────────────────────────────
FEATURE_COLS = [
    # Name features (9)
    'name_jaccard', 'name_levenshtein', 'name_jaro_winkler',
    'name_token_sort', 'name_token_set',
    'core_name_jaccard', 'core_name_levenshtein', 'core_name_jaro_winkler',
    'name_len_ratio',
    # Address features (5)
    'addr_jaccard', 'addr_levenshtein',
    'addr_token_overlap', 'addr_numeric_overlap', 'addr_missing',
    # Blocking features (3)
    'name_overlap', 'addr_overlap', 'postal_match',
    # E5 embedding features (2)
    'e5_name_cosine', 'e5_addr_cosine',
]

# ── LightGBM Hyperparameters ─────────────────────────────────────
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'verbosity': -1,
    'n_estimators': 800,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': -1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 50,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
}

# ── Training Hyperparameters ─────────────────────────────────────
VAL_RATIO = 0.2              # fraction for validation (GroupShuffleSplit)
THRESHOLD_SEARCH_MIN = 0.30  # F0.5 threshold grid search range
THRESHOLD_SEARCH_MAX = 0.90
THRESHOLD_SEARCH_STEP = 0.01

def print_config():
    """Print resolved paths for verification."""
    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"DATA_ROOT    : {DATA_ROOT}")
    print(f"OUTPUT_DIR   : {OUTPUT_DIR}")
    print(f"MODEL_DIR    : {MODEL_DIR}")
    print(f"TRAIN_S1     : {TRAIN_S1}  (exists: {TRAIN_S1.exists()})")
    print(f"TRAIN_GT     : {TRAIN_GT}  (exists: {TRAIN_GT.exists()})")
    print(f"TEST_S1      : {TEST_S1}  (exists: {TEST_S1.exists()})")


if __name__ == "__main__":
    print_config()
