"""Stage 5: Traditional Feature Engineering — string similarity, token overlap.

Computes 17 pairwise features for each candidate pair using rapidfuzz (C++ backend)
and simple token/numeric operations. All CPU-based, no GPU required.

Input:  candidate_pairs.tsv (from blocking) + normalized source files
Output: features TSV with 17 columns per pair

Usage:
    python -m src.features
"""
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler

from . import config


# ═══════════════════════════════════════════════════════════════════
# SIMILARITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def jaccard_similarity(tokens_a, tokens_b):
    """Jaccard similarity between two sets of tokens."""
    if not tokens_a and not tokens_b:
        return 0.0
    a = set(tokens_a)
    b = set(tokens_b)
    intersection = a & b
    union = a | b
    if not union:
        return 0.0
    return len(intersection) / len(union)


def token_overlap_count(tokens_a, tokens_b):
    """Count of shared tokens between two sets."""
    if not tokens_a or not tokens_b:
        return 0
    return len(set(tokens_a) & set(tokens_b))


def numeric_overlap(text_a, text_b):
    """Count shared numeric substrings (street numbers, postal codes)."""
    if not text_a or not text_b:
        return 0
    nums_a = set(re.findall(r'\d+', text_a))
    nums_b = set(re.findall(r'\d+', text_b))
    if not nums_a or not nums_b:
        return 0
    return len(nums_a & nums_b)


def length_ratio(text_a, text_b):
    """Ratio of shorter to longer string length."""
    len_a = len(text_a) if text_a else 0
    len_b = len(text_b) if text_b else 0
    if max(len_a, len_b) == 0:
        return 0.0
    return min(len_a, len_b) / max(len_a, len_b)


def safe_tokenize(text):
    """Split text into tokens for similarity computation."""
    if not isinstance(text, str) or not text.strip():
        return []
    return text.lower().split()


# ═══════════════════════════════════════════════════════════════════
# PAIR FEATURE COMPUTATION
# ═══════════════════════════════════════════════════════════════════

def compute_pair_features_row(s1_name, s1_core, s1_addr,
                               cand_name, cand_core, cand_addr,
                               name_overlap, addr_overlap, postal_match):
    """Compute all traditional features for a single (S1, candidate) pair.
    
    Args:
        s1_name: normalized S1 business name
        s1_core: S1 core name (legal suffixes stripped)
        s1_addr: normalized S1 address
        cand_name: normalized candidate business name
        cand_core: candidate core name
        cand_addr: normalized candidate address
        name_overlap: name token overlap from blocking
        addr_overlap: address token overlap from blocking
        postal_match: postal code match from blocking
    
    Returns:
        dict of feature_name -> float
    """
    # Default empty strings
    s1_name = s1_name or ''
    s1_core = s1_core or ''
    s1_addr = s1_addr or ''
    cand_name = cand_name or ''
    cand_core = cand_core or ''
    cand_addr = cand_addr or ''
    
    s1_name_tokens = safe_tokenize(s1_name)
    cand_name_tokens = safe_tokenize(cand_name)
    s1_core_tokens = safe_tokenize(s1_core)
    cand_core_tokens = safe_tokenize(cand_core)
    s1_addr_tokens = safe_tokenize(s1_addr)
    cand_addr_tokens = safe_tokenize(cand_addr)
    
    features = {}
    
    # ── Name Features (9) ─────────────────────────────────────────
    features['name_jaccard'] = jaccard_similarity(s1_name_tokens, cand_name_tokens)
    features['name_levenshtein'] = fuzz.ratio(s1_name, cand_name) / 100.0 if (s1_name and cand_name) else 0.0
    features['name_jaro_winkler'] = JaroWinkler.normalized_similarity(s1_name, cand_name) if (s1_name and cand_name) else 0.0
    features['name_token_sort'] = fuzz.token_sort_ratio(s1_name, cand_name) / 100.0 if (s1_name and cand_name) else 0.0
    features['name_token_set'] = fuzz.token_set_ratio(s1_name, cand_name) / 100.0 if (s1_name and cand_name) else 0.0
    
    features['core_name_jaccard'] = jaccard_similarity(s1_core_tokens, cand_core_tokens)
    features['core_name_levenshtein'] = fuzz.ratio(s1_core, cand_core) / 100.0 if (s1_core and cand_core) else 0.0
    features['core_name_jaro_winkler'] = JaroWinkler.normalized_similarity(s1_core, cand_core) if (s1_core and cand_core) else 0.0
    
    features['name_len_ratio'] = length_ratio(s1_name, cand_name)
    
    # ── Address Features (5) ──────────────────────────────────────
    features['addr_jaccard'] = jaccard_similarity(s1_addr_tokens, cand_addr_tokens)
    features['addr_levenshtein'] = fuzz.ratio(s1_addr, cand_addr) / 100.0 if (s1_addr and cand_addr) else 0.0
    features['addr_token_overlap'] = token_overlap_count(s1_addr_tokens, cand_addr_tokens)
    features['addr_numeric_overlap'] = numeric_overlap(s1_addr, cand_addr)
    features['addr_missing'] = 1.0 if (not s1_addr.strip() or not cand_addr.strip()) else 0.0
    
    # ── Blocking Features (3, passed through) ─────────────────────
    features['name_overlap'] = float(name_overlap)
    features['addr_overlap'] = float(addr_overlap)
    features['postal_match'] = float(postal_match)
    
    return features


# ═══════════════════════════════════════════════════════════════════
# ENTITY LOOKUP
# ═══════════════════════════════════════════════════════════════════

def load_entity_lookup(filepath):
    """Load normalized entities into a dict for fast lookup.
    
    Returns:
        dict: entity_id -> {norm_name, core_name, norm_address, ...}
    """
    lookup = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        col_map = {col: i for i, col in enumerate(header)}
        
        eid_i = col_map['entity_id']
        norm_name_i = col_map.get('norm_name')
        core_name_i = col_map.get('core_name')
        norm_addr_i = col_map.get('norm_address')
        
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) <= eid_i:
                continue
            eid = parts[eid_i]
            lookup[eid] = {
                'norm_name': parts[norm_name_i] if (norm_name_i is not None and norm_name_i < len(parts)) else '',
                'core_name': parts[core_name_i] if (core_name_i is not None and core_name_i < len(parts)) else '',
                'norm_address': parts[norm_addr_i] if (norm_addr_i is not None and norm_addr_i < len(parts)) else '',
            }
    
    return lookup


def load_entity_lookups_for_country(norm_dir, mode, country):
    """Load S1 and candidate entity lookups for a specific country.
    
    Filters entities to the specified country to save memory.
    
    Args:
        norm_dir: directory with normalized TSVs
        mode: 'train' or 'test'
        country: country string to filter
    
    Returns:
        (s1_lookup, cand_lookup) dicts
    """
    prefix = 'train' if mode == 'train' else 'test'
    
    s1_path = norm_dir / f"{prefix}_source1_norm.tsv"
    s2_path = norm_dir / f"{prefix}_source2_norm.tsv"
    s3_path = norm_dir / f"{prefix}_source3_norm.tsv"
    
    def load_filtered(filepath, country_filter):
        """Load entities filtered by country."""
        lookup = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split('\t')
            col_map = {col: i for i, col in enumerate(header)}
            eid_i = col_map['entity_id']
            country_i = col_map['country']
            norm_name_i = col_map.get('norm_name')
            core_name_i = col_map.get('core_name')
            norm_addr_i = col_map.get('norm_address')
            
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) <= max(eid_i, country_i):
                    continue
                if parts[country_i].strip() != country_filter:
                    continue
                eid = parts[eid_i]
                lookup[eid] = {
                    'norm_name': parts[norm_name_i] if (norm_name_i is not None and norm_name_i < len(parts)) else '',
                    'core_name': parts[core_name_i] if (core_name_i is not None and core_name_i < len(parts)) else '',
                    'norm_address': parts[norm_addr_i] if (norm_addr_i is not None and norm_addr_i < len(parts)) else '',
                }
        return lookup
    
    print(f"    Loading S1 entities for {country}...")
    s1_lookup = load_filtered(s1_path, country)
    print(f"    Loaded {len(s1_lookup):,} S1 entities")
    
    print(f"    Loading S2+S3 candidates for {country}...")
    cand_lookup = {}
    if s2_path.exists():
        cand_lookup.update(load_filtered(s2_path, country))
    if s3_path.exists():
        cand_lookup.update(load_filtered(s3_path, country))
    print(f"    Loaded {len(cand_lookup):,} candidate entities")
    
    return s1_lookup, cand_lookup


# ═══════════════════════════════════════════════════════════════════
# BATCH FEATURE COMPUTATION
# ═══════════════════════════════════════════════════════════════════

def compute_features_for_pairs(pairs_path, norm_dir, output_path, mode='train'):
    """Compute traditional features for all candidate pairs.
    
    Processes country-by-country to manage memory.
    
    Args:
        pairs_path: path to candidate_pairs.tsv from blocking
        norm_dir: directory with normalized TSVs
        output_path: path to write features TSV
        mode: 'train' or 'test'
    """
    print(f"\n{'='*60}")
    print(f"  FEATURE ENGINEERING (Traditional)")
    print(f"  Input:  {pairs_path}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")
    
    if not pairs_path.exists():
        print(f"  ERROR: {pairs_path} not found. Run blocking first.")
        return
    
    t0 = time.time()
    
    # Load all candidate pairs grouped by country
    print(f"  Loading candidate pairs...")
    pairs_by_country = defaultdict(list)
    with open(pairs_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        col_map = {col: i for i, col in enumerate(header)}
        
        for line in f:
            parts = line.rstrip('\n').split('\t')
            country = parts[col_map['country']].strip()
            pairs_by_country[country].append({
                's1_entity_id': parts[col_map['s1_entity_id']],
                'cand_entity_id': parts[col_map['cand_entity_id']],
                'cand_source': parts[col_map['cand_source']],
                'name_overlap': int(parts[col_map['name_overlap']]),
                'addr_overlap': int(parts[col_map['addr_overlap']]),
                'postal_match': int(parts[col_map['postal_match']]),
            })
    
    total_pairs = sum(len(v) for v in pairs_by_country.values())
    print(f"  Total pairs: {total_pairs:,} across {list(pairs_by_country.keys())}")
    
    # Process country by country
    os.makedirs(output_path.parent, exist_ok=True)
    first_write = True
    total_computed = 0
    
    # Define the traditional feature columns (without E5 — those come from embeddings.py)
    trad_feature_cols = [
        'name_jaccard', 'name_levenshtein', 'name_jaro_winkler',
        'name_token_sort', 'name_token_set',
        'core_name_jaccard', 'core_name_levenshtein', 'core_name_jaro_winkler',
        'name_len_ratio',
        'addr_jaccard', 'addr_levenshtein',
        'addr_token_overlap', 'addr_numeric_overlap', 'addr_missing',
        'name_overlap', 'addr_overlap', 'postal_match',
    ]
    
    for country, pairs in pairs_by_country.items():
        print(f"\n  [{country}] Computing features for {len(pairs):,} pairs...")
        t1 = time.time()
        
        # Load entity lookups for this country
        s1_lookup, cand_lookup = load_entity_lookups_for_country(
            norm_dir, mode, country
        )
        
        # Compute features for each pair
        rows = []
        for i, pair in enumerate(pairs):
            s1_id = pair['s1_entity_id']
            cand_id = pair['cand_entity_id']
            
            s1_data = s1_lookup.get(s1_id, {})
            cand_data = cand_lookup.get(cand_id, {})
            
            feats = compute_pair_features_row(
                s1_name=s1_data.get('norm_name', ''),
                s1_core=s1_data.get('core_name', ''),
                s1_addr=s1_data.get('norm_address', ''),
                cand_name=cand_data.get('norm_name', ''),
                cand_core=cand_data.get('core_name', ''),
                cand_addr=cand_data.get('norm_address', ''),
                name_overlap=pair['name_overlap'],
                addr_overlap=pair['addr_overlap'],
                postal_match=pair['postal_match'],
            )
            
            # Add metadata columns
            feats['s1_entity_id'] = s1_id
            feats['cand_entity_id'] = cand_id
            feats['cand_source'] = pair['cand_source']
            feats['country'] = country
            rows.append(feats)
            
            if (i + 1) % 500_000 == 0:
                print(f"    [{country}] Computed {i+1:,} / {len(pairs):,} pairs...")
        
        # Write to output
        df = pd.DataFrame(rows)
        meta_cols = ['s1_entity_id', 'cand_entity_id', 'cand_source', 'country']
        col_order = meta_cols + trad_feature_cols
        df = df[col_order]
        
        df.to_csv(
            output_path, sep='\t', index=False,
            mode='w' if first_write else 'a',
            header=first_write,
            encoding='utf-8',
        )
        first_write = False
        total_computed += len(rows)
        
        elapsed = time.time() - t1
        print(f"    [{country}] Done: {len(rows):,} pairs in {elapsed:.1f}s "
              f"({len(rows)/max(elapsed,0.1):.0f} pairs/sec)")
        
        # Free memory
        del s1_lookup, cand_lookup, rows, df
    
    total_elapsed = time.time() - t0
    print(f"\n  FEATURE ENGINEERING COMPLETE")
    print(f"    Total pairs:  {total_computed:,}")
    print(f"    Total time:   {total_elapsed:.1f}s")
    print(f"    Output:       {output_path}")


def run_features(mode='train'):
    """Run traditional feature engineering."""
    norm_dir = config.DATA_ROOT / "normalized"
    block_dir = config.DATA_ROOT / "blocked"
    feat_dir = config.DATA_ROOT / "features"
    os.makedirs(feat_dir, exist_ok=True)
    
    if mode == 'train':
        pairs_path = block_dir / "train_candidate_pairs.tsv"
        output_path = feat_dir / "train_features.tsv"
    else:
        pairs_path = block_dir / "test_candidate_pairs.tsv"
        output_path = feat_dir / "test_features.tsv"
    
    compute_features_for_pairs(pairs_path, norm_dir, output_path, mode=mode)


if __name__ == "__main__":
    run_features(mode='train')
