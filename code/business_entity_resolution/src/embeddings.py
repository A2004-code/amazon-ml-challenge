"""Stage 6: E5 Embeddings — multilingual-e5-base cosine similarity.

Computes semantic similarity features using the multilingual-e5-base model.
This is the ONLY feature capable of matching across scripts
(e.g., "Ram Marketing" vs "राम मार्केटिंग").

Workflow:
  1. Collect unique entity IDs from candidate pairs
  2. Load their normalized name/address from source files
  3. Encode with E5 (GPU), cache embeddings to .npy
  4. Compute cosine similarity for each pair
  5. Merge with existing feature TSV

Input:  candidate_pairs.tsv + normalized source files
Output: 2 additional features appended to features TSV
        (e5_name_cosine, e5_addr_cosine)

Usage:
    python -m src.embeddings
"""
import os
import time
from pathlib import Path

import numpy as np

from . import config


def collect_entity_ids(pairs_path):
    """Collect unique S1 and candidate entity IDs from candidate pairs.
    
    Returns:
        (set of S1 IDs, set of candidate IDs)
    """
    s1_ids = set()
    cand_ids = set()
    
    with open(pairs_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        col_map = {col: i for i, col in enumerate(header)}
        s1_i = col_map['s1_entity_id']
        cand_i = col_map['cand_entity_id']
        
        for line in f:
            parts = line.rstrip('\n').split('\t')
            s1_ids.add(parts[s1_i])
            cand_ids.add(parts[cand_i])
    
    return s1_ids, cand_ids


def load_texts_for_ids(filepath, target_ids):
    """Load norm_name and norm_address for a set of entity IDs.
    
    Returns:
        dict: entity_id -> {'norm_name': str, 'norm_address': str}
    """
    texts = {}
    target = set(target_ids)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        col_map = {col: i for i, col in enumerate(header)}
        eid_i = col_map['entity_id']
        name_i = col_map.get('norm_name')
        addr_i = col_map.get('norm_address')
        
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) <= eid_i:
                continue
            eid = parts[eid_i]
            if eid not in target:
                continue
            texts[eid] = {
                'norm_name': parts[name_i] if (name_i is not None and name_i < len(parts)) else '',
                'norm_address': parts[addr_i] if (addr_i is not None and addr_i < len(parts)) else '',
            }
            # Early exit if all found
            if len(texts) == len(target):
                break
    
    return texts


def encode_texts(model, texts, prefix, batch_size=None):
    """Encode a list of texts with E5 model.
    
    Args:
        model: SentenceTransformer model
        texts: list of strings to encode
        prefix: "query: " for S1 entities, "passage: " for candidates
        batch_size: encoding batch size
    
    Returns:
        np.ndarray of shape (len(texts), embedding_dim), L2-normalized
    """
    if batch_size is None:
        batch_size = config.E5_BATCH_SIZE
    
    # E5 requires prefix
    prefixed = [f"{prefix}{t}" for t in texts]
    
    embeddings = model.encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=True,  # L2 normalize for cosine = dot product
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    
    return embeddings


def compute_e5_embeddings(pairs_path, norm_dir, mode='train'):
    """Compute E5 embeddings for all entities in candidate pairs.
    
    Saves embeddings to cache (.npy). Returns only lightweight ID maps
    and cache paths — the actual arrays are loaded separately by the
    two-pass merge to stay under 25 GB RAM.
    
    Args:
        pairs_path: path to candidate_pairs.tsv
        norm_dir: directory with normalized TSVs
        mode: 'train' or 'test'
    
    Returns:
        (s1_id_to_idx, cand_id_to_idx, cache_paths)
        cache_paths is a dict with keys: s1_name, s1_addr, cand_name, cand_addr
    """
    # Lazy import — only needed on GPU machine
    from sentence_transformers import SentenceTransformer
    
    cache_dir = config.E5_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    
    prefix = 'train' if mode == 'train' else 'test'
    s1_name_cache = cache_dir / f"{prefix}_s1_name_emb.npy"
    s1_addr_cache = cache_dir / f"{prefix}_s1_addr_emb.npy"
    s1_ids_cache = cache_dir / f"{prefix}_s1_ids.npy"
    cand_name_cache = cache_dir / f"{prefix}_cand_name_emb.npy"
    cand_addr_cache = cache_dir / f"{prefix}_cand_addr_emb.npy"
    cand_ids_cache = cache_dir / f"{prefix}_cand_ids.npy"
    
    cache_paths = {
        's1_name': s1_name_cache, 's1_addr': s1_addr_cache,
        'cand_name': cand_name_cache, 'cand_addr': cand_addr_cache,
    }
    
    # Collect entity IDs from candidate pairs
    print(f"  Collecting entity IDs from {pairs_path.name}...")
    s1_ids, cand_ids = collect_entity_ids(pairs_path)
    print(f"  Unique S1 entities: {len(s1_ids):,}")
    print(f"  Unique candidate entities: {len(cand_ids):,}")
    
    # Check if cache exists and is valid
    cache_valid = all(p.exists() for p in [
        s1_name_cache, s1_addr_cache, s1_ids_cache,
        cand_name_cache, cand_addr_cache, cand_ids_cache
    ])
    
    if cache_valid:
        print(f"  Loading embeddings from cache...")
        s1_id_arr = np.load(s1_ids_cache, allow_pickle=True)
        cand_id_arr = np.load(cand_ids_cache, allow_pickle=True)
        
        # Verify cache covers all needed IDs
        cached_s1 = set(s1_id_arr)
        cached_cand = set(cand_id_arr)
        
        if s1_ids.issubset(cached_s1) and cand_ids.issubset(cached_cand):
            print(f"  Cache hit: {len(s1_id_arr):,} S1 + {len(cand_id_arr):,} candidates")
            s1_id_to_idx = {eid: i for i, eid in enumerate(s1_id_arr)}
            cand_id_to_idx = {eid: i for i, eid in enumerate(cand_id_arr)}
            return s1_id_to_idx, cand_id_to_idx, cache_paths
        else:
            print(f"  Cache incomplete, recomputing...")
    
    # Load texts for all needed entities
    print(f"  Loading entity texts from normalized files...")
    s1_texts = {}
    cand_texts = {}
    
    s1_file = norm_dir / f"{prefix}_source1_norm.tsv"
    s2_file = norm_dir / f"{prefix}_source2_norm.tsv"
    s3_file = norm_dir / f"{prefix}_source3_norm.tsv"
    
    if s1_file.exists():
        s1_texts.update(load_texts_for_ids(s1_file, s1_ids))
    if s2_file.exists():
        cand_texts.update(load_texts_for_ids(s2_file, cand_ids))
    if s3_file.exists():
        cand_texts.update(load_texts_for_ids(s3_file, cand_ids))
    
    print(f"  Loaded texts: {len(s1_texts):,} S1, {len(cand_texts):,} candidates")
    
    # Order entities for consistent indexing
    s1_id_list = sorted(s1_ids)
    cand_id_list = sorted(cand_ids)
    
    s1_names = [s1_texts.get(eid, {}).get('norm_name', '') for eid in s1_id_list]
    s1_addrs = [s1_texts.get(eid, {}).get('norm_address', '') for eid in s1_id_list]
    cand_names = [cand_texts.get(eid, {}).get('norm_name', '') for eid in cand_id_list]
    cand_addrs = [cand_texts.get(eid, {}).get('norm_address', '') for eid in cand_id_list]
    
    # Aggressive memory cleanup
    del s1_texts
    del cand_texts
    import gc
    gc.collect()
    
    # Save IDs to cache immediately
    np.save(s1_ids_cache, np.array(s1_id_list, dtype=object))
    np.save(cand_ids_cache, np.array(cand_id_list, dtype=object))
    
    # Load E5 model
    print(f"\n  Loading E5 model: {config.E5_MODEL_NAME}")
    t0 = time.time()
    model = SentenceTransformer(config.E5_MODEL_NAME)
    print(f"  Model loaded in {time.time()-t0:.1f}s")
    
    # Encode S1 names -> Save -> Clear RAM
    print(f"\n  Encoding {len(s1_names):,} S1 names...")
    t0 = time.time()
    emb = encode_texts(model, s1_names, prefix="query: ")
    np.save(s1_name_cache, emb)
    del s1_names, emb
    gc.collect()
    print(f"  Done in {time.time()-t0:.1f}s")
    
    # Encode S1 addresses -> Save -> Clear RAM
    print(f"  Encoding {len(s1_addrs):,} S1 addresses...")
    t0 = time.time()
    emb = encode_texts(model, s1_addrs, prefix="query: ")
    np.save(s1_addr_cache, emb)
    del s1_addrs, emb
    gc.collect()
    print(f"  Done in {time.time()-t0:.1f}s")
    
    # Encode candidate names -> Save -> Clear RAM
    print(f"\n  Encoding {len(cand_names):,} candidate names...")
    t0 = time.time()
    emb = encode_texts(model, cand_names, prefix="passage: ")
    np.save(cand_name_cache, emb)
    del cand_names, emb
    gc.collect()
    print(f"  Done in {time.time()-t0:.1f}s")
    
    # Encode candidate addresses -> Save -> Clear RAM
    print(f"  Encoding {len(cand_addrs):,} candidate addresses...")
    t0 = time.time()
    emb = encode_texts(model, cand_addrs, prefix="passage: ")
    np.save(cand_addr_cache, emb)
    del cand_addrs, emb
    gc.collect()
    print(f"  Done in {time.time()-t0:.1f}s")
    
    # Build lightweight ID to index mappings
    print(f"\n  Building index mappings...")
    s1_id_to_idx = {eid: i for i, eid in enumerate(s1_id_list)}
    cand_id_to_idx = {eid: i for i, eid in enumerate(cand_id_list)}
    
    print(f"  Cached to {cache_dir}")
    
    # Free GPU memory
    del model
    gc.collect()
    
    return s1_id_to_idx, cand_id_to_idx, cache_paths


def _cosine_pass(features_input, output_path, s1_id_to_idx, cand_id_to_idx,
                 s1_emb, cand_emb, col_name, first_pass=True):
    """Single-pass: read chunks, compute one cosine column, write output.
    
    Args:
        features_input: path to input TSV (features or intermediate)
        output_path: path to write output TSV
        s1_id_to_idx: dict mapping S1 entity_id -> row index
        cand_id_to_idx: dict mapping cand entity_id -> row index
        s1_emb: numpy array (n_s1, 768) — one embedding type (name or addr)
        cand_emb: numpy array (n_cand, 768) — matching type
        col_name: column name for the cosine score (e.g. 'e5_name_cosine')
        first_pass: if True, also add a placeholder for the other cosine column
    """
    import pandas as pd
    
    chunksize = 500_000
    reader = pd.read_csv(
        features_input, sep='\t', encoding='utf-8',
        chunksize=chunksize, dtype={'s1_entity_id': str, 'cand_entity_id': str},
    )
    
    first_write = True
    total_pairs = 0
    
    for chunk in reader:
        s1_indices = np.array([s1_id_to_idx.get(eid, -1) for eid in chunk['s1_entity_id']])
        cand_indices = np.array([cand_id_to_idx.get(eid, -1) for eid in chunk['cand_entity_id']])
        
        valid_s1 = s1_indices >= 0
        valid_cand = cand_indices >= 0
        
        s1_mat = np.zeros((len(chunk), config.E5_EMBEDDING_DIM), dtype=np.float32)
        cand_mat = np.zeros((len(chunk), config.E5_EMBEDDING_DIM), dtype=np.float32)
        
        if valid_s1.any():
            s1_mat[valid_s1] = s1_emb[s1_indices[valid_s1]]
        if valid_cand.any():
            cand_mat[valid_cand] = cand_emb[cand_indices[valid_cand]]
        
        chunk[col_name] = (s1_mat * cand_mat).sum(axis=1)
        
        # On first pass, add placeholder for the second cosine column
        if first_pass and 'e5_addr_cosine' not in chunk.columns:
            chunk['e5_addr_cosine'] = 0.0
        
        chunk.to_csv(
            output_path, sep='\t', index=False,
            mode='w' if first_write else 'a',
            header=first_write,
            encoding='utf-8',
        )
        first_write = False
        total_pairs += len(chunk)
        print(f"    Processed {total_pairs:,} pairs...")
    
    return total_pairs


def merge_e5_features(features_path, pairs_path, norm_dir, output_path, mode='train'):
    """Compute E5 cosine similarities and merge with traditional features.
    
    Uses a two-pass approach to stay under 25GB RAM:
      Pass 1: Load name embeddings (~22GB), compute e5_name_cosine
      Pass 2: Load addr embeddings (~22GB), compute e5_addr_cosine
    
    Args:
        features_path: path to traditional features TSV (from features.py)
        pairs_path: path to candidate_pairs.tsv (from blocking)
        norm_dir: directory with normalized TSVs
        output_path: path to write combined features TSV
        mode: 'train' or 'test'
    """
    import gc
    
    print(f"\n{'='*60}")
    print(f"  E5 EMBEDDING FEATURES")
    print(f"{'='*60}")
    
    if not features_path.exists():
        print(f"  ERROR: {features_path} not found. Run feature engineering first.")
        return
    
    t0 = time.time()
    
    # Compute embeddings (or verify cache) — returns lightweight maps only
    s1_id_to_idx, cand_id_to_idx, cache_paths = \
        compute_e5_embeddings(pairs_path, norm_dir, mode)
    
    # ── Pass 1: Name cosine (~22 GB) ─────────────────────────────
    print(f"\n  Pass 1/2: Loading name embeddings into RAM...")
    t1 = time.time()
    s1_name_emb = np.load(cache_paths['s1_name'])
    cand_name_emb = np.load(cache_paths['cand_name'])
    print(f"  Loaded in {time.time()-t1:.1f}s  "
          f"({s1_name_emb.nbytes/1e9:.1f} + {cand_name_emb.nbytes/1e9:.1f} = "
          f"{(s1_name_emb.nbytes + cand_name_emb.nbytes)/1e9:.1f} GB)")
    
    temp_path = output_path.with_suffix('.tmp')
    total_pairs = _cosine_pass(
        features_path, temp_path,
        s1_id_to_idx, cand_id_to_idx,
        s1_name_emb, cand_name_emb,
        col_name='e5_name_cosine', first_pass=True,
    )
    
    del s1_name_emb, cand_name_emb
    gc.collect()
    print(f"  Pass 1 done in {time.time()-t1:.1f}s")
    
    # ── Pass 2: Address cosine (~22 GB) ──────────────────────────
    print(f"\n  Pass 2/2: Loading address embeddings into RAM...")
    t2 = time.time()
    s1_addr_emb = np.load(cache_paths['s1_addr'])
    cand_addr_emb = np.load(cache_paths['cand_addr'])
    print(f"  Loaded in {time.time()-t2:.1f}s  "
          f"({s1_addr_emb.nbytes/1e9:.1f} + {cand_addr_emb.nbytes/1e9:.1f} = "
          f"{(s1_addr_emb.nbytes + cand_addr_emb.nbytes)/1e9:.1f} GB)")
    
    total_pairs = _cosine_pass(
        temp_path, output_path,
        s1_id_to_idx, cand_id_to_idx,
        s1_addr_emb, cand_addr_emb,
        col_name='e5_addr_cosine', first_pass=False,
    )
    
    del s1_addr_emb, cand_addr_emb
    gc.collect()
    
    # Clean up temp file
    if temp_path.exists():
        temp_path.unlink()
    
    print(f"  Pass 2 done in {time.time()-t2:.1f}s")
    
    total_elapsed = time.time() - t0
    print(f"\n  E5 FEATURES COMPLETE")
    print(f"    Total pairs: {total_pairs:,}")
    print(f"    Total time:  {total_elapsed:.1f}s")
    print(f"    Output:      {output_path}")


def run_embeddings(mode='train'):
    """Run E5 embedding feature computation."""
    norm_dir = config.DATA_ROOT / "normalized"
    block_dir = config.DATA_ROOT / "blocked"
    feat_dir = config.DATA_ROOT / "features"
    
    if mode == 'train':
        pairs_path = block_dir / "train_candidate_pairs.tsv"
        features_path = feat_dir / "train_features.tsv"
        output_path = feat_dir / "train_features_full.tsv"
    else:
        pairs_path = block_dir / "test_candidate_pairs.tsv"
        features_path = feat_dir / "test_features.tsv"
        output_path = feat_dir / "test_features_full.tsv"
    
    merge_e5_features(features_path, pairs_path, norm_dir, output_path, mode)


if __name__ == "__main__":
    run_embeddings(mode='train')
