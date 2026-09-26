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
    
    Saves embeddings to cache (.npy) and returns dense arrays + index maps.
    
    Args:
        pairs_path: path to candidate_pairs.tsv
        norm_dir: directory with normalized TSVs
        mode: 'train' or 'test'
    
    Returns:
        (s1_name_emb, s1_addr_emb, cand_name_emb, cand_addr_emb,
         s1_id_to_idx, cand_id_to_idx)
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
            # Load fully into RAM — mmap is too slow on network storage
            s1_name_emb = np.load(s1_name_cache)
            s1_addr_emb = np.load(s1_addr_cache)
            cand_name_emb = np.load(cand_name_cache)
            cand_addr_emb = np.load(cand_addr_cache)
            
            s1_id_to_idx = {eid: i for i, eid in enumerate(s1_id_arr)}
            cand_id_to_idx = {eid: i for i, eid in enumerate(cand_id_arr)}
            return s1_name_emb, s1_addr_emb, cand_name_emb, cand_addr_emb, s1_id_to_idx, cand_id_to_idx
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
    
    # Load back into RAM for fast merge (model + texts are freed, ~44GB fits in 50GB)
    s1_name_emb = np.load(s1_name_cache)
    s1_addr_emb = np.load(s1_addr_cache)
    cand_name_emb = np.load(cand_name_cache)
    cand_addr_emb = np.load(cand_addr_cache)
    
    return s1_name_emb, s1_addr_emb, cand_name_emb, cand_addr_emb, s1_id_to_idx, cand_id_to_idx


def merge_e5_features(features_path, pairs_path, norm_dir, output_path, mode='train'):
    """Compute E5 cosine similarities and merge with traditional features.
    
    Reads the traditional features TSV, adds e5_name_cosine and e5_addr_cosine
    columns, writes the combined output.
    
    Args:
        features_path: path to traditional features TSV (from features.py)
        pairs_path: path to candidate_pairs.tsv (from blocking)
        norm_dir: directory with normalized TSVs
        output_path: path to write combined features TSV
        mode: 'train' or 'test'
    """
    print(f"\n{'='*60}")
    print(f"  E5 EMBEDDING FEATURES")
    print(f"{'='*60}")
    
    if not features_path.exists():
        print(f"  ERROR: {features_path} not found. Run feature engineering first.")
        return
    
    t0 = time.time()
    
    # Compute embeddings (or load from cache)
    s1_name_emb, s1_addr_emb, cand_name_emb, cand_addr_emb, s1_id_to_idx, cand_id_to_idx = \
        compute_e5_embeddings(pairs_path, norm_dir, mode)
    
    # Read features TSV and add E5 columns
    print(f"\n  Merging E5 features with traditional features...")
    
    import pandas as pd
    
    # Process in chunks to manage memory
    chunksize = 500_000
    reader = pd.read_csv(
        features_path, sep='\t', encoding='utf-8',
        chunksize=chunksize, dtype={'s1_entity_id': str, 'cand_entity_id': str},
    )
    
    first_write = True
    total_pairs = 0
    zero_emb = np.zeros(config.E5_EMBEDDING_DIM, dtype=np.float32)
    
    for chunk in reader:
        # Convert indices to numpy arrays for fast bulk indexing
        s1_indices = np.array([s1_id_to_idx.get(eid, -1) for eid in chunk['s1_entity_id']])
        cand_indices = np.array([cand_id_to_idx.get(eid, -1) for eid in chunk['cand_entity_id']])
        
        valid_s1 = s1_indices >= 0
        valid_cand = cand_indices >= 0

        # Build embedding matrices via vectorized C-level memmap lookups
        s1_name_mat = np.zeros((len(chunk), config.E5_EMBEDDING_DIM), dtype=np.float32)
        if valid_s1.any():
            s1_name_mat[valid_s1] = s1_name_emb[s1_indices[valid_s1]]
            
        cand_name_mat = np.zeros((len(chunk), config.E5_EMBEDDING_DIM), dtype=np.float32)
        if valid_cand.any():
            cand_name_mat[valid_cand] = cand_name_emb[cand_indices[valid_cand]]
            
        s1_addr_mat = np.zeros((len(chunk), config.E5_EMBEDDING_DIM), dtype=np.float32)
        if valid_s1.any():
            s1_addr_mat[valid_s1] = s1_addr_emb[s1_indices[valid_s1]]
            
        cand_addr_mat = np.zeros((len(chunk), config.E5_EMBEDDING_DIM), dtype=np.float32)
        if valid_cand.any():
            cand_addr_mat[valid_cand] = cand_addr_emb[cand_indices[valid_cand]]

        # Element-wise multiply then sum across embedding dim — one numpy op per similarity
        chunk['e5_name_cosine'] = (s1_name_mat * cand_name_mat).sum(axis=1)
        chunk['e5_addr_cosine'] = (s1_addr_mat * cand_addr_mat).sum(axis=1)

        chunk.to_csv(
            output_path, sep='\t', index=False,
            mode='w' if first_write else 'a',
            header=first_write,
            encoding='utf-8',
        )
        first_write = False
        total_pairs += len(chunk)
        print(f"    Processed {total_pairs:,} pairs...")
    
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
