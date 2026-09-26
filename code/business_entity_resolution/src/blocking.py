"""Stage 4-5: Blocking — candidate pair generation and recall check.

Generates candidate pairs using 3 channels (unioned per S1 entity):
  1. Name-token inverted index  — shared word tokens in norm_name
  2. Address-token inverted index — shared word tokens in norm_address
  3. Postal-code exact match — same extracted postal/ZIP/PIN code

All channels are scoped by country. Common tokens are skipped
(via stopwords + max_postings cap) to avoid quadratic blowup.

The blocking recall checkpoint measures what fraction of ground-truth
true pairs survived blocking — this is the gate that decides whether
E5 retrieval is needed as a 4th channel.

Usage:
    python -m src.blocking
"""
import os
import re
import time
from collections import defaultdict, Counter
from pathlib import Path

from . import config


# ═══════════════════════════════════════════════════════════════════
# TOKENIZATION
# ═══════════════════════════════════════════════════════════════════

def discover_countries(s1_filepath):
    """Discover all countries present in the S1 dataset dynamically."""
    countries = set()
    with open(s1_filepath, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        country_i = {c: i for i, c in enumerate(header)}['country']
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if country_i < len(parts):
                countries.add(parts[country_i].strip())
    return sorted(countries)


def tokenize(text, min_len=None):
    """Split text into lowercase tokens, filtering stopwords and short tokens.
    
    Args:
        text: input string (should already be normalized)
        min_len: minimum token length (default: config.BLOCKING_MIN_TOKEN_LEN)
    
    Returns:
        set of unique tokens
    """
    if min_len is None:
        min_len = config.BLOCKING_MIN_TOKEN_LEN
    
    if not isinstance(text, str) or not text.strip():
        return set()
    
    # Split on whitespace and punctuation
    raw_tokens = re.split(r'[\s,/\-\.]+', text.lower())
    
    tokens = set()
    for t in raw_tokens:
        t = t.strip("'\"();:")
        if len(t) < min_len:
            continue
        if t in config.BLOCKING_STOPWORDS:
            continue
        tokens.add(t)
    
    return tokens


# ═══════════════════════════════════════════════════════════════════
# INVERTED INDEX
# ═══════════════════════════════════════════════════════════════════

class InvertedIndex:
    """Memory-efficient inverted index: token -> set of entity_ids.
    
    Supports max_postings cap to skip over-common tokens during lookup.
    """
    
    def __init__(self, max_postings=None):
        self.index = defaultdict(set)
        self.max_postings = max_postings or config.BLOCKING_MAX_POSTINGS
        self._capped_tokens = set()  # tokens that exceeded max_postings
    
    def add(self, entity_id, tokens):
        """Add an entity's tokens to the index."""
        for token in tokens:
            self.index[token].add(entity_id)
            
    def apply_cap(self):
        """Apply max_postings globally, deleting overly common tokens."""
        to_delete = []
        for token, posting in self.index.items():
            if len(posting) > self.max_postings:
                to_delete.append(token)
                self._capped_tokens.add(token)
        for token in to_delete:
            del self.index[token]
    
    def lookup(self, tokens, restrict_to_eids=None):
        """Find all entity_ids sharing any token, with overlap counts.
        
        Args:
            tokens: set of tokens to lookup
            restrict_to_eids: if provided, only these entity_ids are counted.
            
        Returns:
            Counter mapping entity_id -> number of shared tokens
        """
        hits = Counter()
        for token in tokens:
            if token in self._capped_tokens:
                continue
            if token in self.index:
                posting = self.index[token]
                if restrict_to_eids is not None:
                    # Fast C-level set intersection avoids massive Python loops
                    valid_eids = posting.intersection(restrict_to_eids)
                    for eid in valid_eids:
                        hits[eid] += 1
                else:
                    for eid in posting:
                        hits[eid] += 1
        return hits
    
    def stats(self):
        """Return index statistics."""
        posting_sizes = [len(v) for v in self.index.values()]
        return {
            'unique_tokens': len(self.index),
            'capped_tokens': len(self._capped_tokens),
            'total_postings': sum(posting_sizes),
            'avg_posting_size': sum(posting_sizes) / max(len(posting_sizes), 1),
            'max_posting_size': max(posting_sizes) if posting_sizes else 0,
        }


# ═══════════════════════════════════════════════════════════════════
# INDEX BUILDING
# ═══════════════════════════════════════════════════════════════════

def build_indexes_from_file(filepath, country_filter=None):
    """Build name, address, and postal inverted indexes from a normalized TSV.
    
    Reads line-by-line for memory efficiency.
    
    Args:
        filepath: path to normalized TSV
        country_filter: if set, only include records from this country
    
    Returns:
        (name_index, addr_index, postal_index, entity_count)
    """
    name_idx = InvertedIndex(max_postings=float('inf'))
    addr_idx = InvertedIndex(max_postings=float('inf'))
    postal_idx = InvertedIndex(max_postings=float('inf'))  # postal codes are less granular
    entity_count = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        col_map = {col: i for i, col in enumerate(header)}
        
        eid_i = col_map['entity_id']
        country_i = col_map['country']
        norm_name_i = col_map.get('norm_name')
        norm_addr_i = col_map.get('norm_address')
        postal_i = col_map.get('postal_code')
        
        if norm_name_i is None:
            raise ValueError(f"Column 'norm_name' not found in {filepath}. Run normalization first.")
        
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) <= max(eid_i, country_i):
                continue
            
            # Country filter
            if country_filter and parts[country_i].strip() != country_filter:
                continue
            
            eid = parts[eid_i].strip()
            entity_count += 1
            
            # Name tokens
            if norm_name_i is not None and norm_name_i < len(parts):
                name_tokens = tokenize(parts[norm_name_i])
                name_idx.add(eid, name_tokens)
            
            # Address tokens
            if norm_addr_i is not None and norm_addr_i < len(parts):
                addr_tokens = tokenize(parts[norm_addr_i])
                addr_idx.add(eid, addr_tokens)
            
            # Postal code (exact match channel)
            if postal_i is not None and postal_i < len(parts):
                pc = parts[postal_i].strip()
                if pc:
                    postal_idx.add(eid, {pc})
    
    return name_idx, addr_idx, postal_idx, entity_count


def derive_source(entity_id):
    """Derive source (s2/s3) from entity_id prefix."""
    if entity_id.startswith('S2-') or entity_id.startswith('s2-'):
        return 's2'
    elif entity_id.startswith('S3-') or entity_id.startswith('s3-'):
        return 's3'
    return 'unknown'


# ═══════════════════════════════════════════════════════════════════
# CANDIDATE GENERATION
# ═══════════════════════════════════════════════════════════════════

def generate_candidates_for_country(
    s1_filepath, cand_filepaths, country, output_file,
    max_candidates=None, append=False, gt_map=None
):
    """Generate candidate pairs for one country partition.
    
    Args:
        s1_filepath: path to normalized S1 TSV
        cand_filepaths: list of paths to normalized S2/S3 TSVs
        country: country string to filter on
        output_file: path to write candidate pairs TSV
        max_candidates: max candidates per S1 entity
        append: if True, append to output_file
    
    Returns:
        dict with statistics
    """
    if max_candidates is None:
        max_candidates = config.BLOCKING_MAX_CANDIDATES
    
    print(f"\n  [{country}] Building candidate inverted indexes...")
    t0 = time.time()
    
    # Build indexes from all candidate files (S2 + S3)
    # name_idx and addr_idx use the configured cap (5000) so apply_cap() actually
    # removes over-common tokens after the merge. We build per-file with inf to
    # avoid premature per-file caps that can lose candidates, then apply the
    # real global cap once after all sources are merged.
    # postal_idx starts as inf and gets its dedicated 50k cap set before apply_cap().
    name_idx = InvertedIndex(max_postings=config.BLOCKING_MAX_POSTINGS)
    addr_idx = InvertedIndex(max_postings=config.BLOCKING_MAX_POSTINGS)
    postal_idx = InvertedIndex(max_postings=float('inf'))  # cap set to 50k before apply_cap()
    total_cands = 0
    
    for cpath in cand_filepaths:
        if not cpath.exists():
            print(f"  WARNING: {cpath} not found, skipping.")
            continue
        n_idx, a_idx, p_idx, n_ents = build_indexes_from_file(cpath, country_filter=country)
        
        # Merge into main indexes
        for token, posting in n_idx.index.items():
            name_idx.index[token].update(posting)
        for token, posting in a_idx.index.items():
            addr_idx.index[token].update(posting)
        for token, posting in p_idx.index.items():
            postal_idx.index[token].update(posting)
        
        total_cands += n_ents
        print(f"    Loaded {n_ents:,} candidates from {cpath.name}")
    
    # CRITICAL: Apply max_postings cap globally AFTER merging all candidates.
    # Postal codes use a higher cap than name/address tokens because a postal
    # code appearing in 50k candidates is still structurally discriminative,
    # while a word like 'road' in 50k candidates is noise.
    name_idx.apply_cap()  # uses config.BLOCKING_MAX_POSTINGS (default 5000)
    addr_idx.apply_cap()  # uses config.BLOCKING_MAX_POSTINGS (default 5000)
    postal_idx.max_postings = 50_000  # override before capping
    postal_idx.apply_cap()  # higher cap for postal codes
            
    t_index = time.time() - t0
    
    name_stats = name_idx.stats()
    addr_stats = addr_idx.stats()
    print(f"  [{country}] Index built in {t_index:.1f}s")
    print(f"    Name index:  {name_stats['unique_tokens']:,} tokens, "
          f"{name_stats['total_postings']:,} postings, "
          f"{name_stats['capped_tokens']:,} capped")
    print(f"    Addr index:  {addr_stats['unique_tokens']:,} tokens, "
          f"{addr_stats['total_postings']:,} postings")
    print(f"    Total candidates: {total_cands:,}")
    
    # Now scan S1 and generate pairs
    print(f"  [{country}] Generating candidate pairs...")
    t1 = time.time()
    
    mode = 'a' if append else 'w'
    s1_count = 0
    total_pairs = 0
    candidates_per_s1 = []
    pre_topk_found = 0
    post_topk_found = 0
    gt_pairs_in_country = 0
    
    with open(s1_filepath, 'r', encoding='utf-8') as fin, \
         open(output_file, mode, encoding='utf-8') as fout:
        
        header = fin.readline().strip().split('\t')
        col_map = {col: i for i, col in enumerate(header)}
        eid_i = col_map['entity_id']
        country_i = col_map['country']
        norm_name_i = col_map.get('norm_name')
        norm_addr_i = col_map.get('norm_address')
        postal_i = col_map.get('postal_code')
        
        # Write output header if not appending
        if not append:
            fout.write("s1_entity_id\tcand_entity_id\tcand_source\tcountry\t"
                       "name_overlap\taddr_overlap\tpostal_match\n")
        
        for line in fin:
            parts = line.rstrip('\n').split('\t')
            if len(parts) <= max(eid_i, country_i):
                continue
            if parts[country_i].strip() != country:
                continue
            
            s1_id = parts[eid_i].strip()
            s1_count += 1
            
            true_cids = gt_map.get(s1_id, set()) if gt_map else set()
            if true_cids:
                gt_pairs_in_country += len(true_cids)
            
            # Channel 1: Name token overlap (blocking channel)
            s1_name_tokens = set()
            if norm_name_i is not None and norm_name_i < len(parts):
                s1_name_tokens = tokenize(parts[norm_name_i])
            name_hits = name_idx.lookup(s1_name_tokens)
            
            # Channel 2: Postal code exact match (blocking channel)
            s1_postal = ''
            if postal_i is not None and postal_i < len(parts):
                s1_postal = parts[postal_i].strip()
            postal_hits = set()
            if s1_postal and s1_postal in postal_idx.index:
                postal_hits = postal_idx.index[s1_postal]
            
            # Gating: candidate must enter via NAME or POSTAL.
            # Address is not an independent blocking channel here — it is
            # computed as a FEATURE only for the candidates already retrieved.
            # Rationale: the India address index has 30M postings; a full
            # unrestricted scan for every S1 entity causes catastrophic slowdown.
            # Pure address-only true pairs (name changed, postal missing) are
            # very rare; the ground-truth recall diagnostics below will flag
            # if this gating is causing significant recall loss.
            base_cands = set(name_hits.keys()) | postal_hits
            if not base_cands:
                candidates_per_s1.append(0)
                continue
            
            # Channel 3: Address overlap — scoring feature, restricted to base_cands.
            # Uses C-level set.intersection(), avoiding Python-level loops
            # over the full 30M-posting address index.
            s1_addr_tokens = set()
            if norm_addr_i is not None and norm_addr_i < len(parts):
                s1_addr_tokens = tokenize(parts[norm_addr_i])
            addr_hits = addr_idx.lookup(s1_addr_tokens, restrict_to_eids=base_cands)
            
            # Score each candidate (all must have name or postal hit)
            scored = []
            for cid in base_cands:
                n_ov = name_hits.get(cid, 0)
                a_ov = addr_hits.get(cid, 0)
                p_match = 1 if cid in postal_hits else 0
                # Note: no extra filter needed here — every candidate in base_cands
                # already has n_ov >= 1 (from name_hits) OR p_match == 1 (from postal_hits)
                # by construction. A separate min-overlap filter would be dead code.
                score = n_ov * 3 + a_ov * 2 + p_match * 5
                scored.append((cid, n_ov, a_ov, p_match, score))
                
            if true_cids:
                for cid in true_cids:
                    if any(c == cid for c, _, _, _, _ in scored):
                        pre_topk_found += 1
            
            # Sort by score descending, take top-K
            scored.sort(key=lambda x: -x[4])
            top = scored[:max_candidates]
            
            if true_cids:
                for cid in true_cids:
                    if any(c == cid for c, _, _, _, _ in top):
                        post_topk_found += 1
            
            candidates_per_s1.append(len(top))
            
            # Write pairs
            for cid, n_ov, a_ov, p_match, _ in top:
                src = derive_source(cid)
                fout.write(f"{s1_id}\t{cid}\t{src}\t{country}\t"
                           f"{n_ov}\t{a_ov}\t{p_match}\n")
                total_pairs += 1
            
            if s1_count % 100_000 == 0:
                print(f"    Processed {s1_count:,} S1 entities, "
                      f"{total_pairs:,} pairs so far...")
    
    t_gen = time.time() - t1
    
    # Stats
    if candidates_per_s1:
        candidates_per_s1.sort()
        median_cands = candidates_per_s1[len(candidates_per_s1) // 2]
        avg_cands = sum(candidates_per_s1) / len(candidates_per_s1)
        zero_cands = sum(1 for x in candidates_per_s1 if x == 0)
    else:
        median_cands = avg_cands = zero_cands = 0
    
    stats = {
        'country': country,
        's1_count': s1_count,
        'cand_count': total_cands,
        'total_pairs': total_pairs,
        'avg_cands_per_s1': avg_cands,
        'median_cands_per_s1': median_cands,
        'zero_candidate_s1': zero_cands,
        'index_time': t_index,
        'gen_time': t_gen,
        'pre_topk_found': pre_topk_found,
        'post_topk_found': post_topk_found,
        'gt_pairs': gt_pairs_in_country,
    }
    
    print(f"  [{country}] Done in {t_gen:.1f}s")
    print(f"    S1 entities:        {s1_count:,}")
    print(f"    Total pairs:        {total_pairs:,}")
    print(f"    Avg cands/S1:       {avg_cands:.1f}")
    print(f"    Median cands/S1:    {median_cands}")
    print(f"    S1 with 0 cands:    {zero_cands:,} ({zero_cands/max(s1_count,1)*100:.1f}%)")
    
    return stats


# ═══════════════════════════════════════════════════════════════════
# BLOCKING RECALL CHECK
# ═══════════════════════════════════════════════════════════════════

def load_ground_truth(gt_path):
    """Load ground truth as a set of (s1_id, cand_id) pairs.
    
    Returns:
        gt_pairs: set of (s1_id, cand_id) tuples
        gt_map: dict s1_id -> set of cand_ids
    """
    gt_pairs = set()
    gt_map = defaultdict(set)
    
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
                        gt_map[s1_id].add(cid)
    
    return gt_pairs, gt_map


def measure_blocking_recall(candidate_pairs_path, gt_path):
    """Measure what fraction of ground-truth pairs survived blocking.
    
    This is THE critical gate:
      - recall >= 95%: proceed with token blocking, use E5 as feature only
      - recall < 95%:  add E5 retrieval as 4th blocking channel
    
    Args:
        candidate_pairs_path: path to blocking output TSV
        gt_path: path to ground truth TSV
    
    Returns:
        dict with recall statistics
    """
    print(f"\n{'='*60}")
    print(f"  BLOCKING RECALL CHECK")
    print(f"{'='*60}")
    
    t0 = time.time()
    
    # Load ground truth
    print(f"  Loading ground truth from {gt_path.name}...")
    gt_pairs, gt_map = load_ground_truth(gt_path)
    print(f"  Ground truth: {len(gt_map):,} S1 entities, {len(gt_pairs):,} true pairs")
    
    # Stream candidate pairs to avoid OOM
    print(f"  Streaming candidate pairs from {candidate_pairs_path.name}...")
    found = set()
    total_blocked = 0
    with open(candidate_pairs_path, 'r', encoding='utf-8') as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                total_blocked += 1
                pair = (parts[0].strip(), parts[1].strip())
                if pair in gt_pairs:
                    found.add(pair)
    
    print(f"  Blocked pairs (streamed): {total_blocked:,}")
    
    # Compute recall
    missed = gt_pairs - found
    recall = len(found) / len(gt_pairs) if gt_pairs else 0.0
    
    elapsed = time.time() - t0
    
    print(f"\n  RESULTS (computed in {elapsed:.1f}s):")
    print(f"    True pairs:     {len(gt_pairs):,}")
    print(f"    Found by block: {len(found):,}")
    print(f"    Missed:         {len(missed):,}")
    print(f"    BLOCKING RECALL: {recall:.4f} ({recall*100:.2f}%)")
    
    if recall >= 0.95:
        print(f"\n    >> PASS: Recall >= 95%. Token blocking is sufficient.")
        print(f"    >> E5 can be used as a downstream FEATURE only.")
    else:
        print(f"\n    >> WARNING: Recall < 95%. Consider adding E5 retrieval")
        print(f"    >> as a 4th blocking channel to recover missed pairs.")
    
    # Show sample of missed pairs
    if missed:
        print(f"\n  Sample missed pairs (first 10):")
        for s1_id, cid in list(missed)[:10]:
            print(f"    {s1_id} -- {cid}")
    
    return {
        'total_gt_pairs': len(gt_pairs),
        'found': len(found),
        'missed': len(missed),
        'recall': recall,
    }


# ═══════════════════════════════════════════════════════════════════
# PIPELINE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def run_blocking(mode='train'):
    """Run blocking on train or test data.
    
    Args:
        mode: 'train' to block train data (with recall check),
              'test' to block test data (no recall check)
    """
    print("=" * 60)
    print(f"  BLOCKING (mode={mode})")
    print("=" * 60)
    
    norm_dir = config.DATA_ROOT / "normalized"
    block_dir = config.DATA_ROOT / "blocked"
    os.makedirs(block_dir, exist_ok=True)
    
    if mode == 'train':
        s1_path = norm_dir / "train_source1_norm.tsv"
        cand_paths = [
            norm_dir / "train_source2_norm.tsv",
            norm_dir / "train_source3_norm.tsv",
        ]
        output_path = block_dir / "train_candidate_pairs.tsv"
        countries = discover_countries(s1_path)
    else:
        s1_path = norm_dir / "test_source1_norm.tsv"
        cand_paths = [
            norm_dir / "test_source2_norm.tsv",
            norm_dir / "test_source3_norm.tsv",
        ]
        output_path = block_dir / "test_candidate_pairs.tsv"
        countries = discover_countries(s1_path)
    
    if not s1_path.exists():
        print(f"  ERROR: {s1_path} not found. Run normalization first.")
        return
    
    gt_map = None
    if mode == 'train' and config.TRAIN_GT.exists():
        _, gt_map = load_ground_truth(config.TRAIN_GT)
        
    all_stats = []
    for i, country in enumerate(countries):
        stats = generate_candidates_for_country(
            s1_filepath=s1_path,
            cand_filepaths=cand_paths,
            country=country,
            output_file=output_path,
            append=(i > 0),  # first country writes header, rest append
            gt_map=gt_map,
        )
        all_stats.append(stats)
    
    # Summary
    total_pairs = sum(s['total_pairs'] for s in all_stats)
    total_s1 = sum(s['s1_count'] for s in all_stats)
    print(f"\n  BLOCKING SUMMARY:")
    print(f"    Total S1 entities: {total_s1:,}")
    print(f"    Total candidate pairs: {total_pairs:,}")
    print(f"    Output: {output_path}")
    
    if mode == 'train' and gt_map:
        total_gt = sum(s['gt_pairs'] for s in all_stats)
        pre_found = sum(s['pre_topk_found'] for s in all_stats)
        post_found = sum(s['post_topk_found'] for s in all_stats)
        
        pre_recall = pre_found / max(total_gt, 1)
        post_recall = post_found / max(total_gt, 1)
        
        print("\n  RECALL DIAGNOSTICS:")
        print(f"    Total true pairs:    {total_gt:,}")
        print(f"    Recall BEFORE TOP-K  (after name/postal gating): {pre_found:,} ({pre_recall*100:.2f}%)")
        print(f"    Recall AFTER TOP-K   (final output recall):       {post_found:,} ({post_recall*100:.2f}%)")
        print(f"    Note: 'before top-K' excludes address-only true pairs (name+postal gating applied)")
    
    # Blocking recall check (train mode only)
    if mode == 'train' and config.TRAIN_GT.exists():
        recall_stats = measure_blocking_recall(output_path, config.TRAIN_GT)
    
    print("\n" + "=" * 60)
    print("  BLOCKING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_blocking(mode='train')
