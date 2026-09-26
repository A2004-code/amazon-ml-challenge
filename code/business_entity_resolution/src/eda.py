"""Stage 1: Exploratory Data Analysis.

Prints a comprehensive summary of each source file:
- Shape, dtypes, columns
- Missing values per column
- Duplicate entity_id counts
- Country distributions
- Name/address length statistics
- Non-ASCII name analysis (script detection for Indian records)
- Ground truth match distribution

Usage:
    python -m src.eda
"""
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

from . import config


def load_tsv_lazy(filepath, max_rows=None):
    """Load a TSV file line-by-line without pandas. Returns header + rows."""
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            rows.append(line.strip().split('\t'))
    return header, rows


def count_lines(filepath):
    """Count total data rows (excluding header) in a file."""
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        f.readline()  # skip header
        for _ in f:
            count += 1
    return count


def detect_script(text):
    """Classify the dominant Unicode script block of a string."""
    blocks = Counter()
    for ch in text:
        cp = ord(ch)
        if cp <= 0x007F:
            blocks['ASCII'] += 1
        elif 0x0900 <= cp <= 0x097F:
            blocks['Devanagari'] += 1
        elif 0x0980 <= cp <= 0x09FF:
            blocks['Bengali'] += 1
        elif 0x0A00 <= cp <= 0x0A7F:
            blocks['Gurmukhi'] += 1
        elif 0x0A80 <= cp <= 0x0AFF:
            blocks['Gujarati'] += 1
        elif 0x0B00 <= cp <= 0x0B7F:
            blocks['Oriya'] += 1
        elif 0x0B80 <= cp <= 0x0BFF:
            blocks['Tamil'] += 1
        elif 0x0C00 <= cp <= 0x0C7F:
            blocks['Telugu'] += 1
        elif 0x0C80 <= cp <= 0x0CFF:
            blocks['Kannada'] += 1
        elif 0x0D00 <= cp <= 0x0D7F:
            blocks['Malayalam'] += 1
        elif 0x0080 <= cp <= 0x024F:
            blocks['Latin-Extended'] += 1
        else:
            blocks['Other'] += 1
    
    if not blocks:
        return 'Empty'
    return max(blocks, key=blocks.get)


def analyze_source(name, filepath, sample_size=500_000):
    """Run EDA on a single source TSV file."""
    print(f"\n{'='*70}")
    print(f"  {name}: {filepath.name}")
    print(f"{'='*70}")
    
    if not filepath.exists():
        print(f"  FILE NOT FOUND: {filepath}")
        return
    
    # Count total rows
    t0 = time.time()
    total_rows = count_lines(filepath)
    print(f"  Total rows: {total_rows:,}  (counted in {time.time()-t0:.1f}s)")
    
    # Load sample for detailed analysis
    scan_size = min(sample_size, total_rows)
    header, rows = load_tsv_lazy(filepath, max_rows=scan_size)
    print(f"  Columns: {header}")
    print(f"  Analyzing first {scan_size:,} rows...\n")
    
    # Column indices
    col_map = {col: i for i, col in enumerate(header)}
    eid_idx = col_map.get('entity_id', 0)
    name_idx = col_map.get('business_name', 1)
    addr_idx = col_map.get('business_address', 2)
    country_idx = col_map.get('country', 3)
    
    # --- Missing Values ---
    missing = {col: 0 for col in header}
    for row in rows:
        for i, col in enumerate(header):
            if i >= len(row) or not row[i].strip():
                missing[col] += 1
    
    print("  Missing Values:")
    for col in header:
        pct = missing[col] / scan_size * 100
        print(f"    {col:25s}: {missing[col]:>8,} ({pct:5.2f}%)")
    
    # --- Duplicate entity_ids ---
    eid_counter = Counter()
    for row in rows:
        if eid_idx < len(row):
            eid_counter[row[eid_idx]] += 1
    
    dupes = {k: v for k, v in eid_counter.items() if v > 1}
    print(f"\n  Duplicate entity_ids: {len(dupes):,} IDs appearing more than once")
    if dupes:
        top_dupes = sorted(dupes.items(), key=lambda x: -x[1])[:5]
        for eid, cnt in top_dupes:
            print(f"    {eid}: {cnt} times")
    
    # --- Country Distribution ---
    if country_idx < len(header):
        country_counter = Counter()
        for row in rows:
            if country_idx < len(row):
                country_counter[row[country_idx].strip()] += 1
        
        print(f"\n  Country Distribution:")
        for country, cnt in country_counter.most_common():
            pct = cnt / scan_size * 100
            print(f"    {country:15s}: {cnt:>10,} ({pct:5.1f}%)")
    
    # --- Name Length Stats ---
    name_lengths = []
    empty_names = 0
    for row in rows:
        if name_idx < len(row) and row[name_idx].strip():
            name_lengths.append(len(row[name_idx]))
        else:
            empty_names += 1
    
    if name_lengths:
        name_lengths.sort()
        print(f"\n  Business Name Length:")
        print(f"    Min: {name_lengths[0]}, Max: {name_lengths[-1]}")
        print(f"    Median: {name_lengths[len(name_lengths)//2]}")
        print(f"    Mean: {sum(name_lengths)/len(name_lengths):.1f}")
        print(f"    Empty/missing: {empty_names:,}")
    
    # --- Address Length Stats ---
    addr_lengths = []
    empty_addrs = 0
    for row in rows:
        if addr_idx < len(row) and row[addr_idx].strip():
            addr_lengths.append(len(row[addr_idx]))
        else:
            empty_addrs += 1
    
    if addr_lengths:
        addr_lengths.sort()
        print(f"\n  Business Address Length:")
        print(f"    Min: {addr_lengths[0]}, Max: {addr_lengths[-1]}")
        print(f"    Median: {addr_lengths[len(addr_lengths)//2]}")
        print(f"    Mean: {sum(addr_lengths)/len(addr_lengths):.1f}")
        print(f"    Empty/missing: {empty_addrs:,}")
    
    # --- Non-ASCII / Script Analysis (India only) ---
    if country_idx < len(header):
        india_total = 0
        non_ascii_count = 0
        script_counter = Counter()
        zwnj_count = 0
        
        for row in rows:
            if country_idx >= len(row):
                continue
            if row[country_idx].strip() != 'India':
                continue
            india_total += 1
            
            if name_idx >= len(row):
                continue
            bname = row[name_idx]
            
            has_non_ascii = any(ord(c) > 127 for c in bname)
            if has_non_ascii:
                non_ascii_count += 1
                script_counter[detect_script(bname)] += 1
            
            if '\u200c' in bname:
                zwnj_count += 1
        
        if india_total > 0:
            print(f"\n  India Non-ASCII Name Analysis ({india_total:,} India records):")
            print(f"    Names with non-ASCII: {non_ascii_count:,} ({non_ascii_count/india_total*100:.1f}%)")
            print(f"    Names with ZWNJ char: {zwnj_count:,}")
            print(f"    Script distribution:")
            for script, cnt in script_counter.most_common(10):
                print(f"      {script:15s}: {cnt:>8,}")
    
    # --- Sample Rows ---
    print(f"\n  Sample rows (first 3):")
    for row in rows[:3]:
        display = {header[i]: (row[i][:60] if i < len(row) else 'N/A') for i in range(len(header))}
        print(f"    {display}")


def analyze_ground_truth(filepath, sample_size=500_000):
    """Analyze ground truth match patterns."""
    print(f"\n{'='*70}")
    print(f"  Ground Truth: {filepath.name}")
    print(f"{'='*70}")
    
    if not filepath.exists():
        print(f"  FILE NOT FOUND: {filepath}")
        return
    
    total = 0
    with_matches = 0
    match_counts = Counter()  # number of matches per S1 entity
    
    t0 = time.time()
    with open(filepath, 'r', encoding='utf-8') as f:
        f.readline()  # skip header
        for line in f:
            total += 1
            parts = line.strip().split('\t')
            if len(parts) > 1 and parts[1].strip():
                matches = [m.strip() for m in parts[1].split(',') if m.strip()]
                n = len(matches)
                with_matches += 1
                match_counts[n] += 1
            else:
                match_counts[0] += 1
    
    elapsed = time.time() - t0
    singletons = total - with_matches
    
    print(f"  Total S1 entities: {total:,}  (scanned in {elapsed:.1f}s)")
    print(f"  With matches:      {with_matches:,} ({with_matches/total*100:.1f}%)")
    print(f"  Singletons:        {singletons:,} ({singletons/total*100:.1f}%)")
    
    print(f"\n  Match count distribution:")
    for n_matches in sorted(match_counts.keys()):
        cnt = match_counts[n_matches]
        pct = cnt / total * 100
        print(f"    {n_matches:3d} matches: {cnt:>10,} S1 entities ({pct:5.2f}%)")
    
    # Compute average matches per entity (including singletons)
    total_match_pairs = sum(k * v for k, v in match_counts.items())
    avg_all = total_match_pairs / total if total > 0 else 0
    avg_matched = total_match_pairs / with_matches if with_matches > 0 else 0
    print(f"\n  Total match pairs: {total_match_pairs:,}")
    print(f"  Avg matches/entity (all):      {avg_all:.2f}")
    print(f"  Avg matches/entity (non-zero): {avg_matched:.2f}")


def run_eda():
    """Run full EDA on all sources."""
    print("=" * 70)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    
    config.print_config()
    
    # Analyze each train source
    analyze_source("Train Source 1 (S1)", config.TRAIN_S1)
    analyze_source("Train Source 2 (S2)", config.TRAIN_S2)
    analyze_source("Train Source 3 (S3)", config.TRAIN_S3)
    
    # Analyze ground truth
    analyze_ground_truth(config.TRAIN_GT)
    
    # Analyze test sources
    analyze_source("Test Source 1 (S1)", config.TEST_S1)
    analyze_source("Test Source 2 (S2)", config.TEST_S2)
    analyze_source("Test Source 3 (S3)", config.TEST_S3)
    
    print("\n" + "=" * 70)
    print("  EDA COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_eda()
