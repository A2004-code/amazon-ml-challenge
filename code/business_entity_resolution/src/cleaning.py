"""Stage 2: Data Cleaning — nulls, invalid records, character fixes, dedup.

Reads raw TSV files, applies cleaning transformations, and writes cleaned
versions to a staging directory. Designed to process millions of rows
efficiently using chunked pandas reads.

Cleaning steps:
1. Strip leading/trailing whitespace from all string columns
2. Normalize internal whitespace (collapse multiple spaces)
3. Remove zero-width characters (ZWNJ, ZWJ, ZWSP, BOM, etc.)
4. Handle missing/null values (empty string for name/address)
5. Remove rows with missing entity_id
6. Remove exact duplicate entity_ids (keep first occurrence)
7. Standardize country names (trim, title-case)
8. Flag and report anomalies

Usage:
    python -m src.cleaning
"""
import os
import re
import time
from pathlib import Path

import pandas as pd

from . import config


# ── Cleaning Functions ────────────────────────────────────────────

def strip_zero_width(text):
    """Remove zero-width Unicode characters from a string."""
    if not isinstance(text, str):
        return text
    for ch in config.ZERO_WIDTH_CHARS:
        text = text.replace(ch, '')
    return text


def collapse_whitespace(text):
    """Collapse runs of whitespace into a single space and strip."""
    if not isinstance(text, str):
        return text
    return re.sub(r'\s+', ' ', text).strip()


def clean_dataframe(df):
    """Apply all cleaning transformations to a DataFrame.
    
    Args:
        df: pandas DataFrame with columns [entity_id, business_name, 
            business_address, country]
    
    Returns:
        Cleaned DataFrame (may have fewer rows if duplicates/invalids removed).
    """
    n_before = len(df)
    
    # 1. Strip whitespace from all string columns
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    
    # 2. Remove zero-width characters from name and address
    df[config.COL_NAME] = df[config.COL_NAME].apply(strip_zero_width)
    df[config.COL_ADDRESS] = df[config.COL_ADDRESS].apply(strip_zero_width)
    
    # 3. Collapse internal whitespace
    df[config.COL_NAME] = df[config.COL_NAME].apply(collapse_whitespace)
    df[config.COL_ADDRESS] = df[config.COL_ADDRESS].apply(collapse_whitespace)
    
    # 4. Handle missing values — fill NaN/None with empty string
    df[config.COL_NAME] = df[config.COL_NAME].fillna('')
    df[config.COL_ADDRESS] = df[config.COL_ADDRESS].fillna('')
    
    # Replace pandas 'nan' string artifacts
    df[config.COL_NAME] = df[config.COL_NAME].replace('nan', '')
    df[config.COL_ADDRESS] = df[config.COL_ADDRESS].replace('nan', '')
    
    # 5. Remove rows with missing entity_id
    mask_no_eid = df[config.COL_ENTITY_ID].isna() | (df[config.COL_ENTITY_ID] == '') | (df[config.COL_ENTITY_ID] == 'nan')
    n_no_eid = mask_no_eid.sum()
    if n_no_eid > 0:
        df = df[~mask_no_eid]
    
    # 6. Remove exact duplicate entity_ids (keep first)
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset=[config.COL_ENTITY_ID], keep='first')
    n_dupes = n_before_dedup - len(df)
    
    # 7. Standardize country names (trim, title-case)
    df[config.COL_COUNTRY] = df[config.COL_COUNTRY].str.strip().str.title()
    
    n_after = len(df)
    
    stats = {
        'rows_before': n_before,
        'rows_after': n_after,
        'removed_no_eid': n_no_eid,
        'removed_dupes': n_dupes,
        'removed_total': n_before - n_after,
    }
    
    return df, stats


def clean_source_file(input_path, output_path, source_name, chunksize=500_000):
    """Clean a source TSV file in chunks and write the cleaned version.
    
    Args:
        input_path: Path to raw TSV file
        output_path: Path to write cleaned TSV file
        source_name: Display name for logging (e.g. "Train S1")
        chunksize: Number of rows per chunk for memory efficiency
    """
    print(f"\n{'='*60}")
    print(f"  Cleaning: {source_name}")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")
    
    if not input_path.exists():
        print(f"  ERROR: Input file not found!")
        return None
    
    os.makedirs(output_path.parent, exist_ok=True)
    
    t0 = time.time()
    total_stats = {
        'rows_before': 0,
        'rows_after': 0,
        'removed_no_eid': 0,
        'removed_dupes': 0,
        'removed_total': 0,
    }
    
    first_chunk = True
    chunks_processed = 0
    
    reader = pd.read_csv(
        input_path, sep='\t', encoding='utf-8',
        low_memory=False, chunksize=chunksize,
        dtype=str,  # read everything as string to avoid type issues
        keep_default_na=False,  # don't convert empty strings to NaN
    )
    
    for chunk in reader:
        cleaned, stats = clean_dataframe(chunk)
        
        # Accumulate stats
        for k in total_stats:
            total_stats[k] += stats[k]
        
        # Write chunk
        cleaned.to_csv(
            output_path, sep='\t', index=False,
            mode='w' if first_chunk else 'a',
            header=first_chunk,
            encoding='utf-8',
        )
        first_chunk = False
        chunks_processed += 1
        
        if chunks_processed % 5 == 0:
            print(f"  Processed {total_stats['rows_before']:,} rows so far...")
    
    elapsed = time.time() - t0
    
    print(f"\n  Results:")
    print(f"    Rows before:     {total_stats['rows_before']:,}")
    print(f"    Rows after:      {total_stats['rows_after']:,}")
    print(f"    Removed (no ID): {total_stats['removed_no_eid']:,}")
    print(f"    Removed (dupes): {total_stats['removed_dupes']:,}")
    print(f"    Total removed:   {total_stats['removed_total']:,}")
    print(f"    Time: {elapsed:.1f}s")
    
    return total_stats


def deduplicate_across_chunks(filepath, source_name):
    """Second pass: remove any cross-chunk duplicate entity_ids.
    
    Since we process in chunks, a duplicate entity_id could appear in
    different chunks. This pass catches those.
    """
    print(f"\n  Cross-chunk dedup for {source_name}...")
    
    seen_ids = set()
    dupes_found = 0
    temp_path = filepath.with_suffix('.tmp')
    
    with open(filepath, 'r', encoding='utf-8') as fin, \
         open(temp_path, 'w', encoding='utf-8') as fout:
        header = fin.readline()
        fout.write(header)
        
        for line in fin:
            eid = line.split('\t')[0]
            if eid in seen_ids:
                dupes_found += 1
                continue
            seen_ids.add(eid)
            fout.write(line)
    
    # Replace original with deduped
    os.replace(temp_path, filepath)
    print(f"  Cross-chunk duplicates removed: {dupes_found:,}")
    print(f"  Final unique entities: {len(seen_ids):,}")
    return dupes_found


def run_cleaning():
    """Run data cleaning on all train and test source files."""
    print("=" * 60)
    print("  DATA CLEANING")
    print("=" * 60)
    
    # Output directory for cleaned files
    clean_dir = config.DATA_ROOT / "cleaned"
    os.makedirs(clean_dir, exist_ok=True)
    
    # Define input -> output mappings
    files_to_clean = [
        (config.TRAIN_S1, clean_dir / "train_source1_clean.tsv", "Train S1"),
        (config.TRAIN_S2, clean_dir / "train_source2_clean.tsv", "Train S2"),
        (config.TRAIN_S3, clean_dir / "train_source3_clean.tsv", "Train S3"),
        (config.TEST_S1,  clean_dir / "test_source1_clean.tsv",  "Test S1"),
        (config.TEST_S2,  clean_dir / "test_source2_clean.tsv",  "Test S2"),
        (config.TEST_S3,  clean_dir / "test_source3_clean.tsv",  "Test S3"),
    ]
    
    for input_path, output_path, source_name in files_to_clean:
        stats = clean_source_file(input_path, output_path, source_name)
        if stats is not None:
            deduplicate_across_chunks(output_path, source_name)
    
    print("\n" + "=" * 60)
    print("  CLEANING COMPLETE")
    print(f"  Cleaned files saved to: {clean_dir}")
    print("=" * 60)


if __name__ == "__main__":
    run_cleaning()
