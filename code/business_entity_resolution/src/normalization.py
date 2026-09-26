"""Stage 3: Text Normalization — business names, addresses, countries.

Produces normalized columns used downstream by blocking and feature engineering:
  - norm_name:    lowercased, accent-stripped (Latin only), legal suffixes expanded
  - core_name:    norm_name with legal entity designations removed entirely
  - norm_address: lowercased, abbreviations expanded, whitespace collapsed
  - postal_code:  extracted postal/zip/PIN code (if present)

Design notes:
  - Accent stripping uses NFKD decomposition, applied ONLY to Latin characters.
    Indic scripts (Devanagari, Telugu, Tamil, etc.) are preserved as-is.
  - Legal suffix expansion is country-agnostic: covers US, India, France, UK forms.
  - Core name extraction strips legal designations so "Ram Marketing Pvt Ltd"
    becomes "ram marketing" — useful for matching across jurisdictions.
  - Does NOT transliterate Indic text — that's handled by E5 embeddings downstream.

Usage:
    python -m src.normalization
"""
import os
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd

from . import config


# ═══════════════════════════════════════════════════════════════════
# LEGAL SUFFIX MAPPINGS
# ═══════════════════════════════════════════════════════════════════
# Abbreviation -> full expansion (all lowercase)
LEGAL_SUFFIX_MAP = {
    # India
    'pvt': 'private',
    'pvt.': 'private',
    'ltd': 'limited',
    'ltd.': 'limited',
    'llp': 'llp',
    'opc': 'one person company',
    # US / General
    'inc': 'incorporated',
    'inc.': 'incorporated',
    'corp': 'corporation',
    'corp.': 'corporation',
    'llc': 'llc',
    'l.l.c.': 'llc',
    'l.l.c': 'llc',
    'co': 'company',
    'co.': 'company',
    'intl': 'international',
    "int'l": 'international',
    'assoc': 'associates',
    'assoc.': 'associates',
    'bros': 'brothers',
    'bros.': 'brothers',
    'mfg': 'manufacturing',
    'mfg.': 'manufacturing',
    'svcs': 'services',
    'svc': 'service',
    'grp': 'group',
    'grp.': 'group',
    'hldgs': 'holdings',
    'hldg': 'holding',
    # France
    'sarl': 'sarl',
    's.a.r.l.': 'sarl',
    's.a.r.l': 'sarl',
    'sa': 'sa',
    's.a.': 'sa',
    'sas': 'sas',
    's.a.s.': 'sas',
    'ste': 'societe',
    'sté': 'societe',
    'ets': 'etablissements',
    'cie': 'compagnie',
    # UK
    'plc': 'plc',
    'p.l.c.': 'plc',
    'ltd.': 'limited',
}

# Legal designations to REMOVE for core_name extraction
# These are full words/phrases that indicate legal entity type
LEGAL_DESIGNATIONS = [
    # Multi-word (check these first)
    'private limited',
    'pvt ltd',
    'pvt. ltd.',
    'pvt. ltd',
    'pvt ltd.',
    'one person company',
    # Single word
    'limited',
    'private',
    'incorporated',
    'corporation',
    'llc',
    'llp',
    'plc',
    'sarl',
    'sas',
    'sa',
    'company',
    'societe',
    'etablissements',
    'compagnie',
    'foundation',
    'trust',
    'enterprises',
    'enterprise',
    'holdings',
    'holding',
    'group',
    'associates',
    'solutions',
    'industries',
    'technologies',
    'international',
]

# ═══════════════════════════════════════════════════════════════════
# ADDRESS ABBREVIATION MAPPINGS
# ═══════════════════════════════════════════════════════════════════
ADDRESS_ABBREV_MAP = {
    'rd': 'road',
    'rd.': 'road',
    'st': 'street',
    'st.': 'street',
    'ave': 'avenue',
    'ave.': 'avenue',
    'blvd': 'boulevard',
    'blvd.': 'boulevard',
    'dr': 'drive',
    'dr.': 'drive',
    'ln': 'lane',
    'ln.': 'lane',
    'ct': 'court',
    'ct.': 'court',
    'pl': 'place',
    'pl.': 'place',
    'pkwy': 'parkway',
    'hwy': 'highway',
    'cir': 'circle',
    'sq': 'square',
    'apt': 'apartment',
    'apt.': 'apartment',
    'ste': 'suite',
    'ste.': 'suite',
    'bldg': 'building',
    'bldg.': 'building',
    'fl': 'floor',
    'fl.': 'floor',
    'dept': 'department',
    'dept.': 'department',
    'dist': 'district',
    'dist.': 'district',
    'nagar': 'nagar',  # keep as-is (Indian locality)
    'marg': 'marg',    # keep as-is (Indian road type)
    'no': 'number',
    'no.': 'number',
    'nr': 'near',
    'nr.': 'near',
    'opp': 'opposite',
    'opp.': 'opposite',
    'govt': 'government',
    'govt.': 'government',
}

# Regex to extract postal codes from addresses
# US ZIP:  5 digits or 5+4 format
# India PIN: 6 digits
# France: 5 digits
POSTAL_CODE_PATTERN = re.compile(
    r'\b(\d{5}(?:-\d{4})?)\b'  # US ZIP (12345 or 12345-6789)
    r'|'
    r'\b(\d{6})\b'             # India PIN (6 digits)
)


# ═══════════════════════════════════════════════════════════════════
# NORMALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def strip_accents_latin_only(text):
    """Strip accents from Latin characters only, preserving Indic scripts.
    
    Uses NFKD decomposition: 'é' -> 'e' + combining accent -> 'e'
    Only strips combining marks that follow Latin base characters.
    """
    if not isinstance(text, str):
        return text
    
    # NFKD decompose
    nfkd = unicodedata.normalize('NFKD', text)
    
    result = []
    for ch in nfkd:
        cat = unicodedata.category(ch)
        # Skip combining marks (Mn = Mark, Nonspacing) 
        # These are accent marks from Latin decomposition
        if cat == 'Mn':
            continue
        result.append(ch)
    
    return ''.join(result)


def normalize_business_name(name):
    """Normalize a business name for matching.
    
    Returns:
        norm_name: fully normalized name
    """
    if not isinstance(name, str) or not name.strip():
        return ''
    
    # Lowercase
    text = name.lower()
    
    # Strip accents (Latin only — Indic scripts unaffected because
    # Devanagari/Telugu/etc. don't decompose into base + combining mark)
    text = strip_accents_latin_only(text)
    
    # Replace & with 'and'
    text = text.replace('&', ' and ')
    
    # Remove punctuation EXCEPT hyphens and apostrophes (they can be meaningful)
    # e.g., "O'Brien", "Coca-Cola"
    text = re.sub(r"[^\w\s\-']", ' ', text)
    
    # Expand legal suffix abbreviations
    tokens = text.split()
    expanded = []
    for token in tokens:
        lower_tok = token.strip("'\".,;:")
        if lower_tok in LEGAL_SUFFIX_MAP:
            expanded.append(LEGAL_SUFFIX_MAP[lower_tok])
        else:
            expanded.append(token)
    text = ' '.join(expanded)
    
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_core_name(norm_name):
    """Extract core business name by removing legal designations.
    
    'ram marketing private limited' -> 'ram marketing'
    'abc corporation llc'           -> 'abc'
    """
    if not norm_name:
        return ''
    
    text = norm_name
    
    # Remove multi-word designations first (longer patterns first)
    sorted_desigs = sorted(LEGAL_DESIGNATIONS, key=len, reverse=True)
    for desig in sorted_desigs:
        # Word-boundary removal to avoid partial matches
        pattern = r'\b' + re.escape(desig) + r'\b'
        text = re.sub(pattern, '', text)
    
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # If stripping removed everything, fall back to original
    if not text:
        return norm_name
    
    return text


def normalize_address(address):
    """Normalize a business address for matching.
    
    Returns:
        norm_address: normalized address string
    """
    if not isinstance(address, str) or not address.strip():
        return ''
    
    # Lowercase
    text = address.lower()
    
    # Strip accents (Latin only)
    text = strip_accents_latin_only(text)
    
    # Replace & with 'and'
    text = text.replace('&', ' and ')
    
    # Light punctuation removal (keep commas, hyphens, slashes — they separate address parts)
    text = re.sub(r"[^\w\s,\-/.]", ' ', text)
    
    # Expand address abbreviations
    tokens = text.split()
    expanded = []
    for token in tokens:
        lower_tok = token.strip(".,;:")
        if lower_tok in ADDRESS_ABBREV_MAP:
            expanded.append(ADDRESS_ABBREV_MAP[lower_tok])
        else:
            expanded.append(token)
    text = ' '.join(expanded)
    
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_postal_code(address):
    """Extract postal/ZIP/PIN code from an address string.
    
    Returns the first match or empty string.
    """
    if not isinstance(address, str):
        return ''
    
    match = POSTAL_CODE_PATTERN.search(address)
    if match:
        # Return whichever group matched
        return match.group(1) or match.group(2) or ''
    return ''


def normalize_dataframe(df):
    """Apply all normalizations to a DataFrame.
    
    Adds columns: norm_name, core_name, norm_address, postal_code
    """
    # Normalize business name
    df['norm_name'] = df[config.COL_NAME].apply(normalize_business_name)
    
    # Extract core name (legal designations stripped)
    df['core_name'] = df['norm_name'].apply(extract_core_name)
    
    # Normalize address
    df['norm_address'] = df[config.COL_ADDRESS].apply(normalize_address)
    
    # Extract postal code (from original address, before normalization mangles it)
    df['postal_code'] = df[config.COL_ADDRESS].apply(extract_postal_code)
    
    return df


# ═══════════════════════════════════════════════════════════════════
# PIPELINE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def normalize_source_file(input_path, output_path, source_name, chunksize=500_000):
    """Normalize a cleaned TSV file in chunks and write the result."""
    print(f"\n{'='*60}")
    print(f"  Normalizing: {source_name}")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")
    
    if not input_path.exists():
        print(f"  ERROR: Input file not found!")
        return
    
    os.makedirs(output_path.parent, exist_ok=True)
    
    t0 = time.time()
    total_rows = 0
    first_chunk = True
    
    reader = pd.read_csv(
        input_path, sep='\t', encoding='utf-8',
        low_memory=False, chunksize=chunksize,
        dtype=str,
        keep_default_na=False,
    )
    
    for chunk in reader:
        chunk = normalize_dataframe(chunk)
        total_rows += len(chunk)
        
        chunk.to_csv(
            output_path, sep='\t', index=False,
            mode='w' if first_chunk else 'a',
            header=first_chunk,
            encoding='utf-8',
        )
        first_chunk = False
        
        print(f"  Processed {total_rows:,} rows...")
    
    elapsed = time.time() - t0
    print(f"  Done: {total_rows:,} rows normalized in {elapsed:.1f}s")
    
    # Show sample of normalized output
    sample = pd.read_csv(output_path, sep='\t', nrows=5, encoding='utf-8', dtype=str)
    print(f"\n  Sample normalized rows:")
    for _, row in sample.iterrows():
        print(f"    name:      {row.get(config.COL_NAME, '')[:50]}")
        print(f"    norm_name: {row.get('norm_name', '')[:50]}")
        print(f"    core_name: {row.get('core_name', '')[:50]}")
        print(f"    address:   {row.get(config.COL_ADDRESS, '')[:50]}")
        print(f"    norm_addr: {row.get('norm_address', '')[:50]}")
        print(f"    postal:    {row.get('postal_code', '')}")
        print()


def run_normalization():
    """Run normalization on all cleaned source files."""
    print("=" * 60)
    print("  TEXT NORMALIZATION")
    print("=" * 60)
    
    clean_dir = config.DATA_ROOT / "cleaned"
    norm_dir = config.DATA_ROOT / "normalized"
    os.makedirs(norm_dir, exist_ok=True)
    
    files = [
        (clean_dir / "train_source1_clean.tsv", norm_dir / "train_source1_norm.tsv", "Train S1"),
        (clean_dir / "train_source2_clean.tsv", norm_dir / "train_source2_norm.tsv", "Train S2"),
        (clean_dir / "train_source3_clean.tsv", norm_dir / "train_source3_norm.tsv", "Train S3"),
        (clean_dir / "test_source1_clean.tsv",  norm_dir / "test_source1_norm.tsv",  "Test S1"),
        (clean_dir / "test_source2_clean.tsv",  norm_dir / "test_source2_norm.tsv",  "Test S2"),
        (clean_dir / "test_source3_clean.tsv",  norm_dir / "test_source3_norm.tsv",  "Test S3"),
    ]
    
    for input_path, output_path, source_name in files:
        normalize_source_file(input_path, output_path, source_name)
    
    print("\n" + "=" * 60)
    print("  NORMALIZATION COMPLETE")
    print(f"  Normalized files saved to: {norm_dir}")
    print("=" * 60)


if __name__ == "__main__":
    run_normalization()
