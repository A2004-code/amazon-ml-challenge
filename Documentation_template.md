# Documentation — Business Entity Resolution Pipeline

## Approach Summary
Our solution uses a **two-stage blocking and reranking architecture** designed to handle over 9 trillion possible entity combinations ($1.5M \times 6M$). We reduce the search space using TF-IDF and MinHash blocking, then extract over 30 string and semantic features per pair. To handle cross-lingual and cross-script variations (highly prevalent in the India dataset), we utilize GPU-accelerated multilingual E5 embeddings. Finally, a LightGBM classifier predicts the match probability, with a threshold explicitly tuned for the F0.5 metric (heavily prioritizing precision).

---

## Pipeline Phases (The 7-Step Architecture)

### 1. Exploratory Data Analysis (EDA)
- **What happens:** The pipeline analyzes the distribution of countries, languages, and character scripts across the datasets. 
- **Why it matters:** We discovered that while the US data is mostly standard English, the India data contains heavy cross-script representations (e.g., matching English letters to Hindi or Telugu scripts). We also detected and mapped "Mojibake" (encoding corruptions) and zero-width characters.

### 2. Data Cleaning
- **What happens:** We strip out invisible zero-width characters (like ZWNJs, which are common in Indic scripts but ruin string matching). We remove all punctuation, convert strings to lowercase, and unify whitespace.
- **Why it matters:** Standardizes the text so that basic differences in formatting or accidental double-spaces don't penalize similarity scores.

### 3. Normalization
- **What happens:** 
  1. **Legal Term Stripping:** We use regex to remove common corporate designations like `LLC, Inc, Pvt, Ltd, SARL, SAS`.
  2. **Alphanumeric Sorting:** Words in the entity names are sorted alphabetically.
- **Why it matters:** Stripping legal terms prevents two completely different companies from getting a high similarity score just because they both end in "Pvt Ltd". Sorting words ensures that "Ram Marketing" and "Marketing Ram" yield a 100% match.

### 4. Blocking (Candidate Generation)
- **What happens:** We cannot compare every Source 1 entity against every Source 2 entity (that would be 9 trillion comparisons and would take years). Instead, we use **TF-IDF on character n-grams** and **MinHash LSH** to quickly find entities that share similar character sequences. We also enforce a strict country-matching rule.
- **Why it matters:** This reduces the search space from 9 Trillion pairs down to a highly targeted ~100 Million candidate pairs. If a true match is missed in this step, it is lost forever (Recall Ceiling), so we cast a wide but smart net.

### 5. Traditional Feature Engineering
- **What happens:** For every single pair in the 100M candidates, we calculate over 30 different string comparison metrics between their names and addresses.
- **Features include:** Jaro-Winkler, Levenshtein distance, Token Overlap, Longest Common Subsequence (LCS), and numerical exact matches (e.g., checking if the Zip Codes match perfectly).
- **Why it matters:** These features easily solve the 60% of the dataset that consists of standard English character overlaps (e.g. typos, missing words).

### 6. E5 Embeddings (Semantic Cross-Lingual Matching)
- **What happens:** String features completely fail when matching "Ram Marketing" to "राम मार्केटिंग". To solve this, we load `intfloat/multilingual-e5-base` onto the GPU. It reads the raw text and converts it into a 768-dimensional mathematical vector (embedding) that represents the *meaning* and *phonetics* of the word across 100+ languages. 
- **The Engine:** Because there are 100M pairs, holding this in memory causes server crashes (OOM). We built a highly optimized two-pass disk-to-RAM memory pipeline that processes Name vectors and Address vectors in separate 22 GB batches.
- **Why it matters:** We calculate the Cosine Similarity between these vectors. This acts as the ultimate cross-script translation layer.

### 7. LightGBM Classifier & Threshold Tuning
- **What happens:** All 30+ string features and 2 E5 embedding features are fed into a LightGBM Gradient Boosting model. The model learns exactly how much weight to give each feature (e.g., it learns to trust E5 heavily for Indian data, but trust Jaro-Winkler for US data).
- **Thresholding:** By default, models classify anything > 0.50 as a match. Because the leaderboard uses **F0.5** (which penalizes False Positives twice as much as False Negatives), we run an optimizer to find the perfect cutoff threshold (e.g., only predicting a match if probability > 0.82) to maximize the competition score.

---

## Key Design Decisions & Overcoming Hurdles

1. **Memory Map (mmap) Bottleneck Fixed:** Initially, using memory-mapped arrays for the 44GB of embeddings caused the RunPod's network-attached storage to freeze due to millions of random I/O reads. We re-engineered Stage 6 to use a Two-Pass RAM architecture (Peak 25GB RAM) which dropped processing time from 4+ hours down to 20 minutes.
2. **Strict Submission Schema Enforcement:** The validator rejects any submission whose column headers don't exactly match the spec (`source1_entity_id`, `matched_entity_ids`). We explicitly separated internal pipeline aliases from the final submission writer to guarantee zero format rejections.
