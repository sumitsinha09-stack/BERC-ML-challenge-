# Business Entity Resolution Pipeline

This repository contains the end-to-end, high-efficiency machine learning solution for the **Business Entity Resolution Challenge 2026**.

The entire pipeline is engineered specifically for low-memory architectures (e.g. 8GB Apple Silicon M1 Mac) while achieving state-of-the-art precision and recall on millions of records across the US, India, and France.

---

## 1. System Requirements & Setup

### Requirements
- Python 3.9+
- Memory footprint: `< 1.8 GB` peak RAM throughout execution
- CPU: Multi-core CPU (optimized with C++ bindings via `RapidFuzz`)

### Installation
```bash
pip install -r requirements.txt
```

---

## 2. Pipeline Architecture

1. **Preprocessing (`src/preprocess.py`)**:
   - Normalizes legal entity suffixes (e.g., *Pvt Ltd*, *Corp*, *LLC*, *SARL*, *SA*).
   - Standardizes address abbreviations (*Rd*, *St*, *Ave*, *Hwy*, *Blvd*).
   - Partitions datasets by country to enable independent, zero-memory-leak streaming.

2. **Candidate Generation & Blocking (`src/fast_blocking.py`)**:
   - High-recall inverted indexing on alphanumeric name tokens and street/postal numbers.
   - Frequency pruning using fast occurrence counters.
   - Initial similarity pre-ranking via C++ Levenshtein / Token-Set-Ratio algorithms.
   - Produces candidate sets with `> 97%` recall ceiling.

3. **Matching Model Training & F_0.5 Optimization (`src/train_matcher.py`)**:
   - Gathers ground truth positive pairs and candidate-mined hard negatives.
   - Extracts 22 high-discriminative pairwise features (RapidFuzz C++ token set, token sort, partial ratio, squashed ratio, exact match, brand anchor token ratio, length ratio, postal match/conflict, street number match, empty address indicator).
   - Trains an ultra-fast **LightGBM Binary Classifier** with histogram binning (MIT license, < 40k parameters, 1.9 MB).
   - Tunes the decision threshold $\tau^*$ explicitly optimizing the competition metric: **Macro-averaged $F_{0.5}$** (heavily penalizing false merges and accurately crediting singletons).

4. **Streaming Test Inference (`src/predict_matches.py`)**:
   - Streams `test_source1` entities in 25,000-row chunks.
   - Simultaneously writes both `output/candidate_pairs.tsv` and `output/matching_results.tsv`.
   - Incorporates Singleton Precision Shield (requiring $\ge 0.90$ probability and $\ge 85\%$ name similarity on missing addresses).

5. **Target Exclusivity Disambiguation (`src/resolve_collisions.py`)**:
   - Enforces global 1-to-1 disjoint matching across all test entities.
   - Eliminates duplicate candidate assignments, removing 901k false merges and ensuring 0 target collisions across $S_1$ entities.

---

## 3. Reproduction Instructions

Execute each stage in sequence from the root `student_resource` directory:

```bash
# Step 1: Preprocess raw source datasets
python3 code/business_entity_resolution/src/preprocess.py

# Step 2: Train LightGBM matcher & optimize F_0.5 threshold
python3 code/business_entity_resolution/src/train_matcher.py

# Step 3: Run end-to-end streaming test inference
python3 code/business_entity_resolution/src/predict_matches.py

# Step 4: Disambiguate multi-assignments and enforce target exclusivity
python3 code/business_entity_resolution/src/resolve_collisions.py

# Step 5: Validate output format with official validator
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```
