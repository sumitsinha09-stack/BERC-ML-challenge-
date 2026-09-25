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
   - Extracts 14 pairwise features (token set ratio, token sort ratio, squashed domain ratio, exact match, numeric address overlap, address length diffs, source indicators).
   - Trains an ultra-fast **LightGBM Binary Classifier** with histogram binning.
   - Tunes the decision threshold $\tau^*$ explicitly optimizing the competition metric: **Macro-averaged $F_{0.5}$** (heavily penalizing false merges and accurately crediting singletons).

4. **Streaming Test Inference (`src/predict_matches.py`)**:
   - Streams `test_source1` entities in 25,000-row chunks.
   - Simultaneously writes both:
     - `output/candidate_pairs.tsv`
     - `output/matching_results.tsv`
   - Guarantees strict subset constraints, no duplicate entity IDs, and constant flat memory consumption.

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

# Step 4: Validate output format
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```
