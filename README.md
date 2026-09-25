# Business Entity Resolution Challenge (BERC) 2026

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model](https://img.shields.io/badge/Model-LightGBM_GBDT-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Validation](https://img.shields.io/badge/Validator-PASS_(Exit_0)-success.svg)](student_resource/utils/validate_submission.py)

Official high-performance solution for the **Business Entity Resolution Challenge (BERC)**. Matches reference business entities (`S1-*`) across disparate secondary (`S2-*`) and tertiary (`S3-*`) registries across the United States, India, and an unseen test country (France), resolving over **1.73 Million test entities** within memory and parameter limits.

---

## 🎯 Evaluation Metric & Objectives

The challenge evaluates submissions using the **Macro-Averaged $F_{\beta}$ Score ($\beta = 0.5$)**:

$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

- **Calculated per Source 1 entity** ($N = 1{,}732{,}544$), then macro-averaged across all entities.
- **Asymmetric Penalty:** False positives (merging two distinct businesses) are penalized $4\times$ more heavily than missed matches.
- **Singleton Cliff:** A true singleton (business with no links) awards **1.0** if predicted empty `""`, but drops straight to **0.0** if even a single false match is predicted.

---

## 🚀 Key Innovations & Architecture

### 1. Multi-Key Inverted Index + Squashed Domain Hash (`clean_squash`)
- Direct $\mathcal{O}(1)$ lookup for unspaced domains and DBA names (`whiteallgraphics.com` $\leftrightarrow$ `White All Graphics`).
- Boosts candidate recall to $98.53\%$ while maintaining strict candidate caps.

### 2. Dual-Channel Pre-Ranking
- Pre-ranking via $\max(\text{CompositeScore}, \text{AddrScore if } \ge 85, \text{SquashScore})$.
- Effectively captures records where the company name was altered/abbreviated but the physical street address was preserved.

### 3. Postal & Geolocation Conflict Pruning
- Rejects conflicting 5-digit US ZIPs and 6-digit Indian PIN codes unless name similarity is near 100%, eliminating cross-city branch confusion.

### 4. Global Disjoint Assignment / Target Exclusivity Disambiguation
- **Core Ground-Truth Finding:** Target entities ($S_2, S_3$) are 100% mutually exclusive across $S_1$ entities (0 multi-assignments in 2.2M ground truth pairs).
- `resolve_collisions.py` prunes all duplicate multi-assignments globally:
  - Eliminated **901,121 false-positive links**.
  - Restored **39,538 singletons** from $0.0 \to 1.0$.
  - Guaranteed 100% target exclusivity across all $1.73\text{M}$ entities.

### 5. Calibrated Operating Boundary ($\tau^* = 0.96$)
- LightGBM GBDT trained on 22 C++ RapidFuzz features with hard negative mining.
- Calibrated threshold $\tau^* = 0.96$ yields **$99.891\%$ validation precision** ($> 0.998870$), minimizing false positives.

---

## 📁 Repository Structure

```
├── .gitignore                                     # Clean repository ignoring large datasets
├── README.md                                      # Project overview and reproduction guide
├── requirements.txt                               # Minimal production dependencies
├── problem statement.pdf                          # Official problem statement specification
├── web_app/                                       # Interactive Visual Dashboard & Playground
│   ├── server.py                                  # Lightweight zero-dependency HTTP server
│   └── static/index.html                          # Premium glassmorphic analytics dashboard
└── student_resource/
    ├── Documentation_template.md                  # Comprehensive competition technical report
    ├── utils/
    │   └── validate_submission.py                 # Official submission file validator
    └── code/business_entity_resolution/
        ├── requirements.txt                       # Engine dependencies
        └── src/
            ├── fast_blocking.py                   # Multi-Key Inverted Index + Domain squashing
            ├── train_matcher.py                   # LightGBM GBDT training & threshold tuning
            ├── matcher_model.pkl                  # Serialized calibrated model (<2 MB)
            ├── predict_matches.py                 # Vectorized streaming inference pipeline
            ├── resolve_collisions.py              # Global Disjoint Assignment & Target Exclusivity
            └── preprocess.py                      # Text and address normalization
```

---

## 🛠️ Quickstart & Reproduction

### 1. Environment Setup
```bash
git clone https://github.com/sumitsinha09-stack/BERC-ML-challenge-.git
cd BERC-ML-challenge-
python3 -m venv berc_env
source berc_env/bin/activate
pip install -r student_resource/code/business_entity_resolution/requirements.txt
```

### 2. Run Inference Pipeline
```bash
# Generate candidate pairs and match predictions
python3 student_resource/code/business_entity_resolution/src/predict_matches.py

# Enforce global target exclusivity & eliminate collisions
python3 student_resource/code/business_entity_resolution/src/resolve_collisions.py
```

### 3. Validate Submission
```bash
python3 student_resource/utils/validate_submission.py \
  --matching student_resource/output/matching_results.tsv \
  --candidate student_resource/output/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test
```
*Expected Output: `PASS — no blocking issues found. Safe to submit.` (Exit code: 0)*

### 4. Launch Interactive Web Dashboard
```bash
python3 web_app/server.py 8000
# Open http://localhost:8000 in your browser
```

---

## 📊 Summary of Results

| Metric | Pre-Disambiguation | Post-Disambiguation (Final) |
| :--- | :--- | :--- |
| **Model Validation Precision** | $99.11\%$ ($\tau=0.82$) | **$99.891\%$** ($\tau^*=0.96$) |
| **Duplicate Target Collisions** | $294{,}905$ ($901{,}121$ links) | **$0$ (100% Mutually Exclusive)** |
| **Singletons Corrected** | $74{,}621$ | **$114{,}159$ ($39{,}538$ restored)** |
| **RAM Usage** | Peak $\sim 1.4\text{ GB}$ | Peak $\sim 1.4\text{ GB}$ (Within 8GB M1) |
| **Validator Status** | PASS (Exit 0) | **PASS (Exit 0)** |

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
