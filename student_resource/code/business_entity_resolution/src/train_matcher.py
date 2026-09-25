import os
import sys
import gc
import re
import time
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from rapidfuzz import fuzz
import lightgbm as lgb
from sklearn.model_selection import train_test_split

def squash_name(name):
    return name.replace(' ', '').replace('com', '').replace('org', '').replace('net', '').replace('www', '')

def extract_features(s1_name, s1_addr, c_name, c_addr, c_id):
    """
    Extract 22 high-discriminative pairwise features between S1 and Candidate.
    Runs in microseconds via C++ RapidFuzz bindings.
    """
    s1_sq = squash_name(s1_name)
    c_sq = squash_name(c_name)
    
    s1_words = [w for w in s1_name.split() if w]
    c_words = [w for w in c_name.split() if w]
    s1_tok_set = set(s1_words)
    c_tok_set = set(c_words)
    
    # 1-6. Standard Name Ratios
    n_ts_ratio = fuzz.token_set_ratio(s1_name, c_name)
    n_sort_ratio = fuzz.token_sort_ratio(s1_name, c_name)
    n_ratio = fuzz.ratio(s1_name, c_name)
    n_partial = fuzz.partial_ratio(s1_name, c_name)
    n_sq_ratio = fuzz.ratio(s1_sq, c_sq)
    n_exact = 1.0 if s1_name and s1_name == c_name else 0.0
    
    # 7-8. First Token / Brand Anchor
    if s1_words and c_words:
        n_first_token_ratio = fuzz.ratio(s1_words[0], c_words[0])
        n_first_token_exact = 1.0 if s1_words[0] == c_words[0] else 0.0
    else:
        n_first_token_ratio = 0.0
        n_first_token_exact = 0.0
        
    # 9. Token Containment
    min_tok_len = min(len(s1_tok_set), len(c_tok_set))
    n_containment = len(s1_tok_set.intersection(c_tok_set)) / min_tok_len if min_tok_len > 0 else 0.0
    
    # 10-11. Length Features
    max_len = max(len(s1_name), len(c_name))
    min_len = min(len(s1_name), len(c_name))
    n_len_ratio = (min_len / max_len) if max_len > 0 else 1.0
    len_diff = abs(len(s1_name) - len(c_name))
    
    # Address Features
    addr_empty = 1.0 if not s1_addr or not c_addr else 0.0
    if not addr_empty:
        a_ts_ratio = fuzz.token_set_ratio(s1_addr, c_addr)
        a_sort_ratio = fuzz.token_sort_ratio(s1_addr, c_addr)
        
        s1_addr_words = set(s1_addr.split())
        c_addr_words = set(c_addr.split())
        min_addr_tok = min(len(s1_addr_words), len(c_addr_words))
        a_containment = len(s1_addr_words.intersection(c_addr_words)) / min_addr_tok if min_addr_tok > 0 else 0.0
        
        # Numeric extraction (all numbers 2-6 digits)
        s1_nums = set(re.findall(r'\b\d{2,6}\b', s1_addr))
        c_nums = set(re.findall(r'\b\d{2,6}\b', c_addr))
        if s1_nums and c_nums:
            inter = len(s1_nums.intersection(c_nums))
            union = len(s1_nums.union(c_nums))
            num_jaccard = inter / union
            num_exact = 1.0 if inter > 0 else 0.0
        else:
            num_jaccard = 0.5 if not s1_nums and not c_nums else 0.0
            num_exact = 0.5 if not s1_nums and not c_nums else 0.0
            
        # Postal code extraction (5-digit US or 6-digit India)
        s1_postals = set(re.findall(r'\b\d{5,6}\b', s1_addr))
        c_postals = set(re.findall(r'\b\d{5,6}\b', c_addr))
        if s1_postals and c_postals:
            if s1_postals.intersection(c_postals):
                a_postal_match = 1.0
                a_postal_conflict = 0.0
            else:
                a_postal_match = 0.0
                a_postal_conflict = 1.0 # Conflicting city/ZIP!
        else:
            a_postal_match = 0.0
            a_postal_conflict = 0.0
            
        # First street number extraction
        s1_first_num = re.search(r'\b\d+\b', s1_addr)
        c_first_num = re.search(r'\b\d+\b', c_addr)
        if s1_first_num and c_first_num:
            a_first_num_match = 1.0 if s1_first_num.group(0) == c_first_num.group(0) else 0.0
        else:
            a_first_num_match = 0.5
    else:
        a_ts_ratio = 0.0
        a_sort_ratio = 0.0
        a_containment = 0.0
        num_jaccard = 0.0
        num_exact = 0.0
        a_postal_match = 0.0
        a_postal_conflict = 0.0
        a_first_num_match = 0.5
        
    is_s2 = 1.0 if c_id.startswith('S2-') else 0.0
    is_s3 = 1.0 if c_id.startswith('S3-') else 0.0
    
    return [
        n_ts_ratio, n_sort_ratio, n_ratio, n_partial, n_sq_ratio, n_exact,
        n_first_token_ratio, n_first_token_exact, n_containment, n_len_ratio, len_diff,
        a_ts_ratio, a_sort_ratio, a_containment, num_jaccard, num_exact,
        a_postal_match, a_postal_conflict, a_first_num_match, addr_empty,
        is_s2, is_s3
    ]

FEATURE_NAMES = [
    'n_ts_ratio', 'n_sort_ratio', 'n_ratio', 'n_partial', 'n_sq_ratio', 'n_exact',
    'n_first_token_ratio', 'n_first_token_exact', 'n_containment', 'n_len_ratio', 'len_diff',
    'a_ts_ratio', 'a_sort_ratio', 'a_containment', 'num_jaccard', 'num_exact',
    'a_postal_match', 'a_postal_conflict', 'a_first_num_match', 'addr_empty',
    'is_s2', 'is_s3'
]

def build_training_dataset(processed_dir, gt_path, n_samples_per_country=35000):
    print(f"=== Building Enhanced Training Dataset (Samples per country: {n_samples_per_country:,}) ===", flush=True)
    
    print("Loading Ground Truth...", flush=True)
    gt_map = {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            s1_id = parts[0]
            matched = set(parts[1].split(',')) if len(parts) > 1 and parts[1].strip() else set()
            gt_map[s1_id] = matched
            
    X = []
    y = []
    
    for country in ['us', 'india']:
        print(f"\nProcessing training data for {country.upper()}...", flush=True)
        s1_path = os.path.join(processed_dir, f"train_source1_{country}.tsv")
        s2_path = os.path.join(processed_dir, f"train_source2_{country}.tsv")
        s3_path = os.path.join(processed_dir, f"train_source3_{country}.tsv")
        
        # Load sample of S1
        s1_df = pd.read_csv(s1_path, sep='\t', usecols=['entity_id', 'name_clean', 'addr_clean'], nrows=n_samples_per_country).fillna('')
        sample_s1_ids = set(s1_df['entity_id'])
        
        target_ids_needed = set()
        for s1_id in sample_s1_ids:
            target_ids_needed.update(gt_map.get(s1_id, set()))
            
        print(f"  Sampled {len(s1_df):,} S1 entities ({len(target_ids_needed):,} true matches).", flush=True)
        
        # Load S2 and S3 targets
        target_records = {}
        inverted_index = defaultdict(list)
        for path in [s2_path, s3_path]:
            if not os.path.exists(path):
                continue
            for chunk in pd.read_csv(path, sep='\t', usecols=['entity_id', 'name_clean', 'addr_clean'], chunksize=100000):
                chunk = chunk.fillna('')
                for r in chunk.itertuples():
                    eid = r.entity_id
                    if eid in target_ids_needed or len(target_records) < 320000:
                        name, addr = r.name_clean, r.addr_clean
                        target_records[eid] = (name, addr)
                        combined = f"{name} {addr}".lower()
                        tokens = re.findall(r'\b[a-z0-9]{3,}\b', combined) + re.findall(r'\b\d{2}\b', combined)
                        for tok in set(tokens):
                            inverted_index[tok].append(eid)
                del chunk
                gc.collect()
                if len(target_records) >= 360000 and target_ids_needed.issubset(target_records.keys()):
                    break
                    
        print(f"  Indexed {len(target_records):,} target records for negative sampling.", flush=True)
        
        # Create pairs with Near-Miss Hard Negative Mining
        for r in s1_df.itertuples():
            s1_id = r.entity_id
            s1_name = r.name_clean
            s1_addr = r.addr_clean
            true_matches = gt_map.get(s1_id, set())
            
            # 1. Add Positive Pairs
            for mid in true_matches:
                if mid in target_records:
                    c_name, c_addr = target_records[mid]
                    feats = extract_features(s1_name, s1_addr, c_name, c_addr, mid)
                    X.append(feats)
                    y.append(1)
                    
            # 2. Add Hard Negatives from inverted index pool
            combined = f"{s1_name} {s1_addr}".lower()
            tokens = re.findall(r'\b[a-z0-9]{3,}\b', combined) + re.findall(r'\b\d{2}\b', combined)
            cand_counts = Counter()
            for tok in set(tokens):
                posting = inverted_index.get(tok)
                if posting and len(posting) < 3500:
                    cand_counts.update(posting)
                    
            # Specifically select hardest negatives (high token count / name overlap)
            negatives_added = 0
            for cid, count in cand_counts.most_common(40):
                if cid not in true_matches and cid in target_records:
                    c_name, c_addr = target_records[cid]
                    feats = extract_features(s1_name, s1_addr, c_name, c_addr, cid)
                    X.append(feats)
                    y.append(0)
                    negatives_added += 1
                    if negatives_added >= 4:
                        break
                        
        del target_records, inverted_index, s1_df
        gc.collect()
        
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"\nFinal Enhanced Dataset: {len(y):,} rows ({np.sum(y == 1):,} Positives, {np.sum(y == 0):,} Hard Negatives).", flush=True)
    return X, y

def train_and_evaluate(X, y, model_save_path):
    print("=== Training Enhanced LightGBM Matcher ===", flush=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=train_data)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.06,
        'num_leaves': 45,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'min_child_samples': 20,
        'verbose': -1,
        'n_jobs': 4
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=400,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
    )
    
    # Threshold Tuning for F_0.5
    print("\n--- Tuning F_0.5 Optimal Threshold ---", flush=True)
    preds = model.predict(X_val)
    
    best_thresh = 0.5
    best_f05 = 0.0
    best_prec = 0.0
    best_rec = 0.0
    
    for thresh in np.arange(0.50, 0.95, 0.01):
        pred_binary = (preds >= thresh).astype(int)
        tp = np.sum((pred_binary == 1) & (y_val == 1))
        fp = np.sum((pred_binary == 1) & (y_val == 0))
        fn = np.sum((pred_binary == 0) & (y_val == 1))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        denom = 0.25 * prec + rec
        f05 = (1.25 * prec * rec) / denom if denom > 0 else 0
        
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            best_prec = prec
            best_rec = rec
            
    print(f"Optimal Threshold: {best_thresh:.2f}")
    print(f"  Precision: {best_prec*100:.2f}%")
    print(f"  Recall:    {best_rec*100:.2f}%")
    print(f"  Pairwise F_0.5: {best_f05:.5f}")
    
    # Feature importances
    imp = model.feature_importance(importance_type='gain')
    sorted_idx = np.argsort(-imp)
    print("\nTop 8 Most Discriminative Features:")
    for i in sorted_idx[:8]:
        print(f"  - {FEATURE_NAMES[i]}: {imp[i]:.1f}")
        
    metadata = {
        'model': model,
        'threshold': float(best_thresh),
        'features': FEATURE_NAMES,
        'f05_score': float(best_f05)
    }
    with open(model_save_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"\nModel successfully saved to {model_save_path}!")
    return model, best_thresh

if __name__ == '__main__':
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..')
    processed_dir = os.path.join(base_dir, 'processed_data')
    gt_path = os.path.join(base_dir, 'dataset', 'train', 'train_ground_truth.tsv')
    model_save_path = os.path.join(os.path.dirname(__file__), 'matcher_model.pkl')
    
    X, y = build_training_dataset(processed_dir, gt_path, n_samples_per_country=35000)
    train_and_evaluate(X, y, model_save_path)
