import os
import sys
import gc
import re
import time
import pickle
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

def get_tokens(name_clean, addr_clean):
    combined = f"{name_clean} {addr_clean}".lower()
    tokens = re.findall(r'\b[a-z0-9]{3,}\b', combined)
    tokens += re.findall(r'\b\d{2}\b', combined)
    return set(tokens)

def squash_name(name):
    s = str(name).lower().replace('@', '').replace(' ', '').replace('&', '').replace('and', '')
    for suf in ['.com', '.org', '.net', '.co', 'com', 'org', 'net', 'www', 'llc', 'inc', 'corp', 'ltd', 'lp', 'pllc']:
        s = s.replace(suf, '')
    return re.sub(r'[^a-z0-9]', '', s)

def extract_features(s1_name, s1_addr, c_name, c_addr, c_id):
    s1_sq = squash_name(s1_name)
    c_sq = squash_name(c_name)
    
    s1_words = [w for w in s1_name.split() if w]
    c_words = [w for w in c_name.split() if w]
    s1_tok_set = set(s1_words)
    c_tok_set = set(c_words)
    
    n_ts_ratio = fuzz.token_set_ratio(s1_name, c_name)
    n_sort_ratio = fuzz.token_sort_ratio(s1_name, c_name)
    n_ratio = fuzz.ratio(s1_name, c_name)
    n_partial = fuzz.partial_ratio(s1_name, c_name)
    n_sq_ratio = fuzz.ratio(s1_sq, c_sq)
    n_exact = 1.0 if s1_name and s1_name == c_name else 0.0
    
    if s1_words and c_words:
        n_first_token_ratio = fuzz.ratio(s1_words[0], c_words[0])
        n_first_token_exact = 1.0 if s1_words[0] == c_words[0] else 0.0
    else:
        n_first_token_ratio = 0.0
        n_first_token_exact = 0.0
        
    min_tok_len = min(len(s1_tok_set), len(c_tok_set))
    n_containment = len(s1_tok_set.intersection(c_tok_set)) / min_tok_len if min_tok_len > 0 else 0.0
    
    max_len = max(len(s1_name), len(c_name))
    min_len = min(len(s1_name), len(c_name))
    n_len_ratio = (min_len / max_len) if max_len > 0 else 1.0
    len_diff = abs(len(s1_name) - len(c_name))
    
    addr_empty = 1.0 if not s1_addr or not c_addr else 0.0
    if not addr_empty:
        a_ts_ratio = fuzz.token_set_ratio(s1_addr, c_addr)
        a_sort_ratio = fuzz.token_sort_ratio(s1_addr, c_addr)
        
        s1_addr_words = set(s1_addr.split())
        c_addr_words = set(c_addr.split())
        min_addr_tok = min(len(s1_addr_words), len(c_addr_words))
        a_containment = len(s1_addr_words.intersection(c_addr_words)) / min_addr_tok if min_addr_tok > 0 else 0.0
        
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
            
        s1_postals = set(re.findall(r'\b\d{5,6}\b', s1_addr))
        c_postals = set(re.findall(r'\b\d{5,6}\b', c_addr))
        if s1_postals and c_postals:
            if s1_postals.intersection(c_postals):
                a_postal_match = 1.0
                a_postal_conflict = 0.0
            else:
                a_postal_match = 0.0
                a_postal_conflict = 1.0
        else:
            a_postal_match = 0.0
            a_postal_conflict = 0.0
            
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

def load_targets(s2_path, s3_path):
    target_records = {}
    inverted_index = defaultdict(list)
    squash_index = defaultdict(list)
    
    for path in [s2_path, s3_path]:
        if not os.path.exists(path):
            continue
        print(f"  Loading targets from {os.path.basename(path)}...", flush=True)
        for chunk in pd.read_csv(path, sep='\t', usecols=['entity_id', 'name_clean', 'addr_clean'], chunksize=100000, dtype=str):
            chunk = chunk.fillna('')
            for r in chunk.itertuples():
                eid = r.entity_id
                name, addr = r.name_clean, r.addr_clean
                sq = squash_name(name)
                target_records[eid] = (name, addr, sq)
                for tok in get_tokens(name, addr):
                    inverted_index[tok].append(eid)
                if len(sq) >= 5:
                    squash_index[sq].append(eid)
            del chunk
            gc.collect()
            
    print(f"  Indexed {len(target_records):,} target entities into {len(inverted_index):,} tokens and {len(squash_index):,} squashed keys.", flush=True)
    return target_records, inverted_index, squash_index

def run_pipeline(model_path, processed_dir, output_dir, countries=['france', 'us', 'india']):
    print("=== Loading Matcher Model ===", flush=True)
    with open(model_path, 'rb') as f:
        meta = pickle.load(f)
    model = meta['model']
    threshold = meta['threshold']
    print(f"Loaded LightGBM model with tuned F_0.5 threshold: {threshold:.2f}", flush=True)
    
    os.makedirs(output_dir, exist_ok=True)
    matching_path = os.path.join(output_dir, 'matching_results.tsv')
    candidate_path = os.path.join(output_dir, 'candidate_pairs.tsv')
    
    # Open both output files for streaming
    with open(matching_path, 'w', encoding='utf-8') as f_match, open(candidate_path, 'w', encoding='utf-8') as f_cand:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        
        for country in countries:
            t0 = time.time()
            print(f"\n=======================================================", flush=True)
            print(f"Processing Inference: Country = {country.upper()}", flush=True)
            print(f"=======================================================", flush=True)
            
            s1_path = os.path.join(processed_dir, f"test_source1_{country}.tsv")
            s2_path = os.path.join(processed_dir, f"test_source2_{country}.tsv")
            s3_path = os.path.join(processed_dir, f"test_source3_{country}.tsv")
            
            if not os.path.exists(s1_path):
                print(f"Test source1 missing: {s1_path}, skipping.")
                continue
                
            target_records, inverted_index, squash_index = load_targets(s2_path, s3_path)
            
            print(f"  Streaming S1 queries and predicting matches...", flush=True)
            s1_count = 0
            t_start = time.time()
            max_posting_len = 4000
            top_k_s2, top_k_s3 = 15, 15
            
            for s1_chunk in pd.read_csv(s1_path, sep='\t', usecols=['entity_id', 'name_clean', 'addr_clean'], chunksize=25000, dtype=str):
                s1_chunk = s1_chunk.fillna('')
                
                s1_candidates = [] # list of (s1_id, [cids])
                batch_features = []
                batch_pair_indices = [] # (s1_index_in_chunk, cid)
                
                for idx, r in enumerate(s1_chunk.itertuples()):
                    s1_id = r.entity_id
                    s1_name, s1_addr = r.name_clean, r.addr_clean
                    s1_sq = squash_name(s1_name)
                    s1_tokens = get_tokens(s1_name, s1_addr)
                    
                    # 1. Candidate blocking (Upgrade 1: Multi-Key Inverted Index + Squashed Domain Hash)
                    cand_counts = Counter()
                    for tok in s1_tokens:
                        posting = inverted_index.get(tok)
                        if posting and len(posting) < max_posting_len:
                            cand_counts.update(posting)
                            
                    top_pool = [cid for cid, cnt in cand_counts.most_common(100)]
                    if len(s1_sq) >= 5 and s1_sq in squash_index:
                        top_pool.extend(squash_index[s1_sq])
                    top_pool = list(dict.fromkeys(top_pool))
                    
                    if not top_pool:
                        s1_candidates.append((s1_id, []))
                        continue
                        
                    # 2. Fast candidate pre-ranking (Upgrade 2: Dual-Channel Scoring + Upgrade 3: Postal Conflict Guard)
                    s2_cands = []
                    s3_cands = []
                    s1_postal = re.search(r'\b\d{5,6}\b', s1_addr)
                    s1_post_code = s1_postal.group(0) if s1_postal else None
                    s1_has_num = bool(re.search(r'\b\d+\b', s1_addr))
                    
                    for cid in top_pool:
                        c_name, c_addr, c_sq = target_records[cid]
                        
                        # Upgrade 3: Postal conflict filter
                        if s1_post_code and c_addr:
                            c_postal = re.search(r'\b\d{5,6}\b', c_addr)
                            if c_postal and c_postal.group(0) != s1_post_code:
                                name_quick = fuzz.token_set_ratio(s1_name, c_name)
                                if name_quick < 92.0:
                                    continue
                                    
                        name_score = fuzz.token_set_ratio(s1_name, c_name)
                        sq_score = 0.0
                        if s1_sq and c_sq and abs(len(s1_sq) - len(c_sq)) <= 5:
                            sq_score = float(fuzz.ratio(s1_sq, c_sq))
                            if sq_score > name_score:
                                name_score = sq_score
                                
                        addr_score = 0.0
                        if c_addr and s1_addr:
                            addr_score = float(fuzz.token_set_ratio(s1_addr, c_addr))
                            if s1_has_num:
                                s1_num = re.search(r'\b\d+\b', s1_addr).group(0)
                                c_num = re.search(r'\b\d+\b', c_addr)
                                if c_num and c_num.group(0) == s1_num:
                                    addr_score = min(100.0, addr_score + 10.0)
                                    
                        # Upgrade 2: Dual-channel composite
                        if c_addr and s1_addr:
                            comp_score = name_score * 0.70 + addr_score * 0.30
                            score = max(comp_score, addr_score if addr_score >= 85.0 else 0.0, sq_score)
                        else:
                            score = name_score * 0.85
                            
                        if cid.startswith('S2-'):
                            s2_cands.append((score, cid, name_score, addr_score))
                        else:
                            s3_cands.append((score, cid, name_score, addr_score))
                            
                    s2_cands.sort(key=lambda x: x[0], reverse=True)
                    s3_cands.sort(key=lambda x: x[0], reverse=True)
                    
                    selected_cands = [cid for sc, cid, n_sc, a_sc in s2_cands[:top_k_s2] if sc >= 35.0 or a_sc >= 85.0]
                    selected_cands.extend([cid for sc, cid, n_sc, a_sc in s3_cands[:top_k_s3] if sc >= 35.0 or a_sc >= 85.0])
                    selected_cands = list(dict.fromkeys(selected_cands))
                    
                    s1_candidates.append((s1_id, selected_cands))
                    
                    # Accumulate features for batch scoring
                    for cid in selected_cands:
                        c_name, c_addr, _ = target_records[cid]
                        batch_features.append(extract_features(s1_name, s1_addr, c_name, c_addr, cid))
                        batch_pair_indices.append((idx, cid))
                
                # 3. Vectorized Chunk-Level Model Prediction (Upgrade 4: Precision Shield & Singleton Guard)
                matched_dict = defaultdict(list)
                if batch_features:
                    probs = model.predict(np.array(batch_features, dtype=np.float32))
                    s1_names = s1_chunk['name_clean'].values
                    s1_addrs = s1_chunk['addr_clean'].values
                    for (s1_idx, cid), prob in zip(batch_pair_indices, probs):
                        c_name, c_addr, _ = target_records[cid]
                        s1_n, s1_a = s1_names[s1_idx], s1_addrs[s1_idx]
                        
                        # Upgrade 4: Singleton Guard on empty candidate address
                        if not c_addr and s1_a:
                            name_sim = fuzz.token_set_ratio(s1_n, c_name)
                            if prob >= 0.90 and name_sim >= 85.0:
                                matched_dict[s1_idx].append(cid)
                        else:
                            if prob >= threshold:
                                matched_dict[s1_idx].append(cid)
                            
                # 4. Stream to disk
                match_lines = []
                cand_lines = []
                for idx, (s1_id, cands) in enumerate(s1_candidates):
                    cand_str = ",".join(cands)
                    cand_lines.append(f"{s1_id}\t{cand_str}\n")
                    
                    m_list = matched_dict.get(idx, [])
                    # Deduplicate preserving order
                    m_list = list(dict.fromkeys(m_list))
                    match_str = ",".join(m_list)
                    match_lines.append(f"{s1_id}\t{match_str}\n")
                    s1_count += 1
                    
                f_cand.writelines(cand_lines)
                f_match.writelines(match_lines)
                f_cand.flush()
                f_match.flush()
                
                elapsed = time.time() - t_start
                rate = s1_count / elapsed if elapsed > 0 else 0
                print(f"    Inference for {s1_count:,} S1 entities ({rate:.0f} entities/sec)...", flush=True)
                
                del s1_chunk, s1_candidates, batch_features, batch_pair_indices, matched_dict, cand_lines, match_lines
                gc.collect()
                
            print(f"Completed {country.upper()} in {time.time()-t0:.1f}s!", flush=True)
            del target_records, inverted_index, squash_index
            gc.collect()
            
    print(f"\n=======================================================", flush=True)
    print(f"Inference Completed! Files generated:")
    print(f"  1. {matching_path}")
    print(f"  2. {candidate_path}")
    print(f"=======================================================", flush=True)

if __name__ == '__main__':
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..')
    processed_dir = os.path.join(base_dir, 'processed_data')
    output_dir = os.path.join(base_dir, 'output')
    model_path = os.path.join(os.path.dirname(__file__), 'matcher_model.pkl')
    
    run_pipeline(model_path, processed_dir, output_dir)
