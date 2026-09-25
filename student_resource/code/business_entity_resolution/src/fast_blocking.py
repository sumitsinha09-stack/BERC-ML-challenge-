import os
import sys
import gc
import re
import time
from collections import defaultdict, Counter
import pandas as pd
from rapidfuzz import fuzz

def get_tokens(name_clean, addr_clean):
    """
    Extract alphanumeric tokens:
    - words and alphanumeric strings of length >= 3
    - 2-digit street/house numbers
    """
    combined = f"{name_clean} {addr_clean}".lower()
    tokens = re.findall(r'\b[a-z0-9]{3,}\b', combined)
    tokens += re.findall(r'\b\d{2}\b', combined)
    return set(tokens)

def squash_name(name):
    s = str(name).lower().replace('@', '').replace(' ', '').replace('&', '').replace('and', '')
    for suf in ['.com', '.org', '.net', '.co', 'com', 'org', 'net', 'www', 'llc', 'inc', 'corp', 'ltd', 'lp', 'pllc']:
        s = s.replace(suf, '')
    return re.sub(r'[^a-z0-9]', '', s)

def load_targets(s2_path, s3_path):
    target_records = {}
    inverted_index = defaultdict(list)
    squash_index = defaultdict(list)
    
    for path, prefix in [(s2_path, 'S2'), (s3_path, 'S3')]:
        if not os.path.exists(path):
            continue
        print(f"  Loading targets from {os.path.basename(path)}...", flush=True)
        for chunk in pd.read_csv(path, sep='\t', usecols=['entity_id', 'name_clean', 'addr_clean'], chunksize=100000, dtype=str):
            chunk = chunk.fillna('')
            for r in chunk.itertuples():
                eid = r.entity_id
                name = r.name_clean
                addr = r.addr_clean
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

def run_blocking_for_country(processed_dir, split, country, out_file_handle, top_k_s2=15, top_k_s3=15, pool_size=100, max_posting_len=4000):
    t0 = time.time()
    print(f"\n=======================================================", flush=True)
    print(f"Starting Fast Blocking: [{split.upper()}] - {country.upper()}", flush=True)
    print(f"=======================================================", flush=True)
    
    s1_path = os.path.join(processed_dir, f"{split}_source1_{country}.tsv")
    s2_path = os.path.join(processed_dir, f"{split}_source2_{country}.tsv")
    s3_path = os.path.join(processed_dir, f"{split}_source3_{country}.tsv")
    
    if not os.path.exists(s1_path):
        print(f"Source 1 file missing: {s1_path}, skipping.")
        return
        
    target_records, inverted_index, squash_index = load_targets(s2_path, s3_path)
    
    print(f"  Processing S1 queries from {os.path.basename(s1_path)}...", flush=True)
    s1_count = 0
    t_start_queries = time.time()
    
    for s1_chunk in pd.read_csv(s1_path, sep='\t', usecols=['entity_id', 'name_clean', 'addr_clean'], chunksize=25000, dtype=str):
        s1_chunk = s1_chunk.fillna('')
        
        lines_to_write = []
        for r in s1_chunk.itertuples():
            s1_id = r.entity_id
            s1_name = r.name_clean
            s1_addr = r.addr_clean
            s1_sq = squash_name(s1_name)
            s1_tokens = get_tokens(s1_name, s1_addr)
            
            # Step 1: Count candidate occurrences across shared tokens + Squashed Hash
            cand_counts = Counter()
            for tok in s1_tokens:
                posting = inverted_index.get(tok)
                if posting and len(posting) < max_posting_len:
                    cand_counts.update(posting)
            
            top_pool = [cid for cid, count in cand_counts.most_common(pool_size)]
            if len(s1_sq) >= 5 and s1_sq in squash_index:
                top_pool.extend(squash_index[s1_sq])
            top_pool = list(dict.fromkeys(top_pool))
            
            if not top_pool:
                lines_to_write.append(f"{s1_id}\t\n")
                s1_count += 1
                continue
            
            # Step 2: Rank top pool using RapidFuzz + Dual-Channel + Postal Guard
            s2_cands = []
            s3_cands = []
            s1_postal = re.search(r'\b\d{5,6}\b', s1_addr)
            s1_post_code = s1_postal.group(0) if s1_postal else None
            s1_has_num = bool(re.search(r'\b\d+\b', s1_addr))
            
            for cid in top_pool:
                c_name, c_addr, c_sq = target_records[cid]
                
                # Postal conflict guard
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
                            
                if c_addr and s1_addr:
                    comp_score = name_score * 0.70 + addr_score * 0.30
                    score = max(comp_score, addr_score if addr_score >= 85.0 else 0.0, sq_score)
                else:
                    score = name_score * 0.85
                    
                if cid.startswith('S2-'):
                    s2_cands.append((score, cid, addr_score))
                else:
                    s3_cands.append((score, cid, addr_score))
                    
            s2_cands.sort(key=lambda x: x[0], reverse=True)
            s3_cands.sort(key=lambda x: x[0], reverse=True)
            
            selected_cands = [cid for score, cid, a_sc in s2_cands[:top_k_s2] if score >= 35.0 or a_sc >= 85.0]
            selected_cands.extend([cid for score, cid, a_sc in s3_cands[:top_k_s3] if score >= 35.0 or a_sc >= 85.0])
            selected_cands = list(dict.fromkeys(selected_cands))
            
            cand_str = ",".join(selected_cands)
            lines_to_write.append(f"{s1_id}\t{cand_str}\n")
            s1_count += 1
            
        out_file_handle.writelines(lines_to_write)
        elapsed = time.time() - t_start_queries
        rate = s1_count / elapsed if elapsed > 0 else 0
        print(f"    Queried {s1_count:,} S1 entities ({rate:.0f} entities/sec)...", flush=True)
        
        del s1_chunk, lines_to_write
        gc.collect()
        
    print(f"Done {country.upper()} in {time.time()-t0:.1f}s! Total S1 processed: {s1_count:,}", flush=True)
    del target_records, inverted_index
    gc.collect()

def generate_candidates(split='test'):
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..')
    processed_dir = os.path.join(base_dir, 'processed_data')
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    if split == 'test':
        out_path = os.path.join(output_dir, 'candidate_pairs.tsv')
        countries = ['france', 'us', 'india']
    else:
        out_path = os.path.join(processed_dir, 'train_candidate_pairs.tsv')
        countries = ['us', 'india']
        
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for country in countries:
            run_blocking_for_country(processed_dir, split, country, out_file_handle=f)
            
    print(f"\n=======================================================", flush=True)
    print(f"Candidates successfully written to: {out_path}", flush=True)
    print(f"=======================================================", flush=True)

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'test'
    generate_candidates(mode)
