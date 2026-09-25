import os
import sys
import gc
import time
from collections import defaultdict
import pandas as pd
from rapidfuzz import fuzz

def resolve_country_collisions(processed_dir, country, s1_to_matches):
    print(f"\nResolving Target Collisions for {country.upper()}...", flush=True)
    t0 = time.time()
    
    # Load S1 records for country
    s1_p = os.path.join(processed_dir, f"test_source1_{country}.tsv")
    s1_records = {}
    for chunk in pd.read_csv(s1_p, sep='\t', usecols=['entity_id', 'name_clean', 'addr_clean']).fillna('').itertuples():
        s1_records[chunk.entity_id] = (chunk.name_clean, chunk.addr_clean)
        
    # Find collisions among these S1 entities
    target_to_s1 = defaultdict(list)
    for s1_id in s1_records:
        for tid in s1_to_matches.get(s1_id, []):
            target_to_s1[tid].append(s1_id)
            
    collisions = {tid: s1s for tid, s1s in target_to_s1.items() if len(s1s) > 1}
    print(f"  {country.upper()}: {len(collisions):,} targets have collision across multiple S1 entities.", flush=True)
    
    if not collisions:
        return {}
        
    # Load Targets only for colliding IDs
    collision_set = set(collisions.keys())
    target_records = {}
    for s in [2, 3]:
        p = os.path.join(processed_dir, f"test_source{s}_{country}.tsv")
        if not os.path.exists(p):
            continue
        for chunk in pd.read_csv(p, sep='\t', usecols=['entity_id', 'name_clean', 'addr_clean']).fillna('').itertuples():
            if chunk.entity_id in collision_set:
                target_records[chunk.entity_id] = (chunk.name_clean, chunk.addr_clean)
                
    # Select winning S1 for each collision target
    winner_map = {} # tid -> winning s1_id
    for tid, s1s in collisions.items():
        if tid not in target_records:
            winner_map[tid] = s1s[0] # fallback
            continue
            
        t_name, t_addr = target_records[tid]
        best_s1 = s1s[0]
        best_score = -1.0
        
        for s1_id in s1s:
            s1_n, s1_a = s1_records.get(s1_id, ('', ''))
            n_sc = fuzz.token_set_ratio(s1_n, t_name)
            a_sc = fuzz.token_set_ratio(s1_a, t_addr) if t_addr and s1_a else 0
            sc = n_sc * 0.70 + a_sc * 0.30 if t_addr and s1_a else n_sc * 0.85
            if sc > best_score:
                best_score = sc
                best_s1 = s1_id
                
        winner_map[tid] = best_s1
        
    print(f"  {country.upper()}: Disambiguated all {len(winner_map):,} collisions in {time.time()-t0:.1f}s.", flush=True)
    del s1_records, target_records, collisions, target_to_s1
    gc.collect()
    return winner_map

def run_global_disambiguation(matching_path, processed_dir, output_path):
    print("=== Global Disjoint Assignment / Target Exclusivity Disambiguation ===", flush=True)
    t_start = time.time()
    
    print(f"Reading current matching results from {matching_path}...", flush=True)
    s1_order = []
    s1_to_matches = {}
    
    with open(matching_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            s1_id = parts[0]
            matches = [m.strip() for m in parts[1].split(',') if m.strip()] if len(parts) > 1 and parts[1].strip() else []
            s1_order.append(s1_id)
            s1_to_matches[s1_id] = matches
            
    print(f"Loaded {len(s1_order):,} S1 entity predictions.", flush=True)
    
    # Resolve country by country
    all_winners = {}
    for country in ['france', 'us', 'india']:
        winners = resolve_country_collisions(processed_dir, country, s1_to_matches)
        all_winners.update(winners)
        
    print(f"\nApplying exclusive target assignments across all 1.73M entities...", flush=True)
    eliminated_fp = 0
    with open(output_path, 'w', encoding='utf-8') as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        lines = []
        for i, s1_id in enumerate(s1_order):
            raw_matches = s1_to_matches.get(s1_id, [])
            clean_matches = []
            for tid in raw_matches:
                # If target had a collision, keep only if s1_id was the winner
                if tid in all_winners:
                    if all_winners[tid] == s1_id:
                        clean_matches.append(tid)
                    else:
                        eliminated_fp += 1
                else:
                    clean_matches.append(tid)
                    
            clean_str = ",".join(clean_matches)
            lines.append(f"{s1_id}\t{clean_str}\n")
            if len(lines) >= 50000:
                f_out.writelines(lines)
                lines = []
        if lines:
            f_out.writelines(lines)
            
    print(f"SUCCESS! Eliminated {eliminated_fp:,} duplicate false-positive assignments.")
    print(f"Cleaned matching results saved to {output_path} in {time.time()-t_start:.1f}s.")

if __name__ == '__main__':
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..')
    processed_dir = os.path.join(base_dir, 'processed_data')
    matching_path = os.path.join(base_dir, 'output', 'matching_results.tsv')
    temp_output_path = os.path.join(base_dir, 'output', 'matching_results_disambiguated.tsv')
    
    run_global_disambiguation(matching_path, processed_dir, temp_output_path)
    
    # Atomically replace matching_results.tsv
    os.replace(temp_output_path, matching_path)
    print("Replaced output/matching_results.tsv with disambiguated results!")
