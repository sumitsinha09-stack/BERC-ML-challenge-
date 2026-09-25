import pandas as pd
import numpy as np
import os
import csv
from sklearn.feature_extraction.text import TfidfVectorizer

def eval_blocking_recall(country='india', top_k=25):
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..')
    processed_dir = os.path.join(base_dir, 'processed_data')
    gt_path = os.path.join(base_dir, 'dataset', 'train', 'train_ground_truth.tsv')

    print(f"--- Evaluating Blocking Recall for Country: {country} (top_k={top_k}) ---")

    s1_df = pd.read_csv(os.path.join(processed_dir, f'train_source1_{country}.tsv'), sep='\t', dtype=str).fillna('')
    s2_df = pd.read_csv(os.path.join(processed_dir, f'train_source2_{country}.tsv'), sep='\t', dtype=str).fillna('')
    s3_df = pd.read_csv(os.path.join(processed_dir, f'train_source3_{country}.tsv'), sep='\t', dtype=str).fillna('')

    s1_ids_set = set(s1_df['entity_id'])

    # Load ground truth for this country's S1 entities
    gt = {}
    total_true_matches = 0
    with open(gt_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if len(row) >= 2:
                s1_id = row[0]
                if s1_id in s1_ids_set:
                    matched = set(row[1].split(',')) if row[1].strip() else set()
                    gt[s1_id] = matched
                    total_true_matches += len(matched)

    print(f"Total S1 entities in split: {len(s1_df)}, Total True Matches: {total_true_matches}")

    # Combine name + address for richer TF-IDF blocking
    s1_df['text'] = s1_df['name_clean'] + " " + s1_df['addr_clean']
    s2_df['text'] = s2_df['name_clean'] + " " + s2_df['addr_clean']
    s3_df['text'] = s3_df['name_clean'] + " " + s3_df['addr_clean']

    s23_df = pd.concat([s2_df, s3_df], ignore_index=True)
    s23_ids = s23_df['entity_id'].values

    print("Fitting TF-IDF (char_wb 3-5)...")
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3, sublinear_tf=True)
    all_texts = pd.concat([s1_df['text'], s23_df['text']], ignore_index=True)
    vec.fit(all_texts)

    s1_vecs = vec.transform(s1_df['text'])
    s23_vecs = vec.transform(s23_df['text'])

    print("Computing cosine similarity batches...")
    found_matches = 0
    batch_size = 20000

    for start in range(0, s1_vecs.shape[0], batch_size):
        end = min(start + batch_size, s1_vecs.shape[0])
        sim = s1_vecs[start:end].dot(s23_vecs.T)

        for i in range(end - start):
            s1_id = s1_df['entity_id'].iloc[start + i]
            true_set = gt.get(s1_id, set())
            if not true_set:
                continue

            row = sim.getrow(i)
            if row.nnz == 0:
                continue

            data = row.data
            indices = row.indices

            if len(data) > top_k:
                top_k_idx = indices[np.argpartition(data, -top_k)[-top_k:]]
            else:
                top_k_idx = indices

            cand_set = set(s23_ids[top_k_idx])
            found_matches += len(true_set.intersection(cand_set))

    recall = found_matches / total_true_matches if total_true_matches > 0 else 0
    print(f"RESULT for {country}: Found {found_matches} / {total_true_matches} matches (Recall = {recall:.4f})")
    return recall

if __name__ == "__main__":
    eval_blocking_recall(country='india', top_k=25)
