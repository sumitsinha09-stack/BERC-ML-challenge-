import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
import os
import gc
import sys
import time
import resource


def get_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def transform_file_in_chunks(filepath, hv, chunksize=50000):
    """
    Stream TSV file in 50,000-row chunks to build a sparse Hashing TF-IDF matrix.
    Memory footprint remains low (< 500 MB per chunk).
    """
    vec_chunks = []
    ids_chunks = []

    print(f"  Loading & Vectorizing {os.path.basename(filepath)}...", flush=True)
    for chunk in pd.read_csv(filepath, sep="\t", usecols=["entity_id", "name_clean", "addr_clean"], chunksize=chunksize):
        chunk = chunk.fillna("")
        text_series = (chunk["name_clean"] + " " + chunk["addr_clean"]).values
        ids_chunks.append(chunk["entity_id"].values)

        v = hv.transform(text_series)
        vec_chunks.append(v)

        del chunk, text_series
        gc.collect()

    all_vecs = sp.vstack(vec_chunks, format="csr")
    all_ids = np.concatenate(ids_chunks)

    del vec_chunks, ids_chunks
    gc.collect()

    return all_vecs, all_ids


def run_blocking_for_country(processed_dir, prefix, country, out_file_handle, top_k_s2=15, top_k_s3=15, batch_size=2000, min_sim=0.08):
    print(f"\n=== Low-RAM Streaming Blocking for {prefix.upper()} - Country: {country.upper()} ===", flush=True)

    s1_path = os.path.join(processed_dir, f"{prefix}_source1_{country}.tsv")
    s2_path = os.path.join(processed_dir, f"{prefix}_source2_{country}.tsv")
    s3_path = os.path.join(processed_dir, f"{prefix}_source3_{country}.tsv")

    hv = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 4), n_features=2**16, dtype=np.float32, norm="l2")

    t0 = time.time()
    s1_vecs, s1_ids = transform_file_in_chunks(s1_path, hv)
    s2_vecs, s2_ids = transform_file_in_chunks(s2_path, hv)
    s3_vecs, s3_ids = transform_file_in_chunks(s3_path, hv)

    n_queries = s1_vecs.shape[0]
    print(f"  Loaded: S1={n_queries:,}, S2={len(s2_ids):,}, S3={len(s3_ids):,} (RAM: {get_mem_mb():.1f} MB)", flush=True)

    t1 = time.time()
    print("  Matching queries in batches and streaming to disk...", flush=True)

    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        q_sub = s1_vecs[start:end]

        # Dot product with S2
        sim_s2 = q_sub.dot(s2_vecs.T)
        # Dot product with S3
        sim_s3 = q_sub.dot(s3_vecs.T)

        for i in range(end - start):
            s1_id = s1_ids[start + i]
            cand_ids = []

            # Extract top S2 matches
            r2 = sim_s2.getrow(i)
            if r2.nnz > 0:
                d2, idx2 = r2.data, r2.indices
                m2 = d2 >= min_sim
                if np.any(m2):
                    d2, idx2 = d2[m2], idx2[m2]
                    if len(d2) > top_k_s2:
                        top_p = np.argpartition(d2, -top_k_s2)[-top_k_s2:]
                        cand_ids.extend(s2_ids[idx2[top_p[np.argsort(-d2[top_p])]]])
                    else:
                        cand_ids.extend(s2_ids[idx2[np.argsort(-d2)]])

            # Extract top S3 matches
            r3 = sim_s3.getrow(i)
            if r3.nnz > 0:
                d3, idx3 = r3.data, r3.indices
                m3 = d3 >= min_sim
                if np.any(m3):
                    d3, idx3 = d3[m3], idx3[m3]
                    if len(d3) > top_k_s3:
                        top_p = np.argpartition(d3, -top_k_s3)[-top_k_s3:]
                        cand_ids.extend(s3_ids[idx3[top_p[np.argsort(-d3[top_p])]]])
                    else:
                        cand_ids.extend(s3_ids[idx3[np.argsort(-d3)]])

            cand_str = ",".join(cand_ids) if cand_ids else ""
            out_file_handle.write(f"{s1_id}\t{cand_str}\n")

        del sim_s2, sim_s3
        if (start // batch_size) % 10 == 0:
            gc.collect()
            print(f"    Processed query {start}/{n_queries} (RAM: {get_mem_mb():.1f} MB)...", flush=True)

    print(f"  Matching finished for {country.upper()} in {time.time()-t1:.1f}s!", flush=True)

    del s1_vecs, s1_ids, s2_vecs, s2_ids, s3_vecs, s3_ids
    gc.collect()


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    processed_dir = os.path.join(base_dir, "processed_data")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    mode = sys.argv[1] if len(sys.argv) > 1 else "test"

    if mode == "test":
        print("=== Generating Candidate Pairs for TEST set (Low-RAM Streaming) ===", flush=True)
        cand_path = os.path.join(output_dir, "candidate_pairs.tsv")

        with open(cand_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for country in ["france", "us", "india"]:
                run_blocking_for_country(processed_dir, "test", country, out_file_handle=f)

        print(f"\nTest candidate pairs saved successfully to {cand_path}!", flush=True)

    elif mode == "train":
        print("=== Generating Candidate Pairs for TRAIN set (Low-RAM Streaming) ===", flush=True)
        train_cand_path = os.path.join(processed_dir, "train_candidate_pairs.tsv")

        with open(train_cand_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for country in ["us", "india"]:
                run_blocking_for_country(processed_dir, "train", country, out_file_handle=f)

        print(f"\nTrain candidate pairs saved successfully to {train_cand_path}!", flush=True)
