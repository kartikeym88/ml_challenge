import pandas as pd
import numpy as np
import re
import os
from tqdm import tqdm

def clean_name(text):
    if pd.isna(text): return ""
    return re.sub(r'[^a-z0-9]', '', str(text).lower())

def clean_address(text):
    if pd.isna(text): return ""
    return re.sub(r'[^a-z0-9\s]', ' ', str(text).lower())

def get_trigrams(text):
    text = re.sub(r'\s+', '', text)
    if len(text) < 3: return set([text])
    return set([text[i:i+3] for i in range(len(text)-2)])

def jaccard_sim(set1, set2):
    if not set1 or not set2: return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

print("Loading test data...")
base_path = "dataset/test"

s1 = pd.read_csv(f"{base_path}/test_source1.tsv", sep='\t', dtype=str)
s2 = pd.read_csv(f"{base_path}/test_source2.tsv", sep='\t', dtype=str)
s3 = pd.read_csv(f"{base_path}/test_source3.tsv", sep='\t', dtype=str)

candidates = pd.concat([s2, s3], ignore_index=True)

print("Cleaning data...")
s1['name_key'] = s1['business_name'].apply(clean_name)
candidates['name_key'] = candidates['business_name'].apply(clean_name)

# CRITICAL FIX: Block on Country AND exact Name
s1['block_key'] = s1['country'].astype(str).str.lower() + "_" + s1['name_key']
candidates['block_key'] = candidates['country'].astype(str).str.lower() + "_" + candidates['name_key']

s1_valid = s1[s1['name_key'] != ""]
candidates_valid = candidates[candidates['name_key'] != ""]

print("Blocking (Exact Match on Country + Name)...")
merged = s1_valid[['entity_id', 'block_key', 'business_address']].merge(
    candidates_valid[['entity_id', 'block_key', 'business_address']], 
    on='block_key', 
    suffixes=('_s1', '_cand')
)

print(f"Generated {len(merged)} candidate pairs. Computing Address Similarity...")

def calc_sim(row):
    addr1 = clean_address(row['business_address_s1'])
    addr2 = clean_address(row['business_address_cand'])
    tri1 = get_trigrams(addr1)
    tri2 = get_trigrams(addr2)
    return jaccard_sim(tri1, tri2)

tqdm.pandas()
merged['addr_sim'] = merged.progress_apply(calc_sim, axis=1)

# Tune this threshold if needed!
ADDRESS_THRESHOLD = 0.40 
print(f"Applying Threshold (> {ADDRESS_THRESHOLD})...")
final_matches = merged[merged['addr_sim'] > ADDRESS_THRESHOLD]

print("Aggregating outputs...")
cand_agg = merged.groupby('entity_id_s1')['entity_id_cand'].apply(lambda x: ",".join(x.unique())).reset_index()
cand_agg.rename(columns={'entity_id_s1': 'source1_entity_id', 'entity_id_cand': 'candidate_entity_ids'}, inplace=True)

match_agg = final_matches.groupby('entity_id_s1')['entity_id_cand'].apply(lambda x: ",".join(x.unique())).reset_index()
match_agg.rename(columns={'entity_id_s1': 'source1_entity_id', 'entity_id_cand': 'matched_entity_ids'}, inplace=True)

print("Formatting...")
final_results = pd.DataFrame({'source1_entity_id': s1['entity_id']})
final_cands = final_results.merge(cand_agg, on='source1_entity_id', how='left')
final_cands['candidate_entity_ids'] = final_cands['candidate_entity_ids'].fillna("")

final_matches_df = final_results.merge(match_agg, on='source1_entity_id', how='left')
final_matches_df['matched_entity_ids'] = final_matches_df['matched_entity_ids'].fillna("")

print("Saving results...")
os.makedirs("output", exist_ok=True)
final_matches_df.to_csv("output/matching_results.tsv", sep='\t', index=False)
final_cands.to_csv("output/candidate_pairs.tsv", sep='\t', index=False)

print("Done! Validating...")
os.system("python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test")
