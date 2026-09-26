import pandas as pd
import numpy as np
import re
import os

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Remove all spaces and special chars for a strict exact match
    text = re.sub(r'[^a-z0-9]', '', text) 
    return text

print("Loading test data...")
base_path = "dataset/test"

# Use dtype=str to avoid any mixed-type warnings
s1 = pd.read_csv(f"{base_path}/test_source1.tsv", sep='\t', dtype=str)
s2 = pd.read_csv(f"{base_path}/test_source2.tsv", sep='\t', dtype=str)
s3 = pd.read_csv(f"{base_path}/test_source3.tsv", sep='\t', dtype=str)

# Combine candidate pool
candidates = pd.concat([s2, s3], ignore_index=True)

print("Cleaning data...")
s1['key'] = s1['business_name'].apply(clean_text)
candidates['key'] = candidates['business_name'].apply(clean_text)

# Filter out empty keys to prevent everything matching with everything
s1_valid = s1[s1['key'] != ""]
candidates_valid = candidates[candidates['key'] != ""]

print("Merging (Exact Match)...")
# Find matching candidates by exact name match
merged = s1_valid[['entity_id', 'key']].merge(
    candidates_valid[['entity_id', 'key']], 
    on='key', 
    suffixes=('_s1', '_cand')
)

print("Aggregating...")
# Group by S1 entity and collect candidate IDs
matches = merged.groupby('entity_id_s1')['entity_id_cand'].apply(lambda x: ",".join(x.unique())).reset_index()
matches.rename(columns={'entity_id_s1': 'source1_entity_id', 'entity_id_cand': 'matched_entity_ids'}, inplace=True)

print("Formatting final output...")
# Ensure ALL S1 entities are in the final output (even if no matches)
final_results = pd.DataFrame({'source1_entity_id': s1['entity_id']})
final_results = final_results.merge(matches, on='source1_entity_id', how='left')
final_results['matched_entity_ids'] = final_results['matched_entity_ids'].fillna("")

# For this exact-match baseline, candidates = matches
final_results['candidate_entity_ids'] = final_results['matched_entity_ids'] 

print("Saving results...")
os.makedirs("output", exist_ok=True)
final_results[['source1_entity_id', 'matched_entity_ids']].to_csv("output/matching_results.tsv", sep='\t', index=False)
final_results[['source1_entity_id', 'candidate_entity_ids']].to_csv("output/candidate_pairs.tsv", sep='\t', index=False)

print("Done! Validating...")
os.system("python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test")
