import pandas as pd
import numpy as np
import re

def clean_name(text):
    if pd.isna(text): return ""
    return re.sub(r'[^a-z0-9]', '', str(text).lower())

def clean_address(text):
    if pd.isna(text): return ""
    # Keep spaces for tokenization
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

print("Loading train data...")
base_path = "dataset/train"
s1 = pd.read_csv(f"{base_path}/train_source1.tsv", sep='\t', dtype=str)
s2 = pd.read_csv(f"{base_path}/train_source2.tsv", sep='\t', dtype=str)
s3 = pd.read_csv(f"{base_path}/train_source3.tsv", sep='\t', dtype=str)
gt = pd.read_csv(f"{base_path}/train_ground_truth.tsv", sep='\t', dtype=str)

# Only evaluate on a subset to save time (e.g., first 10,000 S1 records)
s1 = s1.head(10000)

cands = pd.concat([s2, s3], ignore_index=True)

print("Cleaning keys...")
s1['name_key'] = s1['business_name'].apply(clean_name)
cands['name_key'] = cands['business_name'].apply(clean_name)

# Create a strict blocking key: Country + Name
s1['block_key'] = s1['country'].astype(str).str.lower() + "_" + s1['name_key']
cands['block_key'] = cands['country'].astype(str).str.lower() + "_" + cands['name_key']

s1_valid = s1[s1['name_key'] != ""]
cands_valid = cands[cands['name_key'] != ""]

print("Blocking (Exact Match on Country + Name)...")
merged = s1_valid[['entity_id', 'block_key', 'business_address']].merge(
    cands_valid[['entity_id', 'block_key', 'business_address']], 
    on='block_key', 
    suffixes=('_s1', '_cand')
)

print(f"Generated {len(merged)} candidate pairs for {len(s1)} S1 entities.")

print("Computing Address Similarity...")
# Calculate address trigram Jaccard similarity
def calc_sim(row):
    addr1 = clean_address(row['business_address_s1'])
    addr2 = clean_address(row['business_address_cand'])
    tri1 = get_trigrams(addr1)
    tri2 = get_trigrams(addr2)
    return jaccard_sim(tri1, tri2)

merged['addr_sim'] = merged.apply(calc_sim, axis=1)

# Tune Threshold
ADDRESS_THRESHOLD = 0.40  # Can be tuned!

print(f"Applying Threshold (> {ADDRESS_THRESHOLD})...")
final_matches = merged[merged['addr_sim'] > ADDRESS_THRESHOLD]

print("Aggregating...")
matches = final_matches.groupby('entity_id_s1')['entity_id_cand'].apply(lambda x: ",".join(x.unique())).reset_index()
matches.rename(columns={'entity_id_s1': 'source1_entity_id', 'entity_id_cand': 'pred_ids'}, inplace=True)

print("Evaluating against Ground Truth...")
# Merge with GT
gt['matched_entity_ids'] = gt['matched_entity_ids'].fillna("")
eval_df = gt[gt['source1_entity_id'].isin(s1['entity_id'])].merge(matches, on='source1_entity_id', how='left')
eval_df['pred_ids'] = eval_df['pred_ids'].fillna("")

def compute_f05(row):
    true_ids = set(row['matched_entity_ids'].split(",")) if row['matched_entity_ids'] else set()
    pred_ids = set(row['pred_ids'].split(",")) if row['pred_ids'] else set()
    
    if not true_ids and not pred_ids:
        return 1.0, 1.0, 1.0
    if not true_ids and pred_ids:
        return 0.0, 0.0, 0.0
    if true_ids and not pred_ids:
        return 0.0, 0.0, 0.0
        
    tp = len(true_ids.intersection(pred_ids))
    fp = len(pred_ids - true_ids)
    fn = len(true_ids - pred_ids)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    if precision == 0 and recall == 0:
        return 0.0, precision, recall
        
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return f05, precision, recall

metrics = eval_df.apply(lambda r: pd.Series(compute_f05(r)), axis=1)
eval_df[['f05', 'precision', 'recall']] = metrics

print("\n--- LOCAL VALIDATION SCORE ---")
print(f"Mean F0.5:     {eval_df['f05'].mean():.4f}")
print(f"Mean Precision:{eval_df['precision'].mean():.4f}")
print(f"Mean Recall:   {eval_df['recall'].mean():.4f}")
