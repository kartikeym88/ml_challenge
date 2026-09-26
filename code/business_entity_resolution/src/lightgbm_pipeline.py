"""
High-Precision Entity Resolution Pipeline
==========================================
Strategy:
  1. Multi-pass blocking (Country + Name, Country + Name-Prefix, Country + Soundex)
  2. Rich feature engineering (RapidFuzz: Jaro-Winkler, Token Sort Ratio; 
     Trigram Jaccard on address; Number overlap; Exact country match)
  3. LightGBM classifier trained on training data
  4. F0.5-optimal threshold tuning on a local validation split
  5. Final inference on test set
"""

import pandas as pd
import numpy as np
import re
import os
import lightgbm as lgb
from rapidfuzz import fuzz
from sklearn.model_selection import train_test_split
from tqdm import tqdm

tqdm.pandas()

# ─────────────────────────────────────────────
# 1. TEXT CLEANING & KEY GENERATION
# ─────────────────────────────────────────────

LEGAL_SUFFIXES = {
    r'\bcorp\b': 'corporation', r'\bcorporation\b': 'corporation',
    r'\bltd\b': 'limited',      r'\blimited\b': 'limited',
    r'\bpvt\b': 'private',      r'\bprivate\b': 'private',
    r'\binc\b': 'incorporated',  r'\bincorporated\b': 'incorporated',
    r'\bllc\b': 'llc',
    r'\bco\b': 'company',       r'\bcompany\b': 'company',
    r'\bllp\b': 'llp',
    r' & ': ' and ',
}

def normalize_name(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    for pattern, replacement in LEGAL_SUFFIXES.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_key(text):
    """Aggressive clean: only alphanumeric, no spaces. Used for blocking."""
    if pd.isna(text): return ""
    return re.sub(r'[^a-z0-9]', '', normalize_name(text))

def simple_soundex(word):
    """Simple soundex for phonetic blocking."""
    if not word: return ""
    word = word.upper()
    soundex_table = str.maketrans('AEHIOUWY', '00000000', '')
    code_table = str.maketrans('BFPVCGJKQSXZDTLMNR', '111122222222334556', '')
    code = word[0]
    rest = word[1:].translate(soundex_table)
    rest = rest.translate(code_table)
    code += rest.replace('0','')
    # Remove consecutive duplicates
    deduped = code[0]
    for ch in code[1:]:
        if ch != deduped[-1]:
            deduped += ch
    return (deduped + '000')[:4]

def name_soundex(text):
    if pd.isna(text) or not str(text).strip(): return ""
    words = normalize_name(str(text)).split()
    if not words: return ""
    return simple_soundex(words[0])

def extract_numbers(text):
    if pd.isna(text): return set()
    return set(re.findall(r'\d+', str(text)))

def name_prefix4(text):
    k = clean_key(text)
    return k[:4] if len(k) >= 4 else k

def clean_addr(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    abbreviations = {
        r'\brd\b': 'road', r'\bst\b': 'street', r'\bave\b': 'avenue',
        r'\bblvd\b': 'boulevard', r'\bdr\b': 'drive', r'\bln\b': 'lane',
        r'\bct\b': 'court', r'\bpl\b': 'place', r'\bsq\b': 'square',
        r'\bnr\b': 'near', r'\bno\b': 'number',
    }
    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def trigrams(text):
    text = re.sub(r'\s+', '', text)
    if len(text) < 3: return set([text]) if text else set()
    return {text[i:i+3] for i in range(len(text)-2)}

def jaccard(s1, s2):
    if not s1 or not s2: return 0.0
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter/union if union else 0.0

# ─────────────────────────────────────────────
# 2. DATA LOADING & FEATURE PREP
# ─────────────────────────────────────────────

def load_and_prepare(path_s1, path_s2, path_s3):
    s1 = pd.read_csv(path_s1, sep='\t', dtype=str).fillna("")
    s2 = pd.read_csv(path_s2, sep='\t', dtype=str).fillna("")
    s3 = pd.read_csv(path_s3, sep='\t', dtype=str).fillna("")
    cands = pd.concat([s2, s3], ignore_index=True)

    for df in [s1, cands]:
        df['norm_name']    = df['business_name'].apply(normalize_name)
        df['clean_name']   = df['business_name'].apply(clean_key)
        df['name_prefix']  = df['business_name'].apply(name_prefix4)
        df['name_soundex'] = df['business_name'].apply(name_soundex)
        df['norm_addr']    = df['business_address'].apply(clean_addr)
        df['country_c']    = df['country'].str.strip().str.lower()

    return s1, cands

# ─────────────────────────────────────────────
# 3. MULTI-PASS BLOCKING
# ─────────────────────────────────────────────

def multi_pass_blocking(s1, cands):
    """
    Pass 1: Country + Exact clean_name
    Pass 2: Country + First-4-chars of name
    Pass 3: Country + Soundex of first word of name
    """
    s1_cols = ['entity_id', 'norm_name', 'norm_addr', 'country_c', 'clean_name', 'name_prefix', 'name_soundex']
    c_cols  = ['entity_id', 'norm_name', 'norm_addr', 'country_c', 'clean_name', 'name_prefix', 'name_soundex']

    pairs_list = []

    # Pass 1: exact clean name + country
    s1_1 = s1[s1['clean_name'] != ""].copy()
    c_1  = cands[cands['clean_name'] != ""].copy()
    s1_1['_bk'] = s1_1['country_c'] + "|" + s1_1['clean_name']
    c_1['_bk']  = c_1['country_c']  + "|" + c_1['clean_name']
    m1 = s1_1[['entity_id','_bk'] + ['norm_name','norm_addr','country_c']].merge(
             c_1[['entity_id','_bk'] + ['norm_name','norm_addr','country_c']],
             on='_bk', suffixes=('_s1','_cand'))
    pairs_list.append(m1)

    # Pass 2: 4-char prefix + country
    s1_2 = s1[(s1['name_prefix'] != "") & (s1['name_prefix'].str.len() == 4)].copy()
    c_2  = cands[(cands['name_prefix'] != "") & (cands['name_prefix'].str.len() == 4)].copy()
    s1_2['_bk'] = s1_2['country_c'] + "|" + s1_2['name_prefix']
    c_2['_bk']  = c_2['country_c']  + "|" + c_2['name_prefix']
    m2 = s1_2[['entity_id','_bk','norm_name','norm_addr','country_c']].merge(
             c_2[['entity_id','_bk','norm_name','norm_addr','country_c']],
             on='_bk', suffixes=('_s1','_cand'))
    pairs_list.append(m2)

    # Pass 3: soundex + country
    s1_3 = s1[s1['name_soundex'] != ""].copy()
    c_3  = cands[cands['name_soundex'] != ""].copy()
    s1_3['_bk'] = s1_3['country_c'] + "|" + s1_3['name_soundex']
    c_3['_bk']  = c_3['country_c']  + "|" + c_3['name_soundex']
    m3 = s1_3[['entity_id','_bk','norm_name','norm_addr','country_c']].merge(
             c_3[['entity_id','_bk','norm_name','norm_addr','country_c']],
             on='_bk', suffixes=('_s1','_cand'))
    pairs_list.append(m3)

    all_pairs = pd.concat(pairs_list, ignore_index=True)
    # Deduplicate pairs
    all_pairs = all_pairs.drop_duplicates(subset=['entity_id_s1', 'entity_id_cand'])
    # Remove self-matches just in case
    all_pairs = all_pairs[all_pairs['entity_id_s1'] != all_pairs['entity_id_cand']]
    print(f"  Total candidate pairs after multi-pass blocking: {len(all_pairs):,}")
    return all_pairs

# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def compute_features(pairs_df):
    """Compute pairwise similarity features using RapidFuzz + custom features."""
    s1_names   = pairs_df['norm_name_s1'].tolist()
    cand_names = pairs_df['norm_name_cand'].tolist()
    s1_addrs   = pairs_df['norm_addr_s1'].tolist()
    cand_addrs = pairs_df['norm_addr_cand'].tolist()

    print("  Computing name features...")
    name_jw        = [fuzz.WRatio(a, b)/100.0 for a,b in tqdm(zip(s1_names, cand_names), total=len(s1_names))]
    name_tsr       = [fuzz.token_sort_ratio(a, b)/100.0 for a,b in zip(s1_names, cand_names)]
    name_partial   = [fuzz.partial_ratio(a, b)/100.0 for a,b in zip(s1_names, cand_names)]

    print("  Computing address features...")
    addr_jw        = [fuzz.WRatio(a, b)/100.0 for a,b in tqdm(zip(s1_addrs, cand_addrs), total=len(s1_addrs))]
    addr_tsr       = [fuzz.token_sort_ratio(a, b)/100.0 for a,b in zip(s1_addrs, cand_addrs)]

    print("  Computing trigram/number features...")
    addr_tri_jac   = [jaccard(trigrams(a), trigrams(b)) for a,b in zip(s1_addrs, cand_addrs)]
    
    s1_nums   = [extract_numbers(a) for a in s1_addrs]
    cand_nums = [extract_numbers(a) for a in cand_addrs]
    num_overlap = [
        len(a & b) / max(len(a | b), 1) if (a or b) else 1.0
        for a, b in zip(s1_nums, cand_nums)
    ]

    country_match = (pairs_df['country_c_s1'] == pairs_df['country_c_cand']).astype(int).tolist()

    features = pd.DataFrame({
        'name_wratio':    name_jw,
        'name_tsr':       name_tsr,
        'name_partial':   name_partial,
        'addr_wratio':    addr_jw,
        'addr_tsr':       addr_tsr,
        'addr_tri_jac':   addr_tri_jac,
        'num_overlap':    num_overlap,
        'country_match':  country_match,
    })
    return features

FEATURE_COLS = ['name_wratio','name_tsr','name_partial','addr_wratio','addr_tsr','addr_tri_jac','num_overlap','country_match']

# ─────────────────────────────────────────────
# 5. F0.5 METRIC
# ─────────────────────────────────────────────

def f05_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    if p == 0 and r == 0: return 0.0
    return (1.25 * p * r) / (0.25 * p + r)

def macro_f05(gt_df, pred_df, s1_ids):
    """Compute macro F0.5 score per entity."""
    pred_map = pred_df.set_index('source1_entity_id')['matched_entity_ids'].to_dict()
    gt_map   = gt_df.set_index('source1_entity_id')['matched_entity_ids'].fillna("").to_dict()
    
    scores = []
    for sid in s1_ids:
        true_ids = set(gt_map.get(sid, "").split(",")) - {""}
        pred_ids = set(pred_map.get(sid, "").split(",")) - {""}
        
        if not true_ids and not pred_ids:
            scores.append(1.0)
        elif not true_ids and pred_ids:
            scores.append(0.0)
        elif true_ids and not pred_ids:
            scores.append(0.0)
        else:
            tp = len(true_ids & pred_ids)
            fp = len(pred_ids - true_ids)
            fn = len(true_ids - pred_ids)
            p = tp/(tp+fp) if (tp+fp) > 0 else 0
            r = tp/(tp+fn) if (tp+fn) > 0 else 0
            if p == 0 and r == 0:
                scores.append(0.0)
            else:
                scores.append((1.25 * p * r) / (0.25 * p + r))
    return np.mean(scores)

# ─────────────────────────────────────────────
# 6. TRAIN LIGHTGBM & TUNE THRESHOLD
# ─────────────────────────────────────────────

def train_model(train_dir):
    print("\n=== TRAINING PHASE ===")
    print("Loading training data...")
    s1, cands = load_and_prepare(
        f"{train_dir}/train_source1.tsv",
        f"{train_dir}/train_source2.tsv",
        f"{train_dir}/train_source3.tsv",
    )
    gt = pd.read_csv(f"{train_dir}/train_ground_truth.tsv", sep='\t', dtype=str).fillna("")

    # Build a lookup: which (s1_id, cand_id) pairs are true matches?
    positive_pairs = set()
    for _, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        for cid in row['matched_entity_ids'].split(","):
            cid = cid.strip()
            if cid:
                positive_pairs.add((s1_id, cid))

    print("Multi-pass blocking on training set...")
    pairs = multi_pass_blocking(s1, cands)
    pairs['label'] = pairs.apply(
        lambda r: 1 if (r['entity_id_s1'], r['entity_id_cand']) in positive_pairs else 0, axis=1
    )
    
    pos_count = pairs['label'].sum()
    neg_count = len(pairs) - pos_count
    print(f"  Positive pairs: {pos_count:,} | Negative pairs: {neg_count:,}")
    print(f"  Blocking recall: {pos_count / max(len(positive_pairs), 1):.4f}")
    
    print("Computing features for training pairs...")
    X = compute_features(pairs)
    y = pairs['label'].values

    # Stratified train/val split for threshold tuning
    X_train, X_val, y_train, y_val, pairs_train, pairs_val = train_test_split(
        X, y, pairs[['entity_id_s1','entity_id_cand']], test_size=0.2, stratify=y, random_state=42
    )

    print("\nTraining LightGBM...")
    scale_pos_weight = neg_count / max(pos_count, 1)
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=63,
        min_child_samples=20,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])

    print("\nTuning threshold for optimal Macro F0.5...")
    val_probs = model.predict_proba(X_val)[:, 1]
    pairs_val = pairs_val.copy()
    pairs_val['prob'] = val_probs
    gt_val = gt[gt['source1_entity_id'].isin(s1[s1['entity_id'].isin(pairs_val['entity_id_s1'])]['entity_id'])]

    best_threshold = 0.5
    best_score = 0.0
    val_s1_ids = s1['entity_id'].tolist()[:5000]  # subset for speed

    for thresh in np.arange(0.3, 0.98, 0.02):
        matched = pairs_val[pairs_val['prob'] >= thresh]
        match_agg = matched.groupby('entity_id_s1')['entity_id_cand'].apply(
            lambda x: ",".join(x.unique())).reset_index()
        match_agg.columns = ['source1_entity_id', 'matched_entity_ids']
        
        score = macro_f05(gt_val, match_agg, 
                         [sid for sid in val_s1_ids if sid in gt_val['source1_entity_id'].values])
        if score > best_score:
            best_score = score
            best_threshold = thresh

    print(f"  Best threshold: {best_threshold:.2f} → Val Macro F0.5: {best_score:.4f}")
    return model, best_threshold

# ─────────────────────────────────────────────
# 7. INFERENCE ON TEST SET
# ─────────────────────────────────────────────

def run_inference(model, threshold, test_dir, output_dir):
    print("\n=== INFERENCE PHASE ===")
    print("Loading test data...")
    s1, cands = load_and_prepare(
        f"{test_dir}/test_source1.tsv",
        f"{test_dir}/test_source2.tsv",
        f"{test_dir}/test_source3.tsv",
    )

    print("Multi-pass blocking on test set...")
    pairs = multi_pass_blocking(s1, cands)

    print("Computing features for test pairs...")
    X_test = compute_features(pairs)

    print("Running inference...")
    pairs['prob'] = model.predict_proba(X_test[FEATURE_COLS])[:, 1]

    print(f"Applying threshold {threshold:.2f}...")
    matched = pairs[pairs['prob'] >= threshold]

    # Candidates = all pairs from blocking
    cand_agg = pairs.groupby('entity_id_s1')['entity_id_cand'].apply(
        lambda x: ",".join(x.unique())).reset_index()
    cand_agg.columns = ['source1_entity_id', 'candidate_entity_ids']

    # Matches = pairs above threshold
    match_agg = matched.groupby('entity_id_s1')['entity_id_cand'].apply(
        lambda x: ",".join(x.unique())).reset_index()
    match_agg.columns = ['source1_entity_id', 'matched_entity_ids']

    # Ensure all S1 entities are present
    all_s1 = pd.DataFrame({'source1_entity_id': s1['entity_id']})
    final_matches = all_s1.merge(match_agg, on='source1_entity_id', how='left')
    final_matches['matched_entity_ids'] = final_matches['matched_entity_ids'].fillna("")
    final_cands = all_s1.merge(cand_agg, on='source1_entity_id', how='left')
    final_cands['candidate_entity_ids'] = final_cands['candidate_entity_ids'].fillna("")

    os.makedirs(output_dir, exist_ok=True)
    final_matches.to_csv(f"{output_dir}/matching_results.tsv", sep='\t', index=False)
    final_cands.to_csv(f"{output_dir}/candidate_pairs.tsv", sep='\t', index=False)
    print(f"\nSaved to {output_dir}/")
    print(f"  Matched {(final_matches['matched_entity_ids'] != '').sum():,} entities")
    print(f"  Singletons: {(final_matches['matched_entity_ids'] == '').sum():,}")

# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    TRAIN_DIR  = "dataset/train"
    TEST_DIR   = "dataset/test"
    OUTPUT_DIR = "output"

    model, threshold = train_model(TRAIN_DIR)
    run_inference(model, threshold, TEST_DIR, OUTPUT_DIR)

    print("\nValidating submission...")
    os.system(f"python utils/validate_submission.py "
              f"--matching {OUTPUT_DIR}/matching_results.tsv "
              f"--candidate {OUTPUT_DIR}/candidate_pairs.tsv "
              f"--test-dir {TEST_DIR}")
