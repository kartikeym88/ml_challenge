"""
FAST Entity Resolution Pipeline — Pass-1-Only + Vectorized RapidFuzz + LightGBM
=================================================================================
Key insight from the crash: Pass 1 (exact clean name + country) already covers
96%+ of S1 entities. The looser passes (prefix/soundex) generate 1.5 BILLION
join rows and crash RAM. We skip them and rely on:
  - Pass 1: Country + exact clean name  (covers 96% of S1)
  - Feature engine: Vectorized RapidFuzz cdist (multi-threaded C++)
  - LightGBM: trained to distinguish true match from false match
  - Threshold: grid-searched to maximize Macro F0.5
"""

import pandas as pd
import numpy as np
import re, os
import lightgbm as lgb
from rapidfuzz import process as rfp, fuzz
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ─────────────────────────────────────────────
# TEXT NORMALIZATION
# ─────────────────────────────────────────────
LEGAL = {
    r'\bcorp\b':'corporation', r'\bltd\b':'limited', r'\bpvt\b':'private',
    r'\binc\b':'incorporated', r'\bco\b':'company', r'\bllc\b':'llc',
    r'\bllp\b':'llp', r'\bplc\b':'plc', r' & ':' and ',
}
ADDR = {
    r'\brd\b':'road', r'\bst\b':'street', r'\bave\b':'avenue',
    r'\bblvd\b':'boulevard', r'\bdr\b':'drive', r'\bln\b':'lane',
    r'\bct\b':'court', r'\bpl\b':'place',
}

def norm_name(text):
    if pd.isna(text): return ""
    t = str(text).lower()
    for p, r in LEGAL.items(): t = re.sub(p, r, t)
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s]', ' ', t)).strip()

def clean_key(text):
    return re.sub(r'[^a-z0-9]', '', norm_name(text))

def norm_addr(text):
    if pd.isna(text): return ""
    t = str(text).lower()
    for p, r in ADDR.items(): t = re.sub(p, r, t)
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s]', ' ', t)).strip()

def extract_nums(text):
    return set(re.findall(r'\d+', str(text))) if not pd.isna(text) else set()

def word_jac(a, b):
    wa, wb = set(a.split()), set(b.split())
    if not wa and not wb: return 1.0
    if not wa or not wb: return 0.0
    return len(wa & wb) / len(wa | wb)

def tri_jac(a, b):
    a = re.sub(r'\s+', '', a); b = re.sub(r'\s+', '', b)
    sa = {a[i:i+3] for i in range(len(a)-2)} if len(a) >= 3 else {a}
    sb = {b[i:i+3] for i in range(len(b)-2)} if len(b) >= 3 else {b}
    if not sa and not sb: return 1.0
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_and_prep(s1_path, s2_path, s3_path):
    s1    = pd.read_csv(s1_path, sep='\t', dtype=str).fillna("")
    s2    = pd.read_csv(s2_path, sep='\t', dtype=str).fillna("")
    s3    = pd.read_csv(s3_path, sep='\t', dtype=str).fillna("")
    cands = pd.concat([s2, s3], ignore_index=True)
    for df in [s1, cands]:
        df['nn'] = df['business_name'].apply(norm_name)
        df['ck'] = df['business_name'].apply(clean_key)
        df['na'] = df['business_address'].apply(norm_addr)
        df['cc'] = df['country'].str.strip().str.lower()
    return s1, cands

# ─────────────────────────────────────────────
# BLOCKING (Pass 1 only — covers 96%+ of S1)
# ─────────────────────────────────────────────
CAP = 50  # max candidates per S1 entity

def blocking(s1, cands):
    print("  Pass 1: Country + exact clean name")
    s1v = s1[s1['ck'].str.len() > 0].copy()
    cv  = cands[cands['ck'].str.len() > 0].copy()
    s1v['_bk'] = s1v['cc'] + "|" + s1v['ck']
    cv['_bk']  = cv['cc']  + "|" + cv['ck']

    # IMPORTANT: filter out super-common keys before merge to prevent OOM
    key_freq = cv['_bk'].value_counts()
    hot_keys = set(key_freq[key_freq > 5000].index)
    print(f"  Dropping {len(hot_keys)} hot keys (>5000 cands each) to prevent OOM")
    s1v = s1v[~s1v['_bk'].isin(hot_keys)]
    cv  = cv[~cv['_bk'].isin(hot_keys)]

    pairs = s1v[['entity_id', '_bk', 'nn', 'na', 'cc']].merge(
        cv[['entity_id', '_bk', 'nn', 'na', 'cc']],
        on='_bk', suffixes=('_s1', '_c')
    ).drop(columns=['_bk'])
    pairs = pairs[pairs['entity_id_s1'] != pairs['entity_id_c']]
    pairs = pairs.groupby('entity_id_s1').head(CAP).reset_index(drop=True)
    print(f"  {len(pairs):,} pairs covering {pairs['entity_id_s1'].nunique():,} S1 entities")
    return pairs

# ─────────────────────────────────────────────
# VECTORIZED FEATURE ENGINEERING (chunked)
# ─────────────────────────────────────────────
FCOLS = ['nm_wr','nm_tsr','nm_pr','nm_wj','nm_ex',
         'ad_wr','ad_tsr','ad_tj','ad_wj','num_j','cc_m']

CHUNK = 50_000

def compute_features(pairs):
    n = len(pairs)
    print(f"  Features for {n:,} pairs in chunks of {CHUNK:,}...")
    results = []
    for start in tqdm(range(0, n, CHUNK)):
        end   = min(start + CHUNK, n)
        chunk = pairs.iloc[start:end]
        s1n = chunk['nn_s1'].tolist(); cn = chunk['nn_c'].tolist()
        s1a = chunk['na_s1'].tolist(); ca = chunk['na_c'].tolist()
        s1c = chunk['cc_s1'].tolist(); cc = chunk['cc_c'].tolist()

        nm_wr  = rfp.cdist(s1n, cn, scorer=fuzz.WRatio, workers=-1, dtype=np.float32).diagonal() / 100.0
        nm_tsr = rfp.cdist(s1n, cn, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float32).diagonal() / 100.0
        nm_pr  = rfp.cdist(s1n, cn, scorer=fuzz.partial_ratio, workers=-1, dtype=np.float32).diagonal() / 100.0
        ad_wr  = rfp.cdist(s1a, ca, scorer=fuzz.WRatio, workers=-1, dtype=np.float32).diagonal() / 100.0
        ad_tsr = rfp.cdist(s1a, ca, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float32).diagonal() / 100.0

        nm_wj  = np.array([word_jac(a, b) for a, b in zip(s1n, cn)], dtype=np.float32)
        nm_ex  = np.array([1.0 if a == b else 0.0 for a, b in zip(s1n, cn)], dtype=np.float32)
        ad_tj  = np.array([tri_jac(a, b) for a, b in zip(s1a, ca)], dtype=np.float32)
        ad_wj  = np.array([word_jac(a, b) for a, b in zip(s1a, ca)], dtype=np.float32)
        s1nums = [extract_nums(a) for a in s1a]
        cnums  = [extract_nums(a) for a in ca]
        num_j  = np.array([len(a&b)/len(a|b) if (a|b) else 1.0 for a,b in zip(s1nums,cnums)], dtype=np.float32)
        cc_m   = np.array([1 if a == b else 0 for a, b in zip(s1c, cc)], dtype=np.float32)

        results.append(pd.DataFrame({
            'nm_wr':nm_wr,'nm_tsr':nm_tsr,'nm_pr':nm_pr,'nm_wj':nm_wj,'nm_ex':nm_ex,
            'ad_wr':ad_wr,'ad_tsr':ad_tsr,'ad_tj':ad_tj,'ad_wj':ad_wj,
            'num_j':num_j,'cc_m':cc_m,
        }))
    return pd.concat(results, ignore_index=True)

# ─────────────────────────────────────────────
# F0.5 METRIC
# ─────────────────────────────────────────────
def macro_f05(pairs_prob, gt_df, s1_ids, threshold):
    matched  = pairs_prob[pairs_prob['prob'] >= threshold]
    pred_map = matched.groupby('entity_id_s1')['entity_id_c'].apply(set).to_dict()
    gt_map   = gt_df.set_index('source1_entity_id')['matched_entity_ids'].fillna("").to_dict()
    scores = []
    for sid in s1_ids:
        true_ids = set(gt_map.get(sid,"").split(",")) - {""}
        pred_ids = pred_map.get(sid, set())
        if   not true_ids and not pred_ids: scores.append(1.0)
        elif not true_ids:                  scores.append(0.0)
        elif not pred_ids:                  scores.append(0.0)
        else:
            tp = len(true_ids & pred_ids); fp = len(pred_ids-true_ids); fn = len(true_ids-pred_ids)
            p  = tp/(tp+fp) if (tp+fp) else 0; r = tp/(tp+fn) if (tp+fn) else 0
            scores.append((1.25*p*r)/(0.25*p+r) if (p or r) else 0.0)
    return float(np.mean(scores)) if scores else 0.0

# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
def train(train_dir):
    print("\n=== TRAINING ===")
    s1, cands = load_and_prep(f"{train_dir}/train_source1.tsv",
                               f"{train_dir}/train_source2.tsv",
                               f"{train_dir}/train_source3.tsv")
    gt = pd.read_csv(f"{train_dir}/train_ground_truth.tsv", sep='\t', dtype=str).fillna("")
    pos_pairs = set()
    for _, row in gt.iterrows():
        for cid in row['matched_entity_ids'].split(","):
            if cid.strip(): pos_pairs.add((row['source1_entity_id'], cid.strip()))
    print(f"GT positive pairs: {len(pos_pairs):,}")

    pairs = blocking(s1, cands)
    pairs['label'] = [(1 if (r.entity_id_s1, r.entity_id_c) in pos_pairs else 0)
                      for r in pairs.itertuples(index=False)]
    pos = int(pairs['label'].sum()); neg = len(pairs)-pos
    print(f"Blocking recall: {pos/max(len(pos_pairs),1):.4f} | Pos:{pos:,} Neg:{neg:,}")

    print("Computing features...")
    X = compute_features(pairs); y = pairs['label'].values
    X_tr,X_val,y_tr,y_val,_,idx_val = train_test_split(
        X,y,np.arange(len(pairs)),test_size=0.2,stratify=y,random_state=42)
    pairs_val = pairs.iloc[idx_val].reset_index(drop=True)

    print(f"Training LightGBM on {len(X_tr):,} pairs...")
    model = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.05, max_depth=7, num_leaves=127,
        min_child_samples=30, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=5, lambda_l1=0.1, lambda_l2=0.1,
        scale_pos_weight=neg/max(pos,1), random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(X_tr[FCOLS], y_tr,
              eval_set=[(X_val[FCOLS], y_val)],
              callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(-1)])

    print("\nTuning threshold...")
    pairs_val['prob'] = model.predict_proba(X_val[FCOLS])[:,1]
    val_ids = list(set(pairs_val['entity_id_s1']))
    best_t, best_f = 0.5, 0.0
    for t in np.arange(0.30, 0.97, 0.01):
        s = macro_f05(pairs_val, gt, val_ids, t)
        if s > best_f: best_f=s; best_t=round(float(t),2)
    print(f"Best threshold: {best_t:.2f} → Val F0.5: {best_f:.4f}")
    return model, best_t

# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def infer(model, threshold, test_dir, output_dir):
    print("\n=== INFERENCE ===")
    s1, cands = load_and_prep(f"{test_dir}/test_source1.tsv",
                               f"{test_dir}/test_source2.tsv",
                               f"{test_dir}/test_source3.tsv")
    pairs = blocking(s1, cands)
    X = compute_features(pairs)
    pairs['prob'] = model.predict_proba(X[FCOLS])[:,1]

    # Hard-match override for extremely confident pairs
    hard = X[(X['nm_wr'] > 0.97) & (X['ad_wr'] > 0.70)].index
    pairs.loc[hard, 'prob'] = 1.0
    print(f"Hard-match overrides: {len(hard):,}")

    matched = pairs[pairs['prob'] >= threshold]
    cand_agg = pairs.groupby('entity_id_s1')['entity_id_c'].apply(
        lambda x: ",".join(x.unique())).reset_index()
    cand_agg.columns = ['source1_entity_id','candidate_entity_ids']
    match_agg = matched.groupby('entity_id_s1')['entity_id_c'].apply(
        lambda x: ",".join(x.unique())).reset_index()
    match_agg.columns = ['source1_entity_id','matched_entity_ids']

    base = pd.DataFrame({'source1_entity_id': s1['entity_id']})
    fm = base.merge(match_agg, on='source1_entity_id', how='left')
    fm['matched_entity_ids'] = fm['matched_entity_ids'].fillna("")
    fc = base.merge(cand_agg, on='source1_entity_id', how='left')
    fc['candidate_entity_ids'] = fc['candidate_entity_ids'].fillna("")

    os.makedirs(output_dir, exist_ok=True)
    fm.to_csv(f"{output_dir}/matching_results.tsv", sep='\t', index=False)
    fc.to_csv(f"{output_dir}/candidate_pairs.tsv",  sep='\t', index=False)
    print(f"Matched: {(fm['matched_entity_ids']!='').sum():,} | Singletons: {(fm['matched_entity_ids']=='').sum():,}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    model, threshold = train("dataset/train")
    infer(model, threshold, "dataset/test", "output")
    print("\nValidating...")
    ret = os.system("python utils/validate_submission.py "
                    "--matching output/matching_results.tsv "
                    "--candidate output/candidate_pairs.tsv "
                    "--test-dir dataset/test")
    print("\n✅ PASS - Ready to submit!" if ret == 0 else "\n❌ Validation failed!")
