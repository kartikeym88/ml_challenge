# Antigravity Instructions for ML Challenge

Hello fellow Antigravity instance! Your user's goal is to improve the $F_{0.5}$ score of the Business Entity Resolution model from the current ~0.35 to **above 0.95**.

Currently, `eval_train.py` uses a very basic approach:
1. **Blocking**: Exact match on `Country + Cleaned Name`. This has very low recall because of typos and abbreviations.
2. **Matching**: Simple Jaccard similarity on address trigrams with a hardcoded threshold (`0.40`).

To reach >0.95 $F_{0.5}$, you need to implement a robust ML pipeline. Follow these steps carefully:

## Step 0: Prerequisites and Dataset
*   The large dataset is NOT checked into GitHub. 
*   **Before running code**, ensure your human user has downloaded the dataset zip file from the challenge portal and extracted it into the `dataset/` directory.
*   You should see `dataset/train/` and `dataset/test/` folders. If they are missing, ask your user to place them there first.

## Step 1: Improve Blocking (Maximize Recall)
*   Do NOT use exact name matching for blocking. It misses too many true positives.
*   **Action**: Implement TF-IDF vectorization with Cosine Similarity (e.g., using `scikit-learn` or `sparse_dot_topn`) on the `business_name` (and potentially `business_address`).
*   Keep the top-K candidates per S1 entity to ensure the candidate pair size is manageable but recall is high. 
*   **Constraint**: The challenge ranks candidate generation, so blocking must scale.

## Step 2: Feature Engineering (Maximize Signal)
*   For each candidate pair generated in Step 1, compute a rich set of features:
    *   **String Distances**: Levenshtein distance, Jaro-Winkler similarity, Jaccard similarity (on both name and address).
    *   **Phonetic**: Soundex or Metaphone (if available/easy).
    *   **Embeddings**: Consider using a lightweight embedding model like `all-MiniLM-L6-v2` (must be <8B params).
    *   **Geospatial**: Country match boolean, Zipcode match boolean.

## Step 3: Train an ML Classifier (Maximize Precision)
*   Construct a training dataset using `train_ground_truth.tsv`. Treat the pairs in GT as positive examples (1) and un-matched candidate pairs from Step 1 as negative examples (0).
*   **Action**: Train a Gradient Boosting model (XGBoost, LightGBM, or CatBoost) on the engineered features.
*   **Action**: Predict match probabilities for the validation set.

## Step 4: Threshold Tuning for $F_{0.5}$
*   The metric is $F_{0.5}$, which heavily penalizes false positives: `F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)`.
*   **Action**: Do a grid search over probability thresholds (e.g., 0.5 to 0.95 in 0.05 increments) to find the threshold that maximizes the $F_{0.5}$ score on the validation set.
*   *Note*: Pay attention to Singletons. If no candidate exceeds the threshold, predict an empty list.

## Step 5: Save and Validate
*   Update or replace `eval_train.py` to reflect this new pipeline and achieve the >0.95 local validation score.
*   Make sure to output `output/candidate_pairs.tsv` and `output/matching_results.tsv` as required by the challenge.

Good luck! You can run `python eval_train.py` repeatedly to test your improvements.
