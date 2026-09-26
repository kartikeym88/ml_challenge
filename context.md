# Amazon ML Challenge 2026: Official Rules & Context

This document contains the consolidated rules, guidelines, and context for the Amazon ML Challenge 2026. Keep this as a reference for all constraints, deadlines, and formatting requirements.

## 🕒 Timeline & Milestones
*   **Challenge Window:** 25th September 2026, 12:00 AM IST to 27th September 2026, 11:59 PM IST.
*   **Current Status:** 36 hours have passed. 
*   **48-hour Milestone:** The Top 500 teams on the leaderboard will receive $100 worth of AWS credits.
*   **Submissions:** Maximum of 5 submissions per day allowed.

## 🎯 The Problem: Business Entity Resolution
*   **Goal:** Match business records from Source 2 and Source 3 to a deduplicated reference Source 1.
*   **Data Format:** Tab-separated values (`.tsv`). Crucial: Read with `sep="\t"` because addresses and IDs contain commas.
*   **Noise Patterns:** Name variations (abbreviations, suffixes, typos) and Address variations (missing parts, transliterations, landmarks).
*   **Geography:** Train set contains US and India. **Test set contains France (unseen in training)**. Do not hard-code countries!

## 📏 Evaluation Metric ($F_{0.5}$)
The metric heavily penalizes false positives (merging distinct businesses):
`F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)`
*   Computed per Source 1 entity, then macro-averaged.
*   **Singletons:** Correctly predicting an empty list for an entity with no matches yields a perfect `1.0` for that entity. Predicting a false match for a singleton yields `0.0`.

## 📦 Output Formatting
Your pipeline must produce two files in the `output/` directory:

### 1. `matching_results.tsv` (The Final Matches)
*   **Columns:** `source1_entity_id`, `matched_entity_ids` (comma-separated).
*   **Rules:** Every Source 1 test entity MUST have exactly one row. Empty list for singletons. No duplicates. Only S2/S3 IDs that exist in the test set.

### 2. `candidate_pairs.tsv` (The Blocking Candidates)
*   **Columns:** `source1_entity_id`, `candidate_entity_ids` (comma-separated).
*   **Rules:** Same formatting as above. This is the exact set fed into your ML model for inference. 
*   **Crucial Update:** Candidate generation counts toward final rankings. Blocking MUST scale. Smaller candidate sets with high recall rank higher.

## 🚀 Final Submission Structure
A single `.zip` file `<team_name>_submission.zip` containing:
```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

## 🚫 Constraints & Disqualifiers (STRICT)
*   **NO External Data:** Using commercial APIs, government databases, geocoding APIs, or internet lookups is strictly prohibited and results in instant disqualification.
*   **Model Size:** Final model must be open-source (MIT/Apache 2.0) and **under 8 Billion parameters**.
*   **Validation:** Always run `python3 utils/validate_submission.py` locally before submitting.

## 💡 Tips for Success from Organizers
1.  **Strong Blocking:** Determines the upper bound of your recall. Must be highly scalable.
2.  **String Similarity:** Use Jaccard, Levenshtein, TF-IDF cosine.
3.  **Address Patterns:** Pay attention to country-specific formats.
4.  **$F_{0.5}$ Trade-off:** Rewards precision more than recall. Tune your thresholds accordingly.
5.  **Singletons:** Do not neglect them! Predicting "no match" is highly rewarding.
