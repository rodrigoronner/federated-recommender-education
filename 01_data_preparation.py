"""
Step 1 — Data Preparation and Feature Engineering
==================================================
Loads the raw ASSISTments Skill Builder dataset, applies the filtering
and feature-engineering pipeline described in the paper (Section 4.2),
and saves the processed dataset ready for both centralised and federated training.

Dataset download:
    https://doi.org/10.34740/kaggle/dsv/13081046
    Place 'skill_builder_data_corrected_collapsed.csv' in the data/ folder.

Output:
    data/processed_assistments.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

RAW_PATH   = os.path.join("data", "skill_builder_data_corrected_collapsed.csv")
OUT_PATH   = os.path.join("data", "processed_assistments.csv")

MIN_STUDENT_INTERACTIONS = 50   # minimum interactions per student (Section 4.2, Step 1)
MIN_SKILL_ATTEMPTS       = 100  # minimum total attempts per skill  (Section 4.2, Step 1)
SUCCESS_THRESHOLD        = 0.70 # target_correct_rate >= 0.70 → class 1 (Section 4.2, Step 3)

# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------
print("Loading raw ASSISTments dataset …")
df = pd.read_csv(RAW_PATH, encoding="latin-1", low_memory=False,
                 usecols=["user_id", "skill_id", "correct"])
df.dropna(inplace=True)
df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
df.dropna(inplace=True)
df["correct"] = df["correct"].astype(int)
print(f"  Raw interactions: {len(df):,}")

# ---------------------------------------------------------------------------
# 2. Filtering (Step 1 — Section 4.2)
# ---------------------------------------------------------------------------
valid_students = df.groupby("user_id").size()
valid_students = valid_students[valid_students >= MIN_STUDENT_INTERACTIONS].index

valid_skills = df.groupby("skill_id").size()
valid_skills = valid_skills[valid_skills >= MIN_SKILL_ATTEMPTS].index

df = df[df["user_id"].isin(valid_students) & df["skill_id"].isin(valid_skills)].copy()
print(f"  After filtering: {df['user_id'].nunique()} students, "
      f"{df['skill_id'].nunique()} skills, {len(df):,} interactions")

# ---------------------------------------------------------------------------
# 3. Feature Engineering (Step 2 — Section 4.2)
# ---------------------------------------------------------------------------
user_stats = df.groupby("user_id").agg(
    user_mean_correct    =("correct", "mean"),
    user_interaction_count=("correct", "count"),
).reset_index()

skill_stats = df.groupby("skill_id").agg(
    skill_mean_correct=("correct", "mean"),
).reset_index()

# ---------------------------------------------------------------------------
# 4. Target variable (Step 3 — Section 4.2)
# ---------------------------------------------------------------------------
pair_stats = df.groupby(["user_id", "skill_id"]).agg(
    target_correct_rate=("correct", "mean")
).reset_index()

pair_stats["target"] = (pair_stats["target_correct_rate"] >= SUCCESS_THRESHOLD).astype(int)

# Merge engineered features
pair_stats = pair_stats.merge(user_stats, on="user_id")
pair_stats = pair_stats.merge(skill_stats, on="skill_id")

# ---------------------------------------------------------------------------
# 5. ID mapping — zero-indexed sequential integers (Step 4 — Section 4.2)
# ---------------------------------------------------------------------------
user_ids = sorted(pair_stats["user_id"].unique())
skill_ids = sorted(pair_stats["skill_id"].unique())
user_map  = {uid: i for i, uid in enumerate(user_ids)}
skill_map = {sid: i for i, sid in enumerate(skill_ids)}

pair_stats["user_id_new"]  = pair_stats["user_id"].map(user_map)
pair_stats["skill_id_new"] = pair_stats["skill_id"].map(skill_map)

NUM_USERS  = len(user_ids)
NUM_SKILLS = len(skill_ids)
print(f"  Unique users (clients):  {NUM_USERS}")
print(f"  Unique skills (items):   {NUM_SKILLS}")

# ---------------------------------------------------------------------------
# 6. Feature scaling — MinMax [0, 1] (Step 5 — Section 4.2)
# ---------------------------------------------------------------------------
feat_cols = ["user_mean_correct", "user_interaction_count", "skill_mean_correct"]
scaler = MinMaxScaler()
pair_stats[feat_cols] = scaler.fit_transform(pair_stats[feat_cols])

# ---------------------------------------------------------------------------
# 7. Class distribution report
# ---------------------------------------------------------------------------
dist = pair_stats["target"].value_counts(normalize=True)
print(f"\nClass distribution (after thresholding at {SUCCESS_THRESHOLD}):")
print(f"  Success (class 1): {dist.get(1, 0):.1%}")
print(f"  Failure (class 0): {dist.get(0, 0):.1%}")

# ---------------------------------------------------------------------------
# 8. Save
# ---------------------------------------------------------------------------
pair_stats.to_csv(OUT_PATH, index=False)
print(f"\nProcessed dataset saved to: {OUT_PATH}")
print(f"Total student-skill pairs:  {len(pair_stats):,}")
