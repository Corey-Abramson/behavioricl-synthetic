"""
Participant Validity Matrix
===========================
For each participant, checks whether they have valid data in each game app.

Validity rules:
  - Multiplayer apps (compete, cooperation, coordination, trust, guessing_game):
      valid = participant was in a complete group (correct group size) in ALL rounds
  - Solo/session apps (games_solo, redistribute, sel, wikipedia_task, redistribute_team):
      valid = participant has at least one non-null decision column

Test set  : trust
Train set : compete, cooperation, coordination, guessing_game, games_solo,
            redistribute, sel, wikipedia_task, redistribute_team
"""

import pandas as pd
import re

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAW_FILE = "synthetic_data.csv"   # synthetic export: see make_synthetic_data.py
SESSION_COL = "session.code"
PLAYER_COL = "participant.code"

MULTIPLAYER_SIZES = {
    "compete": 2,
    "cooperation": 2,
    "coordination": 2,
    "trust": 2,
    "guessing_game": 4,
}

# Apps treated as solo (data-presence check only)
SOLO_APPS = [
    "games_solo",
    "redistribute",
    "sel",
    "wikipedia_task",
    "redistribute_team",
]

TEST_APPS = ["trust"]

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
df = pd.read_csv(RAW_FILE, low_memory=False)
print(f"Loaded {len(df):,} participants\n")

participants = df[PLAYER_COL].tolist()
validity = pd.DataFrame(index=df[PLAYER_COL], dtype=bool)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_col(columns, app, keyword):
    return next((c for c in columns if c.startswith(f"{app}.") and keyword in c), None)

def decision_cols(columns, app):
    """Return player decision columns for a solo app (exclude id/admin cols)."""
    skip = {"id_in_group", "page_sequence", "id_in_subsession"}
    return [
        c for c in columns
        if re.match(rf"{re.escape(app)}\.\d+\.player\.", c)
        and not any(s in c for s in skip)
    ]

# ---------------------------------------------------------------------------
# Multiplayer validity: complete group in every round
# ---------------------------------------------------------------------------
for app, expected_size in MULTIPLAYER_SIZES.items():
    group_col = find_col(df.columns, app, "group.id_in_subsession")
    sub_col = find_col(df.columns, app, "subsession.round_number")

    if not group_col or not sub_col:
        print(f"  {app}: columns not found — skipping")
        validity[app] = False
        continue

    app_df = df[[SESSION_COL, PLAYER_COL, sub_col, group_col]].dropna(
        subset=[group_col, sub_col]
    )

    # For each (session, round, group): is size correct?
    group_validity = (
        app_df
        .groupby([SESSION_COL, sub_col, group_col])[PLAYER_COL]
        .transform("count")
        == expected_size
    )
    app_df = app_df.copy()
    app_df["valid_group"] = group_validity

    # Participant is valid only if ALL their groups in this app are valid
    participant_valid = (
        app_df.groupby(PLAYER_COL)["valid_group"].all()
    )

    validity[app] = df[PLAYER_COL].map(participant_valid).fillna(False).values

# ---------------------------------------------------------------------------
# Solo validity: has at least one non-null decision value
# ---------------------------------------------------------------------------
for app in SOLO_APPS:
    dcols = decision_cols(df.columns, app)

    if not dcols:
        print(f"  {app}: no decision columns found — skipping")
        validity[app] = False
        continue

    has_data = df[dcols].notna().any(axis=1)
    validity[app] = has_data.values


validity.index = df[PLAYER_COL]

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
all_apps = list(MULTIPLAYER_SIZES.keys()) + SOLO_APPS
train_apps = [a for a in all_apps if a not in TEST_APPS]

print("=" * 60)
print("VALIDITY RATES PER APP")
print("=" * 60)
for app in all_apps:
    rate = validity[app].mean() * 100
    tag = "[TEST] " if app in TEST_APPS else "[TRAIN]"
    kind = "multiplayer" if app in MULTIPLAYER_SIZES else "solo"
    print(f"  {tag} {app:<22} {rate:5.1f}%  ({kind})")

# ---------------------------------------------------------------------------
# Per-participant game count
# ---------------------------------------------------------------------------
validity["n_valid_apps"] = validity[all_apps].sum(axis=1)
validity["all_test_valid"] = validity[TEST_APPS].all(axis=1)

print("\n" + "=" * 60)
print("PARTICIPANT COVERAGE")
print("=" * 60)

total = len(validity)
for n in range(len(all_apps), -1, -1):
    count = (validity["n_valid_apps"] >= n).sum()
    pct = count / total * 100
    if count > 0:
        print(f"  Valid in >= {n:2d} apps : {count:>5,}  ({pct:.1f}%)")

print()
evaluable = validity["all_test_valid"].sum()
print(f"  Evaluable (valid in test app) : {evaluable:,} / {total:,}  ({evaluable/total*100:.1f}%)")

# ---------------------------------------------------------------------------
# Save matrix
# ---------------------------------------------------------------------------
out_path = "validity_matrix.csv"
validity.to_csv(out_path)
print(f"\nMatrix saved to: {out_path}")
print(f"Columns: {all_apps + ['n_valid_apps', 'all_test_valid']}")

# ---------------------------------------------------------------------------
# Save filtered dataset
# ---------------------------------------------------------------------------
evaluable_participants = validity[validity["all_test_valid"]].index
filtered_df = df[df[PLAYER_COL].isin(evaluable_participants)]

filtered_out = "evaluable_participants.csv"
filtered_df.to_csv(filtered_out, index=False)
print(f"\nFiltered dataset saved to: {filtered_out}  (valid in trust)")
print(f"  Participants : {filtered_df[PLAYER_COL].nunique():,}")
print(f"  Rows         : {len(filtered_df):,}")
print(f"  Columns      : {len(filtered_df.columns):,}")
