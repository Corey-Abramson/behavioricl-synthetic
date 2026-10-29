"""
Prompt Construction — BehaviorICL
Builds one structured prompt record per participant from their behavioral history,
holding out the target app for a given trait.

Usage:
    python build_prompts.py <trait_name>
    python build_prompts.py trust
    python build_prompts.py competitiveness

Output: 1_Prompts/{trait_name}/prompts.jsonl
  Each record: {participant_code, prompt, target_cols..., partner_send (trust only)}
"""

import json
import math
import sys
import pandas as pd
from pathlib import Path

from trait_configs import TRAIT_CONFIGS, TraitConfig

# Full synthetic dataset; filtered per-trait using validity_matrix.csv
DATA_PATH = Path(__file__).parent.parent / "0_Preprocess" / "synthetic_data.csv"
VALIDITY_PATH = Path(__file__).parent.parent / "0_Preprocess" / "validity_matrix.csv"

# Map target_app → validity column name (as in validity_matrix.csv)
APP_VALIDITY_COL = {
    "trust": "trust",
    "compete": "compete",
    "cooperation": "cooperation",
    "coordination": "coordination",
    "guessing_game": "guessing_game",
    "games_solo": "games_solo",
    "redistribute": "redistribute",
    "redistribute_team": "redistribute_team",
}

MORAL_CHOICES_FOR_SELF = [2, 4, 6, 8, 10]


def _val(v):
    """Return None if NaN, else the value."""
    try:
        if math.isnan(float(v)):
            return None
    except (TypeError, ValueError):
        pass
    return v


# ── Section builders ──────────────────────────────────────────────────────────

def build_compete_section(row):
    decision = _val(row["compete.1.player.compete_decision"])
    centered = _val(row["compete.1.player.centered_sliders"])
    num = _val(row["compete.1.player.num_sliders"])

    if decision is None:
        return None

    choice_str = "Option B (competitive)" if int(decision) == 1 else "Option A (safe/fixed)"
    perf = f"{int(centered)} out of {int(num)}" if centered is not None and num is not None else "N/A"

    return (
        "**Game 1 — Competition Task (Slider Task)**\n"
        "Students completed a slider task (centering sliders on a 0–100 scale) then chose how to be paid:\n"
        "  • Option A (safe): 0.20 credits per correct slider regardless of partner\n"
        "  • Option B (competitive): 0.50 credits per correct slider only if they outperformed their partner; 0 otherwise\n"
        f"Choice: {choice_str}\n"
        f"Performance: {perf} sliders centered correctly"
    )


def build_cooperation_section(row):
    lines = [
        "**Game 2 — Mutual Aid Game (4 rounds)**",
        "Each round, students received 1 credit and chose how much to send to a partner (0–1, in steps of 0.1).",
        "Whatever was sent was doubled before reaching the partner.",
    ]
    for r in range(1, 5):
        col = f"cooperation.{r}.player.cooperate_decision"
        val = _val(row[col]) if col in row.index else None
        if val is not None:
            lines.append(f"  Round {r}: sent {float(val):.1f} credits")
        else:
            lines.append(f"  Round {r}: no data")
    return "\n".join(lines)


def build_coordination_section(row):
    lines = [
        "**Game 3 — Coordination Game (4 rounds)**",
        "Each round, students chose Option A or Option B when paired with a partner:",
        "  • Option A: always earns 3 credits (safe)",
        "  • Option B: earns 5 credits only if partner also chose B; earns 0 otherwise (risky)",
    ]
    for r in range(1, 5):
        col = f"coordination.{r}.player.coordinate_decision"
        val = _val(row[col]) if col in row.index else None
        if val is not None:
            lines.append(f"  Round {r}: chose {val}")
        else:
            lines.append(f"  Round {r}: no data")
    return "\n".join(lines)


def build_guessing_section(row):
    treatment = _val(row["guessing_game.1.player.treatment"])
    lines = [
        "**Game 4 — Guessing Game / Beauty Contest (4 rounds)**",
        f"Each round, students guessed a number from 0–100. The student closest to {treatment or '?'} of the group average won 6 credits.",
    ]
    for r in range(1, 5):
        guess_col = f"guessing_game.{r}.player.guess"
        win_col = f"guessing_game.{r}.player.is_winner"
        guess = _val(row[guess_col]) if guess_col in row.index else None
        won = _val(row[win_col]) if win_col in row.index else None
        if guess is not None:
            won_str = " (won)" if won == 1.0 or won is True else ""
            lines.append(f"  Round {r}: guessed {float(guess):.0f}{won_str}")
        else:
            lines.append(f"  Round {r}: no data")
    return "\n".join(lines)


def build_games_solo_section(row, exclude=None):
    """
    exclude: list of task names to omit from the section.
    Options: "time", "bret", "dictator", "moral", "lying"
    """
    if exclude is None:
        exclude = []

    time_dec = _val(row["games_solo.1.player.time_decision"])
    boxes = _val(row["games_solo.1.player.boxes_collected"])
    dictator = _val(row["games_solo.1.player.dictator_decision"])
    lying = _val(row["games_solo.1.player.lying_decision"])
    moral = [_val(row[f"games_solo.1.player.moral_decision_{n}"]) for n in range(1, 6)]

    lines = ["**Game 5 — Individual Decision Games (5 tasks)**"]

    # 5a Time preferences
    if "time" not in exclude:
        if time_dec is not None:
            today = 5 - int(time_dec)
            delayed = int(time_dec) * 2
            lines.append(
                f"  5a. Time Preferences: allocated {int(time_dec)} credit(s) to next month "
                f"(receives {delayed} credits next month) and {today} credit(s) today"
            )
        else:
            lines.append("  5a. Time Preferences: no data")

    # 5b BRET
    if "bret" not in exclude:
        if boxes is not None:
            lines.append(
                f"  5b. Risk Task (BRET): opened {int(boxes)} out of 10 boxes "
                "(one box hid a bomb; opening it meant earning nothing)"
            )
        else:
            lines.append("  5b. Risk Task (BRET): no data")

    # 5c Dictator
    if "dictator" not in exclude:
        if dictator is not None:
            kept = 10 - int(dictator)
            lines.append(
                f"  5c. Dictator Game: out of 10 credits, gave {int(dictator)} to another student and kept {kept}"
            )
        else:
            lines.append("  5c. Dictator Game: no data")

    # 5d Moral
    if "moral" not in exclude:
        if any(m is not None for m in moral):
            moral_lines = ["  5d. Moral Preferences (5 decisions — for each, chose between personal earnings or donation to UNICEF):"]
            for i, m in enumerate(moral, 1):
                self_amount = MORAL_CHOICES_FOR_SELF[i - 1]
                if m is not None:
                    choice = "donated to UNICEF (10 credits)" if int(m) == 1 else f"kept {self_amount} credits for themselves"
                    moral_lines.append(f"       Decision {i} (self={self_amount} credits vs UNICEF): {choice}")
                else:
                    moral_lines.append(f"       Decision {i}: no data")
            lines.extend(moral_lines)
        else:
            lines.append("  5d. Moral Preferences: no data")

    # 5e Lying
    if "lying" not in exclude:
        if lying is not None:
            lines.append(
                f"  5e. Lying Game: privately rolled a die and reported {int(lying)} "
                "(higher reports = higher payment; truthful reporting cannot be verified)"
            )
        else:
            lines.append("  5e. Lying Game: no data")

    # If all tasks were excluded, return None
    if len(lines) == 1:
        return None

    return "\n".join(lines)


def build_redistribute_section(row):
    decision = _val(row["redistribute.1.player.redistribute_decision"])
    version = _val(row["redistribute.1.player.redistribute_version"])

    if decision is None:
        return None

    # The oTree slider stores B's credits; A gets the remainder
    give_b = int(decision)
    give_a = 10 - give_b

    if version == "merit":
        context = (
            "Participant A earned 9 credits (better slider performance) and "
            "Participant B earned 1 credit. The third party could redistribute."
        )
    elif version == "no_merit":
        context = (
            "Participant A was randomly assigned 9 credits and "
            "Participant B was randomly assigned 1 credit."
        )
    else:
        context = "Allocation context unknown."

    return (
        "**Game 6 — Redistribution Game**\n"
        "A neutral third party decided how to redistribute 10 credits between two other students (A and B).\n"
        f"Context: {context}\n"
        f"Decision: gave Participant A {give_a} credits and Participant B {give_b} credits"
    )


def build_redistribute_team_section(row):
    first = _val(row["redistribute_team.1.player.redistribute_decision_first"])
    second = _val(row["redistribute_team.1.player.redistribute_decision_second"])
    group_dec = _val(row["redistribute_team.1.group.group_redistribute_decision"])

    if first is None:
        return None

    lines = [
        "**Game 7 — Team Redistribution Game**",
        "A pair of students jointly decided how to redistribute 10 credits between two other students (A and B).",
        f"  This student's initial proposal: give A {int(first)} credits and B {10 - int(first)} credits",
    ]
    if second is not None:
        lines.append(f"  After discussion, revised to: give A {int(second)} credits and B {10 - int(second)} credits")
    if group_dec is not None:
        lines.append(f"  Final group decision: give A {int(group_dec)} credits and B {10 - int(group_dec)} credits")

    return "\n".join(lines)


def build_trust_context_section(row):
    """Trust game data as context (for non-trust traits)."""
    send = _val(row["trust.1.player.trust_send"])
    ret = _val(row["trust.1.player.trust_return"])

    lines = [
        "**Game 8 — Trust Game**",
        "Each student played both roles simultaneously:",
        "  • As Sender: receives 5 credits, decides how much to send (0–5, steps of 0.5). Amount sent is tripled.",
        "  • As Receiver: decides how much to return to the sender.",
    ]
    if send is not None:
        lines.append(f"  Sent (as Sender): {float(send):.1f} credits")
    else:
        lines.append("  Sent (as Sender): no data")
    if ret is not None:
        lines.append(f"  Returned (as Receiver): {float(ret):.1f} credits")
    else:
        lines.append("  Returned (as Receiver): no data")

    return "\n".join(lines)


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(row, config: TraitConfig, partner_send: float = None):
    """
    Builds a natural-language behavioral history prompt for the given trait,
    excluding the target app's data and appending the trait prediction question.
    """
    sections = []
    target_app = config.target_app

    header = (
        f"Below is the behavioral history of a student across several economic games played with classmates. "
        f"Use this to predict their behavior in the {config.label} task described at the end.\n"
    )
    sections.append(header)

    # Competition
    if target_app != "compete":
        s = build_compete_section(row)
        if s:
            sections.append(s)

    # Cooperation
    if target_app != "cooperation":
        sections.append(build_cooperation_section(row))

    # Coordination
    if target_app != "coordination":
        sections.append(build_coordination_section(row))

    # Guessing game
    if target_app != "guessing_game":
        sections.append(build_guessing_section(row))

    # Games solo (partial hold-out support)
    if target_app != "games_solo" or config.games_solo_exclude:
        # Include section but exclude specified tasks
        exclude = config.games_solo_exclude if target_app == "games_solo" else []
        s = build_games_solo_section(row, exclude=exclude)
        if s:
            sections.append(s)

    # Redistribute (exclude for tolerance_for_inequality)
    if target_app != "redistribute":
        s = build_redistribute_section(row)
        if s:
            sections.append(s)

    # Redistribute team (exclude for tolerance_for_inequality — closely related)
    if target_app != "redistribute":
        s = build_redistribute_team_section(row)
        if s:
            sections.append(s)

    # Trust context (for all non-trust traits)
    if config.include_trust_context:
        sections.append(build_trust_context_section(row))

    # ── Prediction question ───────────────────────────────────────────────────
    sections.append("---")

    if config.name == "trust":
        # Trust uses partner_send in the question
        received = partner_send * 3
        prediction_q = (
            f"**NOW PREDICT — {config.label}**\n"
            + config.game_description.format(
                partner_send=partner_send,
                received=received,
            )
            + "\n\n"
            + config.prediction_question.format(received=received)
        )
    elif config.name == "depth_of_reasoning":
        treatment = _val(row["guessing_game.1.player.treatment"])
        treatment_str = str(treatment) if treatment is not None else "2/3"
        prediction_q = (
            f"**NOW PREDICT — {config.label}**\n"
            + config.game_description.format(treatment=treatment_str)
            + "\n\n"
            + config.prediction_question
        )
    elif config.name == "tolerance_for_inequality":
        version = _val(row["redistribute.1.player.redistribute_version"])
        if version == "merit":
            context_str = (
                "Participant A earned 9 credits (better slider performance) and "
                "Participant B earned 1 credit. The third party could redistribute."
            )
        elif version == "no_merit":
            context_str = (
                "Participant A was randomly assigned 9 credits and "
                "Participant B was randomly assigned 1 credit."
            )
        else:
            context_str = "Allocation context unknown."
        prediction_q = (
            f"**NOW PREDICT — {config.label}**\n"
            + config.game_description.format(context=context_str)
            + "\n\n"
            + config.prediction_question
        )
    else:
        prediction_q = (
            f"**NOW PREDICT — {config.label}**\n"
            + config.game_description
            + "\n\n"
            + config.prediction_question
        )

    sections.append(prediction_q)

    return "\n\n".join(sections)


# ── Partner send lookup (trust only) ─────────────────────────────────────────

def build_partner_send_lookup(df: pd.DataFrame) -> dict:
    """Returns {participant_code: partner_send} for all matched dyadic pairs."""
    key = df["session.code"] + "_" + df["trust.1.group.id_in_subsession"].astype(str)
    df = df.copy()
    df["_key"] = key
    p1 = df[df["trust.1.player.id_in_group"] == 1][["_key", "participant.code", "trust.1.player.trust_send"]].rename(
        columns={"participant.code": "p1_code", "trust.1.player.trust_send": "p1_send"})
    p2 = df[df["trust.1.player.id_in_group"] == 2][["_key", "participant.code", "trust.1.player.trust_send"]].rename(
        columns={"participant.code": "p2_code", "trust.1.player.trust_send": "p2_send"})
    pairs = p1.merge(p2, on="_key")
    lookup = {}
    for _, r in pairs.iterrows():
        lookup[r["p1_code"]] = float(r["p2_send"])
        lookup[r["p2_code"]] = float(r["p1_send"])
    return lookup


# ── Target value extractor ────────────────────────────────────────────────────

def extract_targets(row, config: TraitConfig) -> dict:
    """Extract true target values for a participant given the trait config."""
    targets = {}
    for col in config.target_cols:
        val = _val(row[col]) if col in row.index else None
        if val is None:
            targets[col] = None
        else:
            # Coordination decisions are stored as "A"/"B" strings; encode B=1, A=0
            if str(val) in ("A", "B"):
                targets[col] = 1.0 if str(val) == "B" else 0.0
            else:
                try:
                    targets[col] = float(val)
                except (TypeError, ValueError):
                    targets[col] = str(val)  # preserve fraction strings like "1/3", "1/2", "2/3"
    return targets


# ── Main ──────────────────────────────────────────────────────────────────────

def main(trait_name: str):
    if trait_name not in TRAIT_CONFIGS:
        print(f"Unknown trait: {trait_name}")
        print(f"Available: {list(TRAIT_CONFIGS.keys())}")
        sys.exit(1)

    config = TRAIT_CONFIGS[trait_name]
    out_dir = Path(__file__).parent / trait_name
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "prompts.jsonl"

    # Load full dataset and filter to participants valid for this trait's target app
    df_full = pd.read_csv(DATA_PATH, low_memory=False)
    validity = pd.read_csv(VALIDITY_PATH, index_col=0)
    print(f"Loaded {len(df_full)} participants (full dataset)")

    validity_col = APP_VALIDITY_COL[config.target_app]
    valid_codes = set(validity[validity[validity_col] == True].index)
    df = df_full[df_full["participant.code"].isin(valid_codes)].copy()
    print(f"Filtered to {len(df)} participants valid in '{config.target_app}'")

    # Partner send lookup — built from full dataset so all dyadic pairs are matched
    partner_send_lookup = build_partner_send_lookup(df_full)
    print(f"Partner send matched for {len(partner_send_lookup)} participants (full dataset)")

    records = []
    skipped = 0
    skipped_nan_send = 0
    for _, row in df.iterrows():
        code = row["participant.code"]

        # Trust requires partner_send for the prediction question
        if trait_name == "trust":
            partner_send = partner_send_lookup.get(code)
            if partner_send is None:
                skipped += 1
                continue
            # A matched partner with no recorded send gives NaN; the prompt would
            # say "partner sent nan credits", so skip the participant
            if math.isnan(partner_send):
                skipped_nan_send += 1
                continue
        else:
            partner_send = None

        # Check that at least one target column has data
        targets = extract_targets(row, config)
        if all(v is None for v in targets.values()):
            skipped += 1
            continue

        prompt = build_prompt(row, config, partner_send=partner_send)
        record = {
            "participant_code": code,
            "prompt": prompt,
            **{col.replace(".", "_"): v for col, v in targets.items()},
        }
        if trait_name == "trust":
            record["partner_send"] = partner_send

        records.append(record)

    if skipped:
        print(f"  Skipped {skipped} participants (no partner match or missing target data)")
    if skipped_nan_send:
        print(f"  Skipped {skipped_nan_send} participants (partner's send is missing)")

    out_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    print(f"Wrote {len(records)} records to {out_path}")

    # Quick sanity check
    sample = records[0]
    print(f"\nSample participant: {sample['participant_code']}")
    print("\n--- PROMPT PREVIEW ---")
    print(sample["prompt"])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python build_prompts.py <trait_name>")
        print(f"Available traits: {list(TRAIT_CONFIGS.keys())}")
        sys.exit(1)
    main(sys.argv[1])
