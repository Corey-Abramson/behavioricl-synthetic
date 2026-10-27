"""
Synthetic Data Generator — BehaviorICL (synthetic demonstration export)
=======================================================================

Writes `synthetic_data.csv`: a fully synthetic oTree-style participant export
that the rest of the pipeline consumes exactly as it consumed the original
study export.

NOTHING HERE IS DERIVED FROM REAL PARTICIPANTS.
Every value is drawn from the generative model below. The parameters were
chosen by hand from two things only:

  1. the valid range / step of each decision variable, taken from the oTree
     game definitions in the original experiment source (not included in this
     export; the ranges themselves are tabulated in the README), and
  2. generic plausibility (e.g. "giving in a dictator game is bounded by the
     endowment and tends to cluster below half of it").

They were NOT estimated from the original sample, so no moment, correlation
or effect size of the real data is reproduced here. Numbers produced by
running the pipeline on this file are a smoke test, not a research result.

Generative model
----------------
Each synthetic participant gets five independent latent factors:

    prosocial   drives cooperation sends, trust send/return, dictator giving,
                UNICEF donations, redistribution toward the poorer party
    risk        drives BRET boxes opened, competitive pay choice, risky
                coordination option
    reasoning   drives level-k depth in the guessing game (deeper = lower guess)
    patience    drives the time-preference allocation
    ability     drives slider-task performance (and so, mildly, the pay choice)

Observed decisions are noisy monotone transforms of these factors, clipped and
rounded onto each game's legal grid. Because one latent factor (`prosocial`)
loads on several games, the file contains a cooperative-phenotype structure
for the ICL prompts and the PCA baseline to pick up. That structure is an
artefact of this generator, chosen so the code paths are exercised.

Realistic messiness is injected on purpose so the validity filter has
something to do:
  * some dyads are left half-empty, which invalidates both members of the pair
  * some participants have one whole app missing

Game ranges (source: the oTree game definitions)
------------------------------------------------
  compete       : 48 sliders, 0-100 scale; pay choice 0 (fixed) / 1 (competitive)
  cooperation   : 4 rounds, endowment 1, multiplier 2, send step 0.1
  coordination  : 4 rounds, "A" (pays 3) or "B" (pays 5 iff matched)
  guessing_game : 4 rounds, groups of 4, guess 0-100 step 0.1,
                  treatment in {"2/3", "1/2", "1/3"}
  games_solo    : time 0-5 | BRET 2x5 = 10 boxes | dictator 0-10 |
                  5 moral decisions (self = 2,4,6,8,10 vs UNICEF 10) |
                  lying report 1-6
  redistribute  : integer 1-9, stores PARTICIPANT B's credits
  trust         : endowment 5, multiplier 3, send step 0.5

Usage
-----
    python make_synthetic_data.py                  # 240 participants, seed 20261019
    python make_synthetic_data.py --n 480 --seed 7
    python make_synthetic_data.py --out other.csv
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

PARTICIPANTS_PER_SESSION = 24          # divisible by both 2 (dyads) and 4 (guessing game)
DYAD_APPS = ["compete", "cooperation", "coordination", "trust", "redistribute_team"]

# Probability that a dyad in a given app is left half-empty (invalidates both members)
P_BROKEN_DYAD = 0.10
# Probability that a participant is missing one whole app
P_MISSING_APP = 0.05

CODE_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


# ── small helpers ─────────────────────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def round_to_step(x, step):
    return round(round(x / step) * step, 10)


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def make_code(rng):
    return "".join(rng.choice(list(CODE_ALPHABET), size=8))


# ── per-participant decision draws ────────────────────────────────────────────

def draw_participant(rng):
    """Return a dict of decision values for one synthetic participant."""
    prosocial = rng.normal()
    risk      = rng.normal()
    reasoning = rng.normal()
    patience  = rng.normal()
    ability   = rng.normal()

    d = {}

    # ── compete: slider task then pay choice ──────────────────────────────────
    d["centered_sliders"] = int(clip(round(22 + 7 * ability + rng.normal() * 3), 0, 48))
    d["num_sliders"] = 48
    p_compete = sigmoid(0.9 * risk + 0.45 * ability - 0.25 * prosocial)
    d["compete_decision"] = int(rng.random() < p_compete)

    # ── cooperation: 4 rounds, send 0.0-1.0 in steps of 0.1, mild decay ───────
    coop_base = sigmoid(0.95 * prosocial) * 0.85
    for r in range(1, 5):
        send = coop_base - 0.045 * (r - 1) + rng.normal() * 0.11
        d[f"cooperate_decision_{r}"] = round_to_step(clip(send, 0.0, 1.0), 0.1)

    # ── coordination: 4 rounds of "A"/"B", sticky across rounds ───────────────
    logit_b = 0.7 * risk + 0.6 * prosocial
    prev_b = 0
    for r in range(1, 5):
        p_b = sigmoid(logit_b + 0.5 * prev_b + rng.normal() * 0.3)
        choice_b = int(rng.random() < p_b)
        d[f"coordinate_decision_{r}"] = "B" if choice_b else "A"
        prev_b = 1 if choice_b else -1

    # ── guessing game: level-k, depth rises with `reasoning` and with rounds ──
    #   stored per round; the group average and winner flag are filled in later
    d["_k1"] = int(clip(round(1.0 + 0.9 * reasoning + rng.normal() * 0.4), 0, 4))
    d["_guess_noise"] = [rng.normal() for _ in range(4)]

    # ── games_solo ────────────────────────────────────────────────────────────
    d["time_decision"] = int(clip(round(5 * sigmoid(0.9 * patience + rng.normal() * 0.4)), 0, 5))

    boxes = round(10 * clip(sigmoid(0.95 * risk) * 0.9 + rng.normal() * 0.08, 0.0, 1.0))
    d["boxes_collected"] = int(clip(boxes, 0, 10))
    d["bomb_row"] = int(rng.integers(1, 3))     # 2 rows
    d["bomb_col"] = int(rng.integers(1, 6))     # 5 cols
    bomb_index = (d["bomb_row"] - 1) * 5 + d["bomb_col"]   # 1..10, order boxes are opened
    d["bomb"] = int(d["boxes_collected"] >= bomb_index)

    give = round(10 * clip(sigmoid(0.9 * prosocial) * 0.75 + rng.normal() * 0.1, 0.0, 1.0))
    d["dictator_decision"] = int(clip(give, 0, 10))

    # moral: self payoff grows 2,4,6,8,10 -> donating gets harder each decision
    for n in range(1, 6):
        donate = (1.0 * prosocial + rng.normal() * 0.55) > 0.35 * (n - 3)
        d[f"moral_decision_{n}"] = int(donate)

    # lying: private die roll, over-reported by the less prosocial
    true_roll = int(rng.integers(1, 7))
    p_honest = sigmoid(0.8 * prosocial + 0.4)
    if rng.random() < p_honest:
        d["lying_decision"] = true_roll
    else:
        d["lying_decision"] = int(rng.integers(true_roll, 7))

    # ── redistribute: stored value is PARTICIPANT B's credits, integer 1-9 ────
    d["redistribute_version"] = "merit" if rng.random() < 0.5 else "no_merit"
    is_merit = 1.0 if d["redistribute_version"] == "merit" else 0.0
    equalize = sigmoid(0.9 * prosocial - 0.6 * is_merit + rng.normal() * 0.3)
    d["redistribute_decision"] = int(clip(round(1 + 8 * equalize), 1, 9))

    # ── redistribute_team: stored value is PARTICIPANT A's credits, 1-9 ───────
    #   This follows what build_prompts.build_redistribute_team_section assumes,
    #   so the demo is self-consistent. Note that `redistribute` above uses the
    #   opposite convention (it stores B's credits), and that there is no oTree
    #   source for redistribute_team to check the direction against. Verify it
    #   before drawing anything from this variable on real data.
    team_equalize = sigmoid(0.9 * prosocial + rng.normal() * 0.35)
    give_a_first = int(clip(round(9 - 8 * team_equalize), 1, 9))
    d["redistribute_decision_first"] = give_a_first
    d["redistribute_decision_second"] = int(
        clip(give_a_first + int(rng.integers(-1, 2)), 1, 9)
    )

    # ── trust: send in steps of 0.5 up to 5; return filled in after matching ──
    p_send = sigmoid(0.8 * prosocial + 0.3 * risk)
    d["trust_send"] = round_to_step(clip(5 * (p_send + rng.normal() * 0.1), 0.0, 5.0), 0.5)
    d["_return_fraction"] = clip(sigmoid(0.95 * prosocial) * 0.55 + rng.normal() * 0.08, 0.0, 1.0)

    # ── filler apps (present only so the validity filter sees them) ───────────
    d["sel_treatment"] = "control" if rng.random() < 0.5 else "affirmation"
    d["self_affirmation_decision"] = int(rng.integers(1, 6))
    d["article_decision"] = int(rng.integers(1, 3))
    d["estimate"] = int(rng.integers(0, 101))

    return d


# ── column schema ─────────────────────────────────────────────────────────────

def build_columns():
    cols = [
        "participant.id_in_session",
        "participant.code",
        "participant.label",
        "session.code",
    ]

    cols += [
        "games_solo.1.player.id_in_group",
        "games_solo.1.player.time_decision",
        "games_solo.1.player.bomb",
        "games_solo.1.player.bomb_row",
        "games_solo.1.player.bomb_col",
        "games_solo.1.player.boxes_collected",
        "games_solo.1.player.dictator_decision",
    ]
    cols += [f"games_solo.1.player.moral_decision_{n}" for n in range(1, 6)]
    cols += [
        "games_solo.1.player.lying_decision",
        "games_solo.1.group.id_in_subsession",
        "games_solo.1.subsession.round_number",
    ]

    cols += [
        "compete.1.player.id_in_group",
        "compete.1.player.centered_sliders",
        "compete.1.player.num_sliders",
        "compete.1.player.compete_decision",
        "compete.1.group.id_in_subsession",
        "compete.1.subsession.round_number",
    ]

    cols += [
        "redistribute.1.player.id_in_group",
        "redistribute.1.player.redistribute_version",
        "redistribute.1.player.redistribute_decision",
        "redistribute.1.group.id_in_subsession",
        "redistribute.1.subsession.round_number",
    ]

    for r in range(1, 5):
        cols += [
            f"cooperation.{r}.player.id_in_group",
            f"cooperation.{r}.player.cooperate_decision",
            f"cooperation.{r}.group.id_in_subsession",
            f"cooperation.{r}.subsession.round_number",
        ]

    for r in range(1, 5):
        cols += [
            f"guessing_game.{r}.player.id_in_group",
            f"guessing_game.{r}.player.treatment",
            f"guessing_game.{r}.player.guess",
            f"guessing_game.{r}.player.is_winner",
            f"guessing_game.{r}.group.id_in_subsession",
            f"guessing_game.{r}.group.avg",
            f"guessing_game.{r}.subsession.round_number",
        ]

    for r in range(1, 5):
        cols += [
            f"coordination.{r}.player.id_in_group",
            f"coordination.{r}.player.coordinate_decision",
            f"coordination.{r}.group.id_in_subsession",
            f"coordination.{r}.subsession.round_number",
        ]

    cols += [
        "redistribute_team.1.player.id_in_group",
        "redistribute_team.1.player.redistribute_decision_first",
        "redistribute_team.1.player.redistribute_decision_second",
        "redistribute_team.1.group.id_in_subsession",
        "redistribute_team.1.group.group_redistribute_decision",
        "redistribute_team.1.subsession.round_number",
    ]

    cols += [
        "trust.1.player.id_in_group",
        "trust.1.player.trust_send",
        "trust.1.player.trust_return",
        "trust.1.group.id_in_subsession",
        "trust.1.subsession.round_number",
    ]

    cols += [
        "sel.1.player.id_in_group",
        "sel.1.player.sel_treatment",
        "sel.1.player.self_affirmation_decision",
        "sel.1.group.id_in_subsession",
        "sel.1.subsession.round_number",
    ]

    cols += [
        "wikipedia_task.1.player.id_in_group",
        "wikipedia_task.1.player.article_decision",
        "wikipedia_task.1.player.estimate",
        "wikipedia_task.1.group.id_in_subsession",
        "wikipedia_task.1.subsession.round_number",
    ]

    return cols


# Player-level columns per app, used to blank out a whole missing app
APP_PLAYER_COLS = {
    "games_solo": [
        "games_solo.1.player.time_decision", "games_solo.1.player.bomb",
        "games_solo.1.player.bomb_row", "games_solo.1.player.bomb_col",
        "games_solo.1.player.boxes_collected", "games_solo.1.player.dictator_decision",
        *[f"games_solo.1.player.moral_decision_{n}" for n in range(1, 6)],
        "games_solo.1.player.lying_decision",
    ],
    "compete": [
        "compete.1.player.centered_sliders", "compete.1.player.num_sliders",
        "compete.1.player.compete_decision",
    ],
    "redistribute": [
        "redistribute.1.player.redistribute_version",
        "redistribute.1.player.redistribute_decision",
    ],
    "cooperation": [f"cooperation.{r}.player.cooperate_decision" for r in range(1, 5)],
    "coordination": [f"coordination.{r}.player.coordinate_decision" for r in range(1, 5)],
    "guessing_game": [
        *[f"guessing_game.{r}.player.treatment" for r in range(1, 5)],
        *[f"guessing_game.{r}.player.guess" for r in range(1, 5)],
        *[f"guessing_game.{r}.player.is_winner" for r in range(1, 5)],
    ],
    "redistribute_team": [
        "redistribute_team.1.player.redistribute_decision_first",
        "redistribute_team.1.player.redistribute_decision_second",
    ],
    "trust": ["trust.1.player.trust_send", "trust.1.player.trust_return"],
}

# Structural columns per app, blanked when a dyad/group is broken
APP_STRUCT_COLS = {
    "compete": ["compete.1.player.id_in_group", "compete.1.group.id_in_subsession",
                "compete.1.subsession.round_number"],
    "cooperation": [c for r in range(1, 5) for c in
                    (f"cooperation.{r}.player.id_in_group",
                     f"cooperation.{r}.group.id_in_subsession",
                     f"cooperation.{r}.subsession.round_number")],
    "coordination": [c for r in range(1, 5) for c in
                     (f"coordination.{r}.player.id_in_group",
                      f"coordination.{r}.group.id_in_subsession",
                      f"coordination.{r}.subsession.round_number")],
    "trust": ["trust.1.player.id_in_group", "trust.1.group.id_in_subsession",
              "trust.1.subsession.round_number"],
    "redistribute_team": ["redistribute_team.1.player.id_in_group",
                          "redistribute_team.1.group.id_in_subsession",
                          "redistribute_team.1.group.group_redistribute_decision",
                          "redistribute_team.1.subsession.round_number"],
    "guessing_game": [c for r in range(1, 5) for c in
                      (f"guessing_game.{r}.player.id_in_group",
                       f"guessing_game.{r}.group.id_in_subsession",
                       f"guessing_game.{r}.group.avg",
                       f"guessing_game.{r}.subsession.round_number")],
}


# ── session assembly ──────────────────────────────────────────────────────────

def build_session(rng, session_code, n_players, start_id):
    """Build one session's worth of rows (list of dicts keyed by column name)."""
    draws = [draw_participant(rng) for _ in range(n_players)]
    codes = [make_code(rng) for _ in range(n_players)]
    rows = [{} for _ in range(n_players)]

    for i, (row, d, code) in enumerate(zip(rows, draws, codes)):
        row["participant.id_in_session"] = i + 1
        row["participant.code"] = code
        row["participant.label"] = f"synthetic_{start_id + i:05d}"
        row["session.code"] = session_code

        # solo apps: each participant is their own group of 1
        for app in ("games_solo", "redistribute", "sel", "wikipedia_task"):
            row[f"{app}.1.player.id_in_group"] = 1
            row[f"{app}.1.group.id_in_subsession"] = i + 1
            row[f"{app}.1.subsession.round_number"] = 1

        row["games_solo.1.player.time_decision"] = d["time_decision"]
        row["games_solo.1.player.bomb"] = d["bomb"]
        row["games_solo.1.player.bomb_row"] = d["bomb_row"]
        row["games_solo.1.player.bomb_col"] = d["bomb_col"]
        row["games_solo.1.player.boxes_collected"] = d["boxes_collected"]
        row["games_solo.1.player.dictator_decision"] = d["dictator_decision"]
        for n in range(1, 6):
            row[f"games_solo.1.player.moral_decision_{n}"] = d[f"moral_decision_{n}"]
        row["games_solo.1.player.lying_decision"] = d["lying_decision"]

        row["redistribute.1.player.redistribute_version"] = d["redistribute_version"]
        row["redistribute.1.player.redistribute_decision"] = d["redistribute_decision"]

        row["sel.1.player.sel_treatment"] = d["sel_treatment"]
        row["sel.1.player.self_affirmation_decision"] = d["self_affirmation_decision"]
        row["wikipedia_task.1.player.article_decision"] = d["article_decision"]
        row["wikipedia_task.1.player.estimate"] = d["estimate"]

        row["compete.1.player.centered_sliders"] = d["centered_sliders"]
        row["compete.1.player.num_sliders"] = d["num_sliders"]
        row["compete.1.player.compete_decision"] = d["compete_decision"]
        for r in range(1, 5):
            row[f"cooperation.{r}.player.cooperate_decision"] = d[f"cooperate_decision_{r}"]
            row[f"coordination.{r}.player.coordinate_decision"] = d[f"coordinate_decision_{r}"]
        row["redistribute_team.1.player.redistribute_decision_first"] = d["redistribute_decision_first"]
        row["redistribute_team.1.player.redistribute_decision_second"] = d["redistribute_decision_second"]
        row["trust.1.player.trust_send"] = d["trust_send"]

    # ── dyadic matching: (0,1), (2,3), ... — one pairing reused across rounds ──
    n_dyads = n_players // 2
    for app in DYAD_APPS:
        rounds = range(1, 5) if app in ("cooperation", "coordination") else [1]
        for g in range(n_dyads):
            a, b = 2 * g, 2 * g + 1
            for r in rounds:
                for idx, id_in_group in ((a, 1), (b, 2)):
                    rows[idx][f"{app}.{r}.player.id_in_group"] = id_in_group
                    rows[idx][f"{app}.{r}.group.id_in_subsession"] = g + 1
                    rows[idx][f"{app}.{r}.subsession.round_number"] = r

    # trust return depends on the partner's send (tripled)
    for g in range(n_dyads):
        a, b = 2 * g, 2 * g + 1
        for me, partner in ((a, b), (b, a)):
            received = rows[partner]["trust.1.player.trust_send"] * 3
            ret = round_to_step(draws[me]["_return_fraction"] * received, 0.5)
            rows[me]["trust.1.player.trust_return"] = round_to_step(clip(ret, 0.0, received), 0.5)

    # team redistribution: the pair's agreed outcome
    for g in range(n_dyads):
        a, b = 2 * g, 2 * g + 1
        first_a = rows[a]["redistribute_team.1.player.redistribute_decision_second"]
        first_b = rows[b]["redistribute_team.1.player.redistribute_decision_second"]
        agreed = int(clip(round((first_a + first_b) / 2), 1, 9))
        for idx in (a, b):
            rows[idx]["redistribute_team.1.group.group_redistribute_decision"] = agreed

    # ── guessing game: groups of 4, level-k guesses, winner nearest f * avg ───
    n_quads = n_players // 4
    for g in range(n_quads):
        members = [4 * g + j for j in range(4)]
        treatment = ["2/3", "1/2", "1/3"][int(rng.integers(0, 3))]
        f = eval_fraction(treatment)
        for r in range(1, 5):
            guesses = {}
            for j, idx in enumerate(members):
                k = min(draws[idx]["_k1"] + (r - 1), 6)      # depth grows with experience
                g_val = 50.0 * (f ** k) + draws[idx]["_guess_noise"][r - 1] * 6.0
                g_val = round(clip(g_val, 0.0, 100.0), 1)
                guesses[idx] = g_val
                rows[idx][f"guessing_game.{r}.player.id_in_group"] = j + 1
                rows[idx][f"guessing_game.{r}.player.treatment"] = treatment
                rows[idx][f"guessing_game.{r}.player.guess"] = g_val
                rows[idx][f"guessing_game.{r}.group.id_in_subsession"] = g + 1
                rows[idx][f"guessing_game.{r}.subsession.round_number"] = r

            avg = sum(guesses.values()) / 4
            target = f * avg
            best = min(guesses, key=lambda i: abs(guesses[i] - target))
            for idx in members:
                rows[idx][f"guessing_game.{r}.group.avg"] = round(avg, 1)
                rows[idx][f"guessing_game.{r}.player.is_winner"] = (idx == best)

    # ── inject messiness ──────────────────────────────────────────────────────
    # (a) broken dyads / quads: blank one member's structural columns, which
    #     leaves an incomplete group and invalidates BOTH members
    for app in DYAD_APPS:
        for g in range(n_dyads):
            if rng.random() < P_BROKEN_DYAD:
                victim = 2 * g + int(rng.integers(0, 2))
                for col in APP_STRUCT_COLS[app]:
                    rows[victim][col] = None
                for col in APP_PLAYER_COLS[app]:
                    rows[victim][col] = None
    for g in range(n_quads):
        if rng.random() < P_BROKEN_DYAD:
            victim = 4 * g + int(rng.integers(0, 4))
            for col in APP_STRUCT_COLS["guessing_game"]:
                rows[victim][col] = None
            for col in APP_PLAYER_COLS["guessing_game"]:
                rows[victim][col] = None

    # (b) a whole app missing for a few participants (solo apps included)
    for row in rows:
        if rng.random() < P_MISSING_APP:
            app = list(APP_PLAYER_COLS)[int(rng.integers(0, len(APP_PLAYER_COLS)))]
            for col in APP_PLAYER_COLS[app]:
                row[col] = None

    return rows


def eval_fraction(treatment):
    num, den = treatment.split("/")
    return int(num) / int(den)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate synthetic BehaviorICL data.")
    ap.add_argument("--n", type=int, default=240,
                    help="number of participants (rounded up to a multiple of 24)")
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--out", default=str(HERE / "synthetic_data.csv"))
    args = ap.parse_args()

    n_sessions = max(1, math.ceil(args.n / PARTICIPANTS_PER_SESSION))
    total = n_sessions * PARTICIPANTS_PER_SESSION
    rng = np.random.default_rng(args.seed)

    columns = build_columns()
    all_rows = []
    for s in range(n_sessions):
        session_code = make_code(rng)
        all_rows.extend(
            build_session(rng, session_code, PARTICIPANTS_PER_SESSION,
                          start_id=s * PARTICIPANTS_PER_SESSION + 1)
        )

    out_path = Path(args.out)
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="raise")
        w.writeheader()
        for row in all_rows:
            w.writerow({c: ("" if row.get(c) is None else row.get(c)) for c in columns})

    print(f"Wrote {len(all_rows)} synthetic participants "
          f"({n_sessions} sessions x {PARTICIPANTS_PER_SESSION}) "
          f"and {len(columns)} columns to {out_path}")
    print(f"Seed: {args.seed}  (re-running with the same seed reproduces the file byte-for-byte)")


if __name__ == "__main__":
    main()
