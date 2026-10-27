"""
Range check for synthetic_data.csv
==================================

Asserts that every synthetic decision value falls inside the legal range and
onto the legal grid defined by the oTree game definitions (tabulated in the
README), and that the dyadic structure is coherent.

    python check_synthetic_data.py                  # checks ./synthetic_data.csv
    python check_synthetic_data.py other_file.csv

Exits non-zero and lists every violation if anything is out of range. Run it
after editing the generator.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent


def num(v):
    """Parse a CSV cell to float, or None if blank."""
    if v is None or v.strip() == "":
        return None
    return float(v)


def is_step(x, step):
    return abs(round(x / step) * step - x) < 1e-6


# (column, predicate, description). Predicates see a non-None float.
CHECKS = [
    # compete — 48 sliders on a 0-100 scale, binary pay choice
    ("compete.1.player.compete_decision", lambda x: x in (0.0, 1.0), "0 or 1"),
    ("compete.1.player.centered_sliders", lambda x: 0 <= x <= 48 and x == int(x), "integer 0-48"),
    ("compete.1.player.num_sliders", lambda x: x == 48, "exactly 48"),

    # guessing game — guess 0-100, step 0.1
    *[(f"guessing_game.{r}.player.guess",
       lambda x: 0 <= x <= 100 and is_step(x, 0.1), "0-100, step 0.1") for r in range(1, 5)],

    # cooperation — endowment 1, step 0.1
    *[(f"cooperation.{r}.player.cooperate_decision",
       lambda x: 0 <= x <= 1 and is_step(x, 0.1), "0.0-1.0, step 0.1") for r in range(1, 5)],

    # games_solo
    ("games_solo.1.player.time_decision", lambda x: 0 <= x <= 5 and x == int(x), "integer 0-5"),
    ("games_solo.1.player.boxes_collected", lambda x: 0 <= x <= 10 and x == int(x), "integer 0-10"),
    ("games_solo.1.player.bomb", lambda x: x in (0.0, 1.0), "0 or 1"),
    ("games_solo.1.player.bomb_row", lambda x: x in (1.0, 2.0), "1 or 2 (2 rows)"),
    ("games_solo.1.player.bomb_col", lambda x: x in (1.0, 2.0, 3.0, 4.0, 5.0), "1-5 (5 cols)"),
    ("games_solo.1.player.dictator_decision", lambda x: 0 <= x <= 10 and x == int(x), "integer 0-10"),
    *[(f"games_solo.1.player.moral_decision_{n}",
       lambda x: x in (0.0, 1.0), "0 or 1") for n in range(1, 6)],
    ("games_solo.1.player.lying_decision", lambda x: x in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0), "die face 1-6"),

    # redistribute — integer 1-9 out of 10
    ("redistribute.1.player.redistribute_decision", lambda x: 1 <= x <= 9 and x == int(x), "integer 1-9"),
    ("redistribute_team.1.player.redistribute_decision_first", lambda x: 1 <= x <= 9 and x == int(x), "integer 1-9"),
    ("redistribute_team.1.player.redistribute_decision_second", lambda x: 1 <= x <= 9 and x == int(x), "integer 1-9"),
    ("redistribute_team.1.group.group_redistribute_decision", lambda x: 1 <= x <= 9 and x == int(x), "integer 1-9"),

    # trust — endowment 5, step 0.5
    ("trust.1.player.trust_send", lambda x: 0 <= x <= 5 and is_step(x, 0.5), "0-5, step 0.5"),
]

CATEGORICAL = [
    ("redistribute.1.player.redistribute_version", {"merit", "no_merit"}),
    *[(f"coordination.{r}.player.coordinate_decision", {"A", "B"}) for r in range(1, 5)],
    *[(f"guessing_game.{r}.player.treatment", {"1/3", "1/2", "2/3"}) for r in range(1, 5)],
]


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "synthetic_data.csv"
    rows = list(csv.DictReader(path.open(newline="")))
    print(f"Checking {len(rows)} rows of {path.name}\n")

    problems = []
    checked = 0

    for i, row in enumerate(rows, start=2):   # line 1 is the header
        for col, ok, desc in CHECKS:
            if col not in row:
                problems.append(f"line {i}: column {col} missing from file")
                continue
            v = num(row[col])
            if v is None:
                continue
            checked += 1
            if not ok(v):
                problems.append(f"line {i}: {col} = {row[col]!r}, expected {desc}")

        for col, allowed in CATEGORICAL:
            raw = (row.get(col) or "").strip()
            if raw == "":
                continue
            checked += 1
            if raw not in allowed:
                problems.append(f"line {i}: {col} = {raw!r}, expected one of {sorted(allowed)}")

    # trust return must not exceed 3x the partner's send
    dyads = defaultdict(list)
    for i, row in enumerate(rows, start=2):
        gid = (row.get("session.code"), row.get("trust.1.group.id_in_subsession"))
        if gid[1]:
            dyads[gid].append((i, row))
    pairs_checked = 0
    for gid, members in dyads.items():
        if len(members) != 2:
            continue
        for (i, me), (_, partner) in (members, members[::-1]):
            send_p = num(partner["trust.1.player.trust_send"])
            ret_me = num(me["trust.1.player.trust_return"])
            if send_p is None or ret_me is None:
                continue
            pairs_checked += 1
            cap = send_p * 3
            if not (0 <= ret_me <= cap + 1e-9):
                problems.append(
                    f"line {i}: trust_return = {ret_me} exceeds 3 x partner send ({cap})")
            if not is_step(ret_me, 0.5):
                problems.append(f"line {i}: trust_return = {ret_me} is not a multiple of 0.5")

    # participant codes must be unique
    codes = [r["participant.code"] for r in rows]
    if len(set(codes)) != len(codes):
        problems.append(f"participant.code is not unique ({len(codes) - len(set(codes))} duplicates)")

    print(f"  {checked:,} value checks")
    print(f"  {pairs_checked:,} trust-dyad consistency checks")
    print(f"  {len(set(codes)):,} unique participant codes")

    if problems:
        print(f"\nFAIL — {len(problems)} problem(s):")
        for p in problems[:50]:
            print("  ", p)
        if len(problems) > 50:
            print(f"   ... and {len(problems) - 50} more")
        sys.exit(1)

    print("\nPASS — every value is inside its game's legal range and on its legal grid.")


if __name__ == "__main__":
    main()
