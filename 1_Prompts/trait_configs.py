"""
Trait configuration definitions for all 9 behavioral traits.

Each TraitConfig specifies:
  - Which app to hold out as the prediction target
  - Which columns to predict
  - Which games_solo tasks to exclude from context (for partial hold-outs)
  - The zero-shot game description and prediction question
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TraitConfig:
    name: str                       # folder name & identifier  e.g. "competitiveness"
    label: str                      # display name              e.g. "Competitiveness"
    target_app: str                 # app to hold out           e.g. "compete"
    target_cols: List[str]          # raw CSV columns to evaluate against
    prediction_key: str             # JSON key in model output  e.g. "prediction"
    prediction_range: str           # human-readable constraint e.g. "0 or 1"
    game_description: str           # description of held-out game (for both ICL target block and zero-shot)
    prediction_question: str        # what the model is asked to predict
    games_solo_exclude: List[str]   # tasks to skip in games_solo context section
                                    # options: "time", "bret", "dictator", "moral", "lying"
    include_trust_context: bool     # True for all non-trust traits


TRAIT_CONFIGS = {

    "trust": TraitConfig(
        name="trust",
        label="Trust",
        target_app="trust",
        target_cols=["trust.1.player.trust_send", "trust.1.player.trust_return"],
        prediction_key="prediction",   # trust uses custom two-key output; handled separately
        prediction_range="trust_send: 0–5 (steps of 0.5); trust_return: 0 to partner_send×3",
        game_description=(
            "**Trust Game**\n"
            "Each student plays both roles simultaneously:\n"
            "  • As Sender: receives 5 credits, decides how much to send (0–5, steps of 0.5). "
            "Amount sent is tripled before reaching the receiver.\n"
            "  • As Receiver: their partner sent {partner_send:.1f} credits, so this student "
            "received {received:.1f} credits. They decide how much to return (0 to {received:.1f} credits)."
        ),
        prediction_question=(
            "Based on the behavioral history above, predict:\n"
            "1. How much does this student send as Sender? (0 to 5, steps of 0.5)\n"
            "2. How much does this student return as Receiver? (0 to {received:.1f})"
        ),
        games_solo_exclude=[],
        include_trust_context=False,
    ),

    "competitiveness": TraitConfig(
        name="competitiveness",
        label="Competitiveness",
        target_app="compete",
        target_cols=["compete.1.player.compete_decision"],
        prediction_key="prediction",
        prediction_range="0 (chose safe/fixed pay) or 1 (chose competitive pay)",
        game_description=(
            "**Competition Task (Slider Task)**\n"
            "Students completed a slider task (centering 48 sliders in 2 minutes) then chose how to be paid:\n"
            "  • Option A (safe): 0.20 credits per correct slider regardless of partner's performance\n"
            "  • Option B (competitive): 0.50 credits per correct slider only if they outperformed "
            "their partner; 0 otherwise"
        ),
        prediction_question=(
            "Based on the behavioral history above, predict:\n"
            "Does this student choose Option A (safe) or Option B (competitive)?\n"
            "Answer 0 for Option A, 1 for Option B."
        ),
        games_solo_exclude=[],
        include_trust_context=True,
    ),

    "cooperation": TraitConfig(
        name="cooperation",
        label="Cooperation",
        target_app="cooperation",
        target_cols=[
            "cooperation.1.player.cooperate_decision",
            "cooperation.2.player.cooperate_decision",
            "cooperation.3.player.cooperate_decision",
            "cooperation.4.player.cooperate_decision",
        ],
        prediction_key="prediction",
        prediction_range="mean send across 4 rounds, 0.0 to 1.0",
        game_description=(
            "**Mutual Aid Game (4 rounds)**\n"
            "Each round, students received 1 credit and chose how much to send to a partner "
            "(0–1, steps of 0.1). Whatever was sent was doubled before reaching the partner."
        ),
        prediction_question=(
            "Based on the behavioral history above, predict:\n"
            "On average across 4 rounds, how much does this student send to their partner? "
            "(a number from 0.0 to 1.0)"
        ),
        games_solo_exclude=[],
        include_trust_context=True,
    ),

    "coordination": TraitConfig(
        name="coordination",
        label="Coordination",
        target_app="coordination",
        target_cols=[
            "coordination.1.player.coordinate_decision",
            "coordination.2.player.coordinate_decision",
            "coordination.3.player.coordinate_decision",
            "coordination.4.player.coordinate_decision",
        ],
        prediction_key="prediction",
        prediction_range="proportion of rounds chose B: 0, 0.25, 0.5, 0.75, or 1.0",
        game_description=(
            "**Coordination Game (4 rounds)**\n"
            "Each round, students chose Option A or Option B when paired with a partner:\n"
            "  • Option A: always earns 3 credits (safe)\n"
            "  • Option B: earns 5 credits only if partner also chose B; earns 0 otherwise (risky)"
        ),
        prediction_question=(
            "Based on the behavioral history above, predict:\n"
            "In what proportion of the 4 rounds does this student choose Option B? "
            "(answer: 0, 0.25, 0.5, 0.75, or 1.0)"
        ),
        games_solo_exclude=[],
        include_trust_context=True,
    ),

    "depth_of_reasoning": TraitConfig(
        name="depth_of_reasoning",
        label="Depth of Reasoning",
        target_app="guessing_game",
        target_cols=["guessing_game.1.player.guess", "guessing_game.1.player.treatment"],
        prediction_key="prediction",
        prediction_range="round 1 guess, 0 to 100",
        game_description=(
            "**Guessing Game / Beauty Contest (4 rounds)**\n"
            "Each round, students guessed a number from 0–100. The student closest to "
            "{treatment} of the group average won 6 credits."
        ),
        prediction_question=(
            "Based on the behavioral history above, predict:\n"
            "What number does this student guess in round 1? (0 to 100)"
        ),
        games_solo_exclude=[],
        include_trust_context=True,
    ),

    "tolerance_for_inequality": TraitConfig(
        name="tolerance_for_inequality",
        label="Tolerance for Inequality",
        target_app="redistribute",
        target_cols=["redistribute.1.player.redistribute_decision"],
        prediction_key="prediction",
        prediction_range="credits given to Participant A, integer 1 to 9 (B receives 10 minus that)",
        game_description=(
            "**Redistribution Game**\n"
            "A neutral third party decided how to redistribute 10 credits between two other "
            "students (A and B). {context}\n"
            "The decision is how many credits to give Participant A (1–9); "
            "Participant B receives the remainder. "
            "Choosing 5 = equal split; choosing 9 = keep the original allocation (A keeps 9)."
        ),
        prediction_question=(
            "Based on the behavioral history above, predict:\n"
            "How many credits does this student give to Participant A? (integer from 1 to 9)"
        ),
        games_solo_exclude=[],
        include_trust_context=True,
    ),

    "risk_tolerance": TraitConfig(
        name="risk_tolerance",
        label="Risk Tolerance",
        target_app="games_solo",   # partial hold-out
        target_cols=["games_solo.1.player.boxes_collected"],
        prediction_key="prediction",
        prediction_range="number of boxes opened, 0 to 10",
        game_description=(
            "**Risk Task (BRET — Bomb Risk Elicitation Task)**\n"
            "Students faced a 2×5 grid of 10 boxes. One box hid a bomb. "
            "Students decided how many boxes to open: each opened box earns 1 credit, "
            "but opening the bomb-box means earning nothing. "
            "More boxes opened = higher risk tolerance."
        ),
        prediction_question=(
            "Based on the behavioral history above, predict:\n"
            "How many boxes does this student open? (0 to 10)"
        ),
        games_solo_exclude=["bret"],
        include_trust_context=True,
    ),

    "altruism": TraitConfig(
        name="altruism",
        label="Altruism",
        target_app="games_solo",   # partial hold-out
        target_cols=["games_solo.1.player.dictator_decision"],
        prediction_key="prediction",
        prediction_range="credits given to another student, 0 to 10",
        game_description=(
            "**Dictator Game**\n"
            "Students received 10 credits and decided how much to give to an anonymous "
            "classmate. They keep the rest. There is no possibility of the recipient "
            "reciprocating — this is a pure measure of unconditional generosity."
        ),
        prediction_question=(
            "Based on the behavioral history above, predict:\n"
            "Out of 10 credits, how much does this student give to another student? (0 to 10)"
        ),
        games_solo_exclude=["dictator"],
        include_trust_context=True,
    ),

    "morality": TraitConfig(
        name="morality",
        label="Morality",
        target_app="games_solo",   # partial hold-out
        target_cols=[f"games_solo.1.player.moral_decision_{n}" for n in range(1, 6)],
        prediction_key="prediction",
        prediction_range="number of UNICEF donations out of 5 decisions, 0 to 5",
        game_description=(
            "**Moral Preferences (5 decisions)**\n"
            "For each of 5 decisions, students chose between keeping credits for themselves "
            "or donating 10 credits to UNICEF:\n"
            "  Decision 1: self=2 credits vs UNICEF\n"
            "  Decision 2: self=4 credits vs UNICEF\n"
            "  Decision 3: self=6 credits vs UNICEF\n"
            "  Decision 4: self=8 credits vs UNICEF\n"
            "  Decision 5: self=10 credits vs UNICEF"
        ),
        prediction_question=(
            "Based on the behavioral history above, predict:\n"
            "In how many of the 5 decisions does this student choose to donate to UNICEF? (0 to 5)"
        ),
        games_solo_exclude=["moral"],
        include_trust_context=True,
    ),
}
