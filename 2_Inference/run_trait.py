"""
Trait-Agnostic Inference Runner — BehaviorICL

Runs all three methods (ICL, zero-shot, other-student ICL) for a given trait.
Reads prompts from 1_Prompts/{trait}/prompts.jsonl
Writes results to 2_Inference/{trait}/

Usage:
    python run_trait.py <trait_name>
    python run_trait.py competitiveness
    python run_trait.py risk_tolerance

Output files (all in 2_Inference/{trait}/):
    icl_results.jsonl
    zero_shot_results.jsonl
    other_student_icl_results.jsonl

Each line:
    {participant_code, pred_<key>, true_<col1>[, true_<col2>...], partner_send (trust only),
     model, temperature, seed}
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "1_Prompts"))

from trait_configs import TRAIT_CONFIGS, TraitConfig
from client import call_model, call_model_trust, run_settings


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_prompts(trait_name: str) -> list[dict]:
    path = ROOT / "1_Prompts" / trait_name / "prompts.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Prompts not found: {path}\n"
            f"Run: python 1_Prompts/build_prompts.py {trait_name}"
        )
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_done(out_path: Path) -> set:
    """Return set of participant_codes already written to output file."""
    if not out_path.exists():
        return set()
    done = set()
    for line in out_path.read_text().splitlines():
        if line.strip():
            try:
                done.add(json.loads(line)["participant_code"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def get_true_values(record: dict, config: TraitConfig) -> dict:
    """Extract true target values from a prompt record."""
    result = {}
    for col in config.target_cols:
        key = col.replace(".", "_")
        result[f"true_{key}"] = record.get(key)
    return result


def infer(
    prompts: list[dict],
    config: TraitConfig,
    out_path: Path,
    make_user_prompt,
    method_label: str,
):
    """
    Generic inference loop with checkpoint support.
    make_user_prompt(record) -> str
    """
    done = load_done(out_path)
    todo = [r for r in prompts if r["participant_code"] not in done]
    print(f"  {method_label}: {len(done)} done, {len(todo)} remaining")

    with out_path.open("a") as f:
        for i, record in enumerate(todo):
            code = record["participant_code"]
            user_prompt = make_user_prompt(record)

            if config.name == "trust":
                result = call_model_trust(user_prompt)
                out = {
                    "participant_code": code,
                    "pred_trust_send": result["trust_send"],
                    "pred_trust_return": result["trust_return"],
                    "true_trust_send": record.get("trust_1_player_trust_send"),
                    "true_trust_return": record.get("trust_1_player_trust_return"),
                    "partner_send": record.get("partner_send"),
                }
            else:
                result = call_model(
                    user_prompt,
                    prediction_key=config.prediction_key,
                    prediction_range=config.prediction_range,
                )
                out = {
                    "participant_code": code,
                    f"pred_{config.prediction_key}": result[config.prediction_key],
                    **get_true_values(record, config),
                }

            out.update(run_settings())
            f.write(json.dumps(out) + "\n")
            f.flush()

            if (i + 1) % 100 == 0:
                print(f"    {method_label}: {i + 1}/{len(todo)}")

    print(f"  {method_label}: complete → {out_path}")


# ── Zero-shot prompt ──────────────────────────────────────────────────────────

def make_zero_shot_prompt(record: dict, config: TraitConfig) -> str:
    """
    Zero-shot: just the game description + prediction question, no behavioral context.
    """
    lines = [
        f"A student is about to play the {config.label} task. Predict their behavior.\n"
    ]
    lines.append("---")

    if config.name == "trust":
        partner_send = record.get("partner_send", 2.5)
        received = partner_send * 3
        lines.append(
            config.game_description.format(partner_send=partner_send, received=received)
        )
        lines.append("")
        lines.append(config.prediction_question.format(received=received))
    elif config.name == "depth_of_reasoning":
        treatment = record.get("guessing_game_1_player_treatment", "2/3")
        lines.append(config.game_description.format(treatment=treatment))
        lines.append("")
        lines.append(config.prediction_question)
    elif config.name == "tolerance_for_inequality":
        version = record.get("redistribute_1_player_redistribute_version")
        if version == "merit":
            context_str = (
                "Participant A earned 9 credits (better slider performance) and "
                "Participant B earned 1 credit."
            )
        elif version == "no_merit":
            context_str = (
                "Participant A was randomly assigned 9 credits and "
                "Participant B was randomly assigned 1 credit."
            )
        else:
            context_str = "Allocation context unknown."
        lines.append(config.game_description.format(context=context_str))
        lines.append("")
        lines.append(config.prediction_question)
    else:
        lines.append(config.game_description)
        lines.append("")
        lines.append(config.prediction_question)

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(trait_name: str):
    if trait_name not in TRAIT_CONFIGS:
        print(f"Unknown trait: {trait_name}")
        print(f"Available: {list(TRAIT_CONFIGS.keys())}")
        sys.exit(1)

    config = TRAIT_CONFIGS[trait_name]
    prompts = load_prompts(trait_name)
    print(f"Loaded {len(prompts)} prompts for trait: {trait_name}")

    out_dir = ROOT / "2_Inference" / trait_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)

    # ── Method 1: ICL — use participant's own behavioral history ──────────────
    print("\n[1/3] ICL")
    infer(
        prompts,
        config,
        out_dir / "icl_results.jsonl",
        make_user_prompt=lambda r: r["prompt"],
        method_label="ICL",
    )

    # ── Method 2: Zero-shot — game description only ───────────────────────────
    print("\n[2/3] Zero-shot")
    infer(
        prompts,
        config,
        out_dir / "zero_shot_results.jsonl",
        make_user_prompt=lambda r: make_zero_shot_prompt(r, config),
        method_label="Zero-shot",
    )

    # ── Method 3: Other-student ICL — random other participant's history ──────
    print("\n[3/3] Other-student ICL")
    codes = [r["participant_code"] for r in prompts]
    prompt_by_code = {r["participant_code"]: r["prompt"] for r in prompts}

    def make_other_student_prompt(record: dict) -> str:
        code = record["participant_code"]
        other_codes = [c for c in codes if c != code]
        other_code = rng.choice(other_codes)
        return prompt_by_code[other_code]

    infer(
        prompts,
        config,
        out_dir / "other_student_icl_results.jsonl",
        make_user_prompt=make_other_student_prompt,
        method_label="Other-student ICL",
    )

    print(f"\nDone. Results in: {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_trait.py <trait_name>")
        print(f"Available traits: {list(TRAIT_CONFIGS.keys())}")
        sys.exit(1)
    main(sys.argv[1])
