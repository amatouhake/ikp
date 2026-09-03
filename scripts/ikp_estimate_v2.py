#!/usr/bin/env python3
"""IKP Estimate v2 — gaming-resistant parameter estimation.

v2 hardens the v1 estimator against the two attacks quantified in
ADVERSARIAL_IKP.md, without changing the underlying calibration:

  1. REFUSAL-ROBUST INTERVAL (defends against sandbagging). Instead of a
     single point, v2 reports [point, adjusted] + a confidence tier:
        point    = size implied by answers the model actually got right
                   (refusals scored 0 = genuine ignorance). A FLOOR:
                   sandbagging can only push it down.
        adjusted = refusal-adjusted size, imputing refused probes at the
                   accuracy the model shows on the questions it *does*
                   attempt (missing-at-random assumption; over-corrects for
                   selective refusers, so it is a soft upper reference).
     Wrong answers are never credited — a confident wrong answer is a
     demonstrated failure, not plausible sandbagging. The interval widens
     with the refusal rate, so a heavily refusing (e.g. safety-tuned) model
     is honestly flagged as "could be much larger than it looks" rather than
     silently under-estimated.

  2. HELD-OUT SPLIT (defends against contamination). `--split private`
     scores only the private half of data/probes/split_manifest_v2.json, so
     an operator who memorized the public probe set gains nothing.

Runs live against any OpenAI-compatible endpoint (like v1), or re-scores an
existing v1 `--output` JSON offline with `--from-results` (no API key).

Usage:
  export OPENROUTER_API_KEY=sk-or-...
  python scripts/ikp_estimate_v2.py --model openai/gpt-4.1
  python scripts/ikp_estimate_v2.py --model openai/gpt-4.1 --split private
  python scripts/ikp_estimate_v2.py --from-results runs/gpt-4.1.json
"""

import argparse
import importlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
v1 = importlib.import_module("ikp_estimate")  # reuse query/judge/calibration

PROJECT_ROOT = Path(__file__).parent.parent
SPLIT_FILE = PROJECT_ROOT / "data" / "probes" / "split_manifest_v2.json"
TIERS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]
VALID_VERDICTS = frozenset(("CORRECT", "REFUSAL", "WRONG"))
CALIB_MAX_B = 1600.0  # largest OPEN model in the calibration cohort (deepseek-v4-pro,
#                       1.6T); above this the estimate is extrapolation

# Reuse v1's forward calibration constants so both estimator entry points are
# guaranteed to consume the same effective_params.json artifact.
_SLOPE, _INTERCEPT = v1.ACC_SLOPE, v1.ACC_INTERCEPT


def acc_to_params_B(acc):
    """Invert accuracy = slope·log10(params_B) + intercept."""
    if acc <= _INTERCEPT:
        return 0.0
    return 10 ** ((acc - _INTERCEPT) / _SLOPE)


DIM = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"


# ── Split ──────────────────────────────────────────────────────
def load_split():
    if not SPLIT_FILE.exists():
        return None
    with open(SPLIT_FILE) as split_file:
        return json.load(split_file).get("assignment", {})


def filter_probes(probes, split, which):
    if which == "all":
        return probes
    assignment = load_split()
    if not assignment:
        print(f"  {DIM}No split manifest; run scripts/make_probe_split.py. "
              f"Using all probes.{RESET}")
        return probes
    kept = [p for p in probes if assignment.get(p.get("id")) == which]
    if not kept:
        print(f"  Warning: no probes in split '{which}'. Using all.")
        return probes
    return kept


# ── Robust scoring ─────────────────────────────────────────────
def tier_stats_from_results(results):
    """Aggregate per-tier {correct, wrong, refusal, total} from probe records."""
    stats = defaultdict(lambda: {"correct": 0, "wrong": 0, "refusal": 0, "total": 0})
    for r in results:
        t = r["tier"]
        stats[t]["total"] += 1
        v = r["verdict"]
        if v == "CORRECT":
            stats[t]["correct"] += 1
        elif v == "REFUSAL":
            stats[t]["refusal"] += 1
        elif v == "WRONG":
            stats[t]["wrong"] += 1
        else:
            raise ValueError(f"unknown benchmark verdict: {v!r}")
    return stats


def validate_result_artifact(data):
    """Validate a v1 result artifact before offline re-scoring."""
    if not isinstance(data, dict):
        raise ValueError("results artifact must be a JSON object")
    if "status" in data and data["status"] != "completed":
        raise ValueError(
            f"results artifact status is not completed: {data['status']!r}"
        )

    results = data.get("results", [])
    if not isinstance(results, list):
        raise ValueError("results artifact 'results' must be a list")

    invalid = []
    for index, result in enumerate(results):
        verdict = result.get("verdict") if isinstance(result, dict) else None
        if verdict not in VALID_VERDICTS:
            invalid.append(f"#{index}={verdict!r}")
    if invalid:
        raise ValueError(
            "results artifact contains INFRASTRUCTURE_ERROR or unknown "
            f"verdict(s): {', '.join(invalid)}"
        )
    return results


def robust_estimate(stats):
    """Return the refusal-robust interval and diagnostics from tier stats.

    lower = observed accuracy (refusals scored 0 = genuine ignorance).
    upper = refusal-adjusted accuracy: within each tier, impute refused
            probes at the model's *attempted* accuracy correct/(correct+wrong)
            — i.e. "what would this look like if the refusals had been
            answered as well as the questions it did attempt?" This assumes
            refusals are representative (missing-at-random); a model that
            refuses *only what it knows* could exceed even this, which is
            why the point estimate is reported as a floor.
    """
    # Fixed 7-tier denominator (missing tier scores 0) — the repo's convention.
    obs_sum = adj_sum = 0.0
    tot = ref = 0
    for t in TIERS:
        s = stats.get(t)
        if not s or not s["total"]:
            continue
        obs_sum += s["correct"] / s["total"]
        attempted = s["correct"] + s["wrong"]
        adj_sum += s["correct"] / attempted if attempted > 0 else s["correct"] / s["total"]
        tot += s["total"]
        ref += s["refusal"]
    acc_lo = obs_sum / len(TIERS)
    acc_hi = adj_sum / len(TIERS)
    return {
        "acc_observed": acc_lo,
        "acc_adjusted": acc_hi,
        "lower_B": acc_to_params_B(acc_lo),
        "upper_B": acc_to_params_B(acc_hi),
        "refusal_rate": ref / tot if tot else 0.0,
        "n": tot,
    }


# ── Display ────────────────────────────────────────────────────
def clamp_label(b):
    """Format a size, marking anything past the calibration ceiling."""
    if b > CALIB_MAX_B:
        return f">{v1.format_params(CALIB_MAX_B)}"
    return v1.format_params(b)


def reliability(refusal_rate):
    """Confidence tier driven by refusal rate (the sandbagging surface)."""
    if refusal_rate < 0.10:
        return "Reliable", "answers rather than refusing; estimate well-constrained"
    if refusal_rate < 0.30:
        return "Caution", "refusals may deflate the point estimate"
    return "Low confidence", "heavy refusal — point estimate is a loose floor only"


def show(model_name, est, n_probes, split):
    lo, hi = est["lower_B"], est["upper_B"]
    lo_s, hi_s = clamp_label(lo), clamp_label(hi)
    tier, why = reliability(est["refusal_rate"])

    probes_line = f"{n_probes}  (split: {split})"
    refusal_line = f"{est['refusal_rate'] * 100:.1f}%   [{tier}]"
    print()
    print(f"  ╔══════════════════════════════════════════════════════════╗")
    print(f"  ║  IKP v2 — gaming-resistant estimate                      ║")
    print(f"  ╠══════════════════════════════════════════════════════════╣")
    print(f"  ║  Model:    {model_name[:44]:44s}  ║")
    print(f"  ║  Probes:   {probes_line[:44]:44s}  ║")
    print(f"  ║  Estimate: {lo_s + ' (point / floor)'[:44]:44s}  ║")
    print(f"  ║  Refusals: {refusal_line[:44]:44s}  ║")
    print(f"  ╚══════════════════════════════════════════════════════════╝")
    print()
    print(f"  Point  {lo_s:>7s}  — from demonstrated correct answers; refusals scored 0.")
    print(f"                    This is a FLOOR: sandbagging can only push it down.")
    print(f"  Adj.   {hi_s:>7s}  — refusal-adjusted (refused probes imputed at the rate the")
    print(f"                    model scores on what it attempts; assumes refusals are")
    print(f"                    representative — over-corrects for selective refusers).")
    print(f"  Confidence: {tier} — {why}.")
    print()
    ratio = (hi / lo) if lo > 0 else float("inf")
    if ratio >= 1.5:
        print(f"  {BOLD}⚠ Wide interval ({ratio:.1f}× span){RESET}: high refusal rate makes this "
              f"model's size ambiguous. It may be substantially larger than the")
        print(f"    lower bound if the refusals are strategic rather than genuine. "
              f"Treat the point estimate as a floor, not a measurement.")
    else:
        print(f"  {DIM}Tight interval: the model answers rather than refusing, so the "
              f"estimate is well-constrained.{RESET}")
    print()


# ── Live run (thin wrapper over v1's query+judge) ──────────────
def run_live(args):
    with open(v1.PROBE_FILE) as probe_file:
        probes = json.load(probe_file)
    split = load_split()
    probes = filter_probes(probes, split, args.split)

    api_style = getattr(args, "api_style", "chat")
    reasoning_effort = getattr(args, "reasoning_effort", "default")
    is_thinking = getattr(args, "thinking", False)
    explicit_api_key = getattr(args, "api_key", None)
    api_key = v1.resolve_target_api_key(args.api_base, explicit_api_key)
    if not api_key and (v1._is_openrouter_base(args.api_base)
                        or v1._is_opencode_go_base(args.api_base)):
        env_name = ("OPENCODE_GO_API_KEY"
                    if v1._is_opencode_go_base(args.api_base)
                    else "OPENROUTER_API_KEY")
        print(f"Error: {env_name} not set for the target.")
        sys.exit(1)

    # The fixed judge credential is independent of the target credential.
    judge_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not judge_key:
        print("Error: OPENROUTER_API_KEY needed for the judge.")
        sys.exit(1)

    query_fn = v1.make_query_fn(
        args.api_base,
        api_key,
        args.model,
        is_thinking=is_thinking,
        api_style=api_style,
        reasoning_effort=reasoning_effort,
    )
    judge_fn = v1.make_judge_fn(judge_key)

    print(f"\n  Scoring {args.model} on {len(probes)} probes (split: {args.split})...")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = []

    def eval_one(p):
        try:
            resp = query_fn(p["question"])
        except v1.TargetAPIError as exc:
            return {
                "probe_id": p.get("id", ""),
                "tier": p["tier"],
                "verdict": "INFRASTRUCTURE_ERROR",
                "error_type": "target_api",
                "error": str(exc),
            }

        try:
            verdict = judge_fn(p["question"], p["answer"], resp)
        except v1.JudgeAPIError as exc:
            return {
                "probe_id": p.get("id", ""),
                "tier": p["tier"],
                "verdict": "INFRASTRUCTURE_ERROR",
                "error_type": "judge_api",
                "error": str(exc),
            }
        return {"tier": p["tier"], "verdict": verdict}

    workers = 1 if args.sequential else args.workers
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(eval_one, p) for p in probes]
        for f in as_completed(futures):
            results.append(f.result())

    infrastructure_errors = [
        result for result in results
        if result.get("verdict") == "INFRASTRUCTURE_ERROR"
    ]
    if infrastructure_errors:
        print(
            f"Error: {len(infrastructure_errors)} infrastructure error(s); "
            "no estimate was computed or scored.",
            file=sys.stderr,
        )
        for result in infrastructure_errors[:3]:
            print(
                f"  {result['probe_id']}: {result['error_type']} - "
                f"{result['error']}",
                file=sys.stderr,
            )
        sys.exit(1)

    est = robust_estimate(tier_stats_from_results(results))
    show(args.model, est, len(results), args.split)


# ── Offline re-score of a v1 results file ──────────────────────
def run_from_results(args):
    with open(args.from_results) as results_file:
        data = json.load(results_file)
    try:
        results = validate_result_artifact(data)
    except ValueError as exc:
        print(f"Error: refusing to re-score {args.from_results}: {exc}",
              file=sys.stderr)
        sys.exit(1)
    model_name = data.get("model", Path(args.from_results).stem)
    if args.split != "all":
        assignment = load_split() or {}
        results = [r for r in results if assignment.get(r.get("probe_id")) == args.split]
    if not results:
        print("  No probe records found (need a v1 --output JSON with 'results').")
        sys.exit(1)
    est = robust_estimate(tier_stats_from_results(results))
    show(model_name, est, len(results), args.split)


def main():
    ap = argparse.ArgumentParser(
        description="IKP v2 — gaming-resistant parameter estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", "-m", help="Target model ID")
    ap.add_argument("--api-base", default="https://openrouter.ai/api/v1")
    ap.add_argument("--api-key")
    ap.add_argument("--api-style", choices=v1.API_STYLES, default="chat",
                    help="Target API style: chat (default) or responses")
    ap.add_argument("--thinking", action="store_true",
                    help="Backward-compatible alias for --reasoning-effort medium")
    ap.add_argument("--reasoning-effort", choices=v1.REASONING_EFFORTS,
                    default="default",
                    help="Target reasoning override: default, low, medium, high, xhigh")
    ap.add_argument("--split", choices=["all", "public", "private"], default="private",
                    help="Which probe split to score (default: private — contamination-resistant)")
    ap.add_argument("--workers", "-w", type=int, default=16)
    ap.add_argument("--sequential", "-s", action="store_true")
    ap.add_argument("--from-results", metavar="FILE",
                    help="Re-score a v1 --output JSON offline instead of querying")
    args = ap.parse_args()

    if args.from_results:
        run_from_results(args)
    elif args.model:
        run_live(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
