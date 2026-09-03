#!/usr/bin/env python3
"""IKP Budget — estimate the dollar cost of an IKP run *before* spending a token.

Scoring a model with IKP makes two API calls per probe: one to the target
model and one to the Gemini judge. This tool multiplies the measured
per-probe token footprint of the benchmark by live (or cached) OpenRouter
prices, so you can size a run to your wallet instead of finding out after
the fact.

Usage:
  # Cost of a full 1,400-probe run against one model
  python scripts/ikp_budget.py --model openai/gpt-4.1

  # A thinking model, quick 200-probe sample
  python scripts/ikp_budget.py --model anthropic/claude-opus-4.7 --thinking --sample 200

  # "I have $10 — what can I run?"
  python scripts/ikp_budget.py --model openai/gpt-4.1 --budget 10

  # Compare a handful of common models at a glance
  python scripts/ikp_budget.py --list

  # Force offline mode (use the built-in price table, don't call OpenRouter)
  python scripts/ikp_budget.py --model openai/gpt-4.1 --offline

Prices are fetched live from OpenRouter's public /models endpoint (no API
key required). If that call fails, a built-in snapshot table is used and
the output is clearly marked as an estimate.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import httpx
except ImportError:  # httpx is already an IKP dependency, but degrade gracefully
    httpx = None

# ── Constants ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
PROBE_FILE = PROJECT_ROOT / "data" / "probes" / "final_probe_set_v8.json"
JUDGE_MODEL = "google/gemini-3-flash-preview"  # must match scripts/ikp_estimate.py
FULL_PROBE_COUNT = 1400

# Per-probe token footprint of the benchmark.
#
# These are grounded in the actual probe set (data/probes/final_probe_set_v8.json)
# and the exact prompts in scripts/ikp_estimate.py:
#   - system message ≈ 22 tokens, question ≈ 15 tokens (mean 57 chars),
#     plus chat-format overhead → ~60 target-input tokens.
#   - a concise factual answer is short; we budget 40 output tokens to be safe.
#   - the judge prompt embeds the question, the gold answer and the (≤500-char)
#     model response inside a fixed rubric → ~320 input tokens; it replies with
#     one word plus low-effort reasoning → ~40 output tokens.
# Everything is a token count so it scales cleanly with any price table.
TOK = {
    "target_in": 60,        # system + question + chat overhead
    "target_out": 40,       # concise factual answer
    "target_out_think": 900,  # + medium-effort reasoning tokens (--thinking)
    "judge_in": 320,        # rubric + question + gold + truncated response
    "judge_out": 40,        # one verdict word + low-effort reasoning
}

# Built-in price snapshot (USD per 1,000,000 tokens), used when OpenRouter
# is unreachable. Rounded early-2026 OpenRouter list prices; live fetch
# overrides these whenever it succeeds. (in, out) = (prompt, completion).
PRICE_TABLE = {
    "openai/gpt-4.1":               (2.00, 8.00),
    "openai/gpt-4.1-mini":          (0.40, 1.60),
    "openai/gpt-5":                 (1.25, 10.00),
    "openai/gpt-5-mini":            (0.25, 2.00),
    "anthropic/claude-opus-4.7":    (15.00, 75.00),
    "anthropic/claude-sonnet-4.5":  (3.00, 15.00),
    "anthropic/claude-haiku-4.5":   (1.00, 5.00),
    "google/gemini-3-pro-preview":  (1.25, 10.00),
    "google/gemini-3-flash-preview": (0.30, 2.50),
    "meta-llama/llama-3.3-70b-instruct": (0.13, 0.40),
    "meta-llama/llama-3.1-8b-instruct":  (0.02, 0.03),
    "deepseek/deepseek-chat":       (0.27, 1.10),
    "qwen/qwen-2.5-72b-instruct":   (0.12, 0.39),
    "mistralai/mistral-small":      (0.20, 0.60),
}

# Models shown by --list (a spread of price points).
LIST_MODELS = [
    "meta-llama/llama-3.1-8b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    "deepseek/deepseek-chat",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1",
    "openai/gpt-5",
    "google/gemini-3-pro-preview",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-opus-4.7",
]

DIM = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"


# ── Pricing ────────────────────────────────────────────────────
def fetch_live_prices(offline: bool):
    """Return {model_id: (usd_per_mtok_in, usd_per_mtok_out)} from OpenRouter.

    Returns (prices, source) where source is 'live' or 'offline'. On any
    failure we fall back to the built-in snapshot table.
    """
    if offline or httpx is None:
        return dict(PRICE_TABLE), "offline"
    try:
        r = httpx.get("https://openrouter.ai/api/v1/models", timeout=30)
        r.raise_for_status()
        prices = {}
        for m in r.json().get("data", []):
            p = m.get("pricing", {})
            try:
                # OpenRouter quotes USD per *single* token as strings.
                pin = float(p["prompt"]) * 1_000_000
                pout = float(p["completion"]) * 1_000_000
            except (KeyError, TypeError, ValueError):
                continue
            prices[m["id"]] = (pin, pout)
        if prices:
            # Overlay the snapshot for anything the live feed omits.
            merged = dict(PRICE_TABLE)
            merged.update(prices)
            return merged, "live"
    except Exception:
        pass
    return dict(PRICE_TABLE), "offline"


def price_for(prices, model_id):
    """Look up (in, out) $/Mtok, tolerating a missing :free/:nitro suffix."""
    if model_id in prices:
        return prices[model_id]
    base = model_id.split(":")[0]
    if base in prices:
        return prices[base]
    return None


# ── Cost model ─────────────────────────────────────────────────
def estimate_cost(target_price, judge_price, n_probes, thinking):
    """Return a cost breakdown dict for n_probes, in USD."""
    t_in, t_out = target_price
    j_in, j_out = judge_price
    out_tok = TOK["target_out_think"] if thinking else TOK["target_out"]

    per_mtok = 1_000_000.0
    target_cost = n_probes * (TOK["target_in"] * t_in + out_tok * t_out) / per_mtok
    judge_cost = n_probes * (TOK["judge_in"] * j_in + TOK["judge_out"] * j_out) / per_mtok
    return {
        "n_probes": n_probes,
        "target_cost": target_cost,
        "judge_cost": judge_cost,
        "total": target_cost + judge_cost,
    }


def fmt_usd(x):
    if x < 0.01:
        return f"${x*100:.2f}¢"  # sub-cent shown in cents
    if x < 1:
        return f"${x:.3f}"
    return f"${x:.2f}"


# ── Display ────────────────────────────────────────────────────
def print_source_note(source):
    if source == "live":
        print(f"  {DIM}Prices: live from OpenRouter /models.{RESET}")
    else:
        print(f"  {DIM}Prices: built-in snapshot (early-2026 OpenRouter list "
              f"prices) — estimate only. Run online for live rates.{RESET}")


def show_single(model, target_price, judge_price, n_probes, thinking, source):
    b = estimate_cost(target_price, judge_price, n_probes, thinking)
    t_in, t_out = target_price
    mode = "thinking" if thinking else "standard"

    W = 56  # inner width between the ║ borders

    def box_line(text):
        return f"  ║ {text[:W]:<{W}} ║"

    print()
    print(f"  ╔{'═' * (W + 2)}╗")
    print(box_line("IKP Budget Estimate"))
    print(f"  ╠{'═' * (W + 2)}╣")
    print(box_line(f"Target:    {model}"))
    print(box_line(f"Probes:    {n_probes}  ({mode} mode)"))
    print(box_line(f"Est. cost: {fmt_usd(b['total'])} per run"))
    print(f"  ╚{'═' * (W + 2)}╝")
    print()
    print(f"  Breakdown (per run of {n_probes} probes):")
    print(f"    Target model  ({t_in:>6.2f}/{t_out:>6.2f} $/Mtok in/out) : {fmt_usd(b['target_cost']):>10s}")
    print(f"    Judge  ({JUDGE_MODEL})            : {fmt_usd(b['judge_cost']):>10s}")
    print(f"    {'─' * 52}")
    print(f"    {BOLD}Total{RESET}                                         : {BOLD}{fmt_usd(b['total']):>10s}{RESET}")
    print()

    # Cost at other common sample sizes, so the reader can trade accuracy for money.
    print(f"  At other sample sizes:")
    for n in [140, 200, 400, 700, FULL_PROBE_COUNT]:
        if n > FULL_PROBE_COUNT:
            continue
        bb = estimate_cost(target_price, judge_price, n, thinking)
        tag = "  (full set)" if n == FULL_PROBE_COUNT else ""
        print(f"    {n:>5} probes : {fmt_usd(bb['total']):>10s}{tag}")
    print()
    print_source_note(source)
    if thinking:
        print(f"  {DIM}Note: thinking-mode output tokens vary widely by model and "
              f"question; treat the target cost as a rough upper-ish estimate.{RESET}")
    print()


def show_budget(model, target_price, judge_price, budget, thinking, source):
    """Given a dollar budget, report how many probes / full runs it buys."""
    per_full = estimate_cost(target_price, judge_price, FULL_PROBE_COUNT, thinking)["total"]
    per_probe = per_full / FULL_PROBE_COUNT
    affordable_probes = int(budget / per_probe) if per_probe > 0 else 0
    full_runs = budget / per_full if per_full > 0 else 0

    print()
    print(f"  Budget: {fmt_usd(budget)}   Target: {model}   "
          f"({'thinking' if thinking else 'standard'} mode)")
    print(f"  {'─' * 60}")
    print(f"  Full 1,400-probe run costs   : {fmt_usd(per_full)}")
    print(f"  Your budget buys             : {full_runs:.1f} full runs")
    print(f"                                 (or one run of up to "
          f"{min(affordable_probes, FULL_PROBE_COUNT)} probes)")
    if affordable_probes < FULL_PROBE_COUNT:
        # Round down to a per-tier-clean sample the estimator accepts.
        clean = (affordable_probes // 7) * 7
        print(f"  Suggested --sample           : {clean} "
              f"(stratified, {clean // 7}/tier)")
    else:
        extra = int(full_runs)
        print(f"  You can score the full set on ~{extra} models within budget.")
    print()
    print_source_note(source)
    print()


def show_list(prices, n_probes, thinking, source):
    print()
    mode = "thinking" if thinking else "standard"
    print(f"  IKP run cost across common models  "
          f"({n_probes} probes, {mode} mode, judge = {JUDGE_MODEL})")
    print(f"  {'─' * 66}")
    print(f"  {'Model':40s} {'$/Mtok in/out':>16s} {'Est. cost':>9s}")
    print(f"  {'─' * 66}")
    rows = []
    for m in LIST_MODELS:
        tp = price_for(prices, m)
        if tp is None:
            continue
        jp = price_for(prices, JUDGE_MODEL) or PRICE_TABLE[JUDGE_MODEL]
        b = estimate_cost(tp, jp, n_probes, thinking)
        rows.append((m, tp, b["total"]))
    for m, tp, total in sorted(rows, key=lambda r: r[2]):
        price_str = f"{tp[0]:.2f}/{tp[1]:.2f}"
        print(f"  {m:40s} {price_str:>16s} {fmt_usd(total):>9s}")
    print(f"  {'─' * 66}")
    print()
    print_source_note(source)
    print()


# ── Main ───────────────────────────────────────────────────────
def resolve_probe_count(sample):
    """Return the exact requested sample count, or the full set by default."""
    if sample is None:
        return FULL_PROBE_COUNT
    if sample < 0:
        raise ValueError("sample size must be non-negative")
    return sample


def main():
    parser = argparse.ArgumentParser(
        description="IKP Budget — estimate the $ cost of a run before you spend a token",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", "-m", metavar="MODEL",
                        help="Target model ID (OpenRouter), e.g. openai/gpt-4.1")
    parser.add_argument("--sample", "-n", type=int, metavar="N",
                        help="Estimate for an N-probe stratified sample (default: full 1400)")
    parser.add_argument("--thinking", action="store_true",
                        help="Assume the target runs in reasoning mode (more output tokens)")
    parser.add_argument("--budget", "-b", type=float, metavar="USD",
                        help="Given a dollar budget, report what it buys")
    parser.add_argument("--list", action="store_true",
                        help="Show a cost table across common models and exit")
    parser.add_argument("--judge", metavar="MODEL", default=JUDGE_MODEL,
                        help=f"Judge model to price (default: {JUDGE_MODEL})")
    parser.add_argument("--offline", action="store_true",
                        help="Skip the live price fetch; use the built-in snapshot")
    parser.add_argument("--json", action="store_true",
                        help="Emit the estimate as JSON instead of a table")
    args = parser.parse_args()

    prices, source = fetch_live_prices(args.offline)
    try:
        n_probes = resolve_probe_count(args.sample)
    except ValueError as exc:
        parser.error(str(exc))
    judge_price = price_for(prices, args.judge) or PRICE_TABLE.get(JUDGE_MODEL)

    if args.list:
        show_list(prices, n_probes, args.thinking, source)
        return

    if not args.model:
        parser.print_help()
        print("\n  Tip: `--list` compares common models; `--model X --budget 10` "
              "sizes a run to $10.\n")
        return

    target_price = price_for(prices, args.model)
    if target_price is None:
        print(f"\n  Unknown price for '{args.model}'.")
        print(f"  It isn't in the live feed or the built-in table. Options:")
        print(f"    • check the exact OpenRouter model ID (see openrouter.ai/models)")
        print(f"    • run without --offline so live prices are fetched")
        print(f"    • or price it yourself: full run ≈ 1400 × "
              f"({TOK['target_in']} in + {TOK['target_out']} out) target tokens "
              f"+ judge.\n")
        sys.exit(1)

    if args.json:
        b = estimate_cost(target_price, judge_price, n_probes, args.thinking)
        out = {
            "model": args.model,
            "judge": args.judge,
            "n_probes": n_probes,
            "thinking": args.thinking,
            "price_source": source,
            "target_price_per_mtok": {"in": target_price[0], "out": target_price[1]},
            "judge_price_per_mtok": {"in": judge_price[0], "out": judge_price[1]},
            "cost_usd": {
                "target": round(b["target_cost"], 6),
                "judge": round(b["judge_cost"], 6),
                "total": round(b["total"], 6),
            },
        }
        print(json.dumps(out, indent=2))
        return

    if args.budget is not None:
        show_budget(args.model, target_price, judge_price, args.budget,
                    args.thinking, source)
        return

    show_single(args.model, target_price, judge_price, n_probes,
                args.thinking, source)


if __name__ == "__main__":
    main()
