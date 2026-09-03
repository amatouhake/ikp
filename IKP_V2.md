# IKP v2 — gaming-resistant parameter estimation

*A hardening of the IKP estimator that closes the two attack surfaces
quantified in [`ADVERSARIAL_IKP.md`](ADVERSARIAL_IKP.md), without changing
the calibration. Everything below is reproducible from committed data;
see **Validation** for how to check it yourself.*

## Why v2

v1 reports a single parameter-count number. The adversarial analysis
showed that number is cheap to move: a model can look **2× smaller by
refusing ~7% of what it knows** (sandbagging), or look larger by
memorizing the public probe answers (contamination). v2 doesn't pretend
those attacks don't exist — it makes them visible and, where possible,
inert.

## What changed

### 1. Refusal-robust interval (defends against sandbagging)

Instead of one number, v2 reports a **point / floor** plus a
**refusal-adjusted** reference and a **confidence tier**:

- **Point (floor).** Size implied by answers the model actually got right;
  refusals scored 0. Sandbagging can only push this *down*, so it is an
  honest lower bound on true capacity.
- **Refusal-adjusted.** Imputes refused probes at the accuracy the model
  shows on the questions it *does* attempt (missing-at-random). It
  over-corrects for models that selectively refuse only hard items, so it
  is a soft upper reference, not a hard ceiling.
- **Confidence tier** from the refusal rate: `Reliable` (<10%),
  `Caution` (10–30%), `Low confidence` (>30%).

Wrong answers are never credited — a confident wrong answer is a
demonstrated failure, not plausible sandbagging. The interval widens with
the refusal rate, so a heavy refuser (e.g. a safety-tuned model) is flagged
as *ambiguous* rather than silently under-estimated. Example on recorded
data:

| Model | refusals | v2 report |
|---|---|---|
| gpt-4.1 | 0.5% | tight, `Reliable` — well-constrained point estimate |
| claude-opus-4.5 | 19% | `Caution` — point is a floor, true size may be higher |
| claude-3-haiku | 56% | `Low confidence` — wide interval; size genuinely ambiguous |

This operationalizes the paper's own "refused-but-known" caveat instead of
leaving it in prose.

### 2. Held-out probe split (defends against contamination)

`scripts/make_probe_split.py` deterministically partitions the 1,400
probes into a **public** half (demos, the interactive CLI, tutorials) and a
**private** half, balanced within every tier (700/700, 100 per tier each).
`ikp_estimate_v2.py --split private` scores only the private half, so an
operator who memorized the published probes gains nothing. The split is a
pure hash of each probe `id` plus a salt; rotating the salt yields a fresh,
non-overlapping split if a private set is suspected of leaking.

## Usage

```bash
# Live, contamination-resistant estimate (private split by default)
export OPENROUTER_API_KEY=sk-or-...
python scripts/ikp_estimate_v2.py --model openai/gpt-4.1

# Re-score an existing v1 run offline — no API key, no new spend
python scripts/ikp_estimate_v2.py --from-results runs/gpt-4.1.json

# Use OpenCode Go as the target; OpenRouter remains the fixed judge
export OPENCODE_GO_API_KEY=...
export OPENROUTER_API_KEY=...
python scripts/ikp_estimate_v2.py \
  --api-base https://opencode.ai/zen/go/v1 \
  --api-style responses \
  --api-key "$OPENCODE_GO_API_KEY" \
  --model muse-spark-1.3-contributor \
  --reasoning-effort xhigh

# (Re)generate the split manifest
python scripts/make_probe_split.py
```

The v2 `--thinking` flag remains a compatibility alias for medium target
reasoning. `xhigh` is requested only when explicitly selected; confirm that
OpenCode Go and Muse apply it from a canary response or provider behavior.

## Validation — how you know these numbers are real, not fabricated

Run `python scripts/18_v2_validation.py`. Every figure below is a
deterministic function of committed, **upstream-authored** data
(`data/results/*.json`), so anyone gets the same output. Three independent
checks:

- **CHECK 1 — faithful reproduction.** We recompute every model's λ=0
  accuracy from raw per-tier verdict counts and compare to the value the
  repository recorded itself. Result: **max difference 0.0 across all 201
  models.** We did not re-grade or invent anything; we reproduce upstream's
  own scoring exactly. (The one model needing the fixed-7-tier convention,
  `ling-2.6-flash`, is the same one the repo lists in `CALIBRATION_EXCLUDE`.)
- **CHECK 2 — the curve reproduces the paper's calibration quality.** We
  rebuild the calibration cohort (open-weight, known size, minus the repo's
  documented exclusions) and refit. Result: **R² = 0.91, median in-sample
  fold error 1.47×, 74% within 2×, 88% within 3×** — the same quality the
  paper reports. The refit slope differs slightly from the stored fit
  because the stored fit uses the curated `final_assembly.json` cohort
  (n=89) while this independent refit uses the broader `evaluation_summary`
  open set (n=93); it is a cohort difference, not staleness — regenerating
  the artifact with `scripts/calibration_refit_v2.py` reproduces the
  committed file to floating-point precision. v2's absolute estimates use
  the *stored* (paper) fit, so they match published numbers.
- **CHECK 3 — intervals behave.** Across the roster, the refusal-robust
  interval width rises from a median **1.35×** for low-refusal models to
  **5.64×** for high-refusal ones — i.e. the interval encodes exactly the
  ambiguity it should.

### What is measured vs. assumed (honest scope)

- **Measured / reproduced from data:** all accuracies, refusal rates, tier
  scores, and the calibration fit. These are re-derivations of upstream's
  recorded evaluations, verified bit-for-bit (CHECK 1).
- **Modeling assumptions (not measurements):** (a) the adversary can
  selectively suppress specific answers — a *threat model*, so the
  sandbagging costs in `ADVERSARIAL_IKP.md` are **lower bounds on attack
  cost**, not observed attacks; (b) the refusal-adjusted upper assumes
  refusals are representative, which over-corrects for selective refusers
  and is labeled a soft reference, not a bound.
- **Inherited limits:** absolute sizes carry IKP's own calibration error
  (~1.5× median fold) and are extrapolation above the largest calibrated
  model (1.8T), where v2 prints `>1.8T` rather than a false-precision number.

## Files

| File | Role |
|---|---|
| `scripts/ikp_estimate_v2.py` | v2 estimator: refusal-robust interval + `--split` |
| `scripts/make_probe_split.py` | Deterministic public/private probe split |
| `scripts/18_v2_validation.py` | The three checks above + `ikp_v2_intervals.png` |
| `data/probes/split_manifest_v2.json` | The committed split (700/700) |
| `data/results/ikp_v2_validation.json` | Machine-readable validation output |
