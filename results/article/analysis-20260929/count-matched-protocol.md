# Proposed count-matched evidence experiment

Status: **proposed design; not executed and not retrospectively preregistered**. No numbers from these conditions appear in the manuscript's Tables 2–5. The author must freeze the final design, random seeds, inference settings, and failure policy before inspecting new verifier outcomes.

## Research question

Does choosing particular snippets improve a fixed verifier's outcomes relative to choosing other snippets from the same candidate pool, when the number supplied for each claim is held constant?

The current selector-versus-all comparison changes both composition and count. Matching only the total number of retained snippets across the dataset would not resolve this because evidence could be allocated differently across claims.

## Inputs and conditions

For claim i, selector a, and comparison pass p, let C_i be the fixed set of ten original candidates and S_iap its recorded selected subset. Set k_iap = |S_iap|. Use the original candidate texts and existing decisions; do not regenerate selectors to obtain a more convenient evidence count.

For each S_iap, sample random subsets uniformly without replacement from C_i, with exactly k_iap snippets. As a planning default, use ten independent draws, retaining sampled subset identifiers and the random seed. This is a proposed computational allocation, not an estimate of statistical power. Freeze the actual draw count and seed list before new verdicts. Do not stop early, select favorable draws, or drop unfavorable verdicts.

Keep selected and random snippets in the same original candidate order. Keep the claim text, verifier checkpoint, prompt template, truncation policy, generation parameters, and serving configuration fixed. Record the actual post-formatting/post-truncation input and token count. If truncation causes the verifier to see fewer than k complete snippets, either use a predeclared policy that preserves the intended count or report the violation explicitly; do not claim a clean equal-count comparison when visible evidence differs.

At k = 0 or k = 10 there is only one possible subset. These are degenerate controls, not independent composition choices. Reusing an identical saved prediction or rerunning the verifier must be declared in advance and applied consistently. A cache is not an independent inference replicate. Where stochastic generation is used, specify how verifier variation is handled; comparison-pass labels alone do not pair random generation states.

Add two diagnostic conditions: all and only grade-2 snippets, and all grade-1 plus grade-2 snippets. These use privileged annotation information, so they are not deployable selectors or guaranteed performance ceilings. Their direct comparison also changes count and cannot independently establish the causal value of context. Keep the existing all-snippets and no-snippets references.

## Outcomes and estimands

Use mean pass-level accuracy and four-class macro-F1 under the same invalid-output convention as the manuscript. Declare which outcome is primary, the comparison family, and the multiplicity strategy before new results. Preserve class-level precision/recall/F1, invalid-output rates, empty-evidence frequency, and the number of available and retained direct-evidence snippets.

For selector a and metric M, the comparison is:

`Delta_a = mean_p M(gold, verifier(S_ap)) - mean_(p,r) M(gold, verifier(R_apr))`.

Here M is calculated on the whole claim sample for each pass/draw before averaging. In particular, do not compute a separate macro-F1 for each claim and average it. Random subsets match the selector's per-claim count, not its relevance grade mix; matching grade mix would answer a different question.

Use paired claim-level bootstrap samples. Each sampled claim retains every selector condition, original pass and random-subset draw. Recalculate the pass/draw metrics within each bootstrap sample and then the contrast. Retain all draws instead of choosing a representative one. These intervals are conditional on the finite random-control draws; report their between-draw variation and distinguish Monte Carlo variability from claim-sampling uncertainty. Repeated claims/draws are not independent new claims.

Report corrected and harmed cases relative to the count-matched controls. A difference whose interval spans zero does not establish equivalence, non-inferiority, or absence of a selection benefit. A non-inferiority claim needs a prospectively justified margin and design, rather than a margin chosen after observing scores.

## Operational controls

Record requests, responses, subset identifiers, original ordering, serving identity, failures, latency, and input/output tokens. Missing or invalid results remain in planned denominators under the declared failure policy; log exclusions and reruns separately. No gold labels or annotations should enter ordinary selector requests. Annotation-defined diagnostic sets must be clearly distinguished in the output schema.

Snippet count is not token cost. Compare actual verifier tokens and runtime; an end-to-end efficiency claim must additionally include selector and orchestration overhead. New verifier input or settings may make the old saved predictions unsuitable as a comparator, requiring a matched rerun under the frozen new protocol.

## Interpretation and next stage

A positive equal-count contrast would support the usefulness of the chosen composition for this fixed verifier and sample. It would not alone establish source credibility, faithful evidence use, improvement on new domains, or superiority of a split-model architecture. The direct-only/direct-plus-context conditions diagnose how the verifier reacts to annotation categories; no-snippets outcomes against original benchmark labels must not be interpreted as a requirement to recover factual labels without evidence.

A separate live-trajectory study should compare shared-model and split-role assignments, allowing actions to alter later states. It should assess process-rule violations, evidence retention, per-class outcomes and total resource use, with analyst corrections feeding a held-out requalification process. That study is distinct from the equal-count composition test.
