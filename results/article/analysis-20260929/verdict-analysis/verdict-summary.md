# Verdict analysis

Mean of three pass-level scores. Each bootstrap sample retains all passes and conditions of a claim.
Pointwise intervals are exploratory; repeating claims does not create 300 independent cases.

| Condition | Accuracy, % (95% CI) | Macro-F1 (95% CI) | Δ accuracy vs all, pp (95% CI) |
|---|---:|---:|---:|
| All snippets | 56.0 (47.0–66.0) | 0.298 (0.249–0.346) | +0.0 (+0.0–+0.0) |
| No snippets | 10.0 (5.0–16.0) | 0.064 (0.030–0.100) | -46.0 (-58.0–-34.0) |
| Always Refuted (reference policy) | 62.0 (52.0–71.0) | 0.191 (0.171–0.208) | +6.0 (-7.0–+18.0) |
| Jev | 47.0 (37.3–56.7) | 0.301 (0.237–0.364) | -9.0 (-17.7–-1.0) |
| Gemini | 50.7 (41.0–60.3) | 0.310 (0.258–0.357) | -5.3 (-14.0–+3.3) |
| Qwen-4B | 40.0 (31.0–50.0) | 0.259 (0.205–0.309) | -16.0 (-24.0–-8.0) |
| LFM-2.6B | 45.7 (36.0–55.7) | 0.275 (0.223–0.325) | -10.3 (-18.3–-2.7) |
| Jeff | 43.0 (33.7–52.3) | 0.275 (0.213–0.337) | -13.0 (-23.0–-3.0) |
| Laya typed | 47.7 (38.0–57.3) | 0.259 (0.209–0.307) | -8.3 (-15.3–-2.0) |
| LFM-1.2B | 56.0 (47.0–66.0) | 0.298 (0.249–0.346) | +0.0 (+0.0–+0.0) |

Gold counts: {'Supported': 24, 'Refuted': 62, 'Not Enough Evidence': 4, 'Conflicting Evidence/Cherrypicking': 10}.
The majority-class row is a fixed, sample-prevalence reference policy, not a HerO inference condition.
A confidence interval crossing zero is not evidence of equivalence or non-inferiority.
