# From A/B Tests to Bandit Evaluation: Causal Inference for Adaptive Systems

When a subscriber opens a streaming app, the system has milliseconds to decide which artwork to show on every content tile. Pick the wrong variant and they scroll past. Enough scroll-pasts and they close the app, maybe cancel next month. Personalization systems use contextual bandits to make these decisions millions of times daily — but how do you know the bandit is actually working if you can't run a clean A/B test on a system that never stops adapting?

This notebook explores that question end to end.

## What's Covered

The notebook walks through the full arc from experimental design to off-policy evaluation:

- **A/B Testing** — power analysis, randomization, hypothesis testing, and the cost of experimentation (regret quantification)
- **Contextual Bandits** — epsilon-greedy implementation with logged propensity scores, cumulative reward comparison against A/B and oracle baselines
- **Why Naive Analysis Fails** — demonstration that simple averages of bandit logs overestimate every arm due to selection bias
- **Off-Policy Evaluation** — three estimators (IPS, SNIPS, Doubly Robust) with vectorized implementations and progressive MSE improvement
- **Failure Modes** — exploration rate sweep showing how epsilon controls evaluation quality, and propensity clipping as a bias-variance tradeoff

Eight visualizations illustrate the key concepts, from the personalization problem itself through estimator comparison and diagnostic plots.

## Running the Notebook

```bash
pip install -r requirements.txt
jupyter notebook ab_test_vs_bandit_ope_with_viz.ipynb
```

## Requirements

Python 3.8+. Dependencies: NumPy, pandas, Matplotlib, SciPy, scikit-learn. See `requirements.txt`.
