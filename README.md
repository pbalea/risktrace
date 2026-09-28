# RiskTrace — Fraud Detection & Explainability Pipeline

An end-to-end fraud detection system built on the IEEE-CIS Fraud Detection dataset (590K+ real-world transactions), combining gradient-boosted tree modeling, graph-based network features, SHAP explainability, and an LLM-generated risk memo layer for compliance review.

## Results

| Metric | Score |
|---|---|
| ROC-AUC | 0.898 |
| PR-AUC | 0.491 (vs. 0.035 random baseline — ~14x lift) |

Evaluated on a **chronologically-split** holdout set (train on earliest 70% of transactions, test on most recent 30%) to simulate realistic deployment conditions rather than a random split, which would leak future information into training.

## Problem

Fraud detection is a classification problem defined by two hard constraints: fraud is rare (~3.5% of transactions in this dataset — severe class imbalance), and the cost of a missed fraud case vs. a wrongly-flagged legitimate transaction are very different. Plain accuracy is a misleading success metric here — a model that always predicts "not fraud" would already be 96.5% accurate while being completely useless.

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) (Kaggle, via Vesta Corporation), two files:
- `train_transaction.csv` — 393 columns: amount, card/product info, and hundreds of anonymized engineered features (`C1-C14`, `D1-D15`, `M1-M9`, `V1-V339`)
- `train_identity.csv` — 41 columns: device type, browser, OS fingerprinting — present for only ~29% of transactions

Many feature meanings are intentionally undisclosed by the data provider. This project treats that as a real constraint, not a gap to paper over — feature usefulness is validated statistically (correlation, SHAP), not by assumed semantic meaning.

> Data files are not committed to this repo (see `.gitignore`) — download from Kaggle and place under `data/` to reproduce.

## Key EDA Findings

- **Identity data presence is predictive, but confounded.** Transactions with device/identity data showed 2–4x higher fraud rates *within* the same product category (`ProductCD`) — but the raw pooled comparison was misleading, since one product type (`W`) never carries identity data and also has the lowest fraud rate.
- **Linear correlation misses real signal.** The `C1-C14` block showed near-zero Pearson correlation with fraud (all under 0.04), yet ranked as the *most important* features in the trained model — the relationships are non-linear, exactly what a tree-based model is built to capture and a correlation check is blind to.
- **`TransactionAmt` works through interaction effects.** Fraud vs. legitimate transactions have nearly identical average amounts (log-scale), yet transaction amount is a top-4 SHAP feature — its predictive power only shows up in combination with other features, not in isolation.
- **Missingness itself is a signal.** The `M1-M9` match flags (billing/shipping name match, etc.) share an identical missingness pattern across all nine columns, and missing values carry a *higher* fraud rate than either a match or a mismatch.

## Graph Feature Engineering

Fraud rings often show up as network patterns invisible to single-row models — e.g., one stolen card tested across many devices, or one compromised device testing many stolen cards. Built with **NetworkX** (a graph library used here purely for feature engineering — not a model itself).

**Approach taken:**
1. Tried card ↔ billing-address linkage first — found no real signal, because the address field turned out to be a coarse regional code, not a specific address (more addresses per card just meant "this person moved," not fraud).
2. Pivoted to card ↔ device linkage — found a real, non-linear signal: cards linked to 10+ distinct devices had roughly double the baseline fraud rate (8.2% vs. 3.5%).
3. Checked the reverse direction (one device → many cards) — found an even stronger signal at moderate degree (up to 15.4% fraud rate), but identified a data-quality ceiling: the highest-degree "devices" were generic phone model strings (e.g., "Samsung Galaxy S7") shared by thousands of unrelated people, not one physical device.
4. Final feature: `high_device_degree` — a binary flag for cards linked to 10+ devices, validated as a meaningful contributor via SHAP.

## Model

**XGBoost** (gradient-boosted decision trees), chosen over alternatives for concrete reasons:
- vs. logistic regression: captures non-linear relationships and feature interactions natively
- vs. deep learning: stronger default for structured/tabular data at this scale
- vs. a Graph Neural Network: more interpretable and faster to justify in a regulated, compliance-driven context — a deliberate trade-off in favor of explainability over marginal performance gains

Class imbalance handled via `scale_pos_weight` (~27.4, the negative:positive class ratio), which weights the rare fraud class more heavily during training.

**Anonymized `V`-block (339 columns) tested and deliberately excluded from the final model** — after deduplicating ~30% of redundant columns via correlation clustering, the retrained model gained only +0.014 PR-AUC (0.491 → 0.504) while adding 6x the feature complexity and a slight regression in ROC-AUC. Chose the leaner, more interpretable model.

## Explainability

**SHAP** (SHapley Additive exPlanations) values attribute each individual prediction across all input features, showing exactly how much each feature pushed a given transaction's score toward fraud or legitimate. Used to validate every EDA hypothesis against real model behavior rather than assuming feature importance.

**LLM risk memo layer:** for any flagged transaction, the top 5 SHAP-contributing features are passed to Claude, which generates a short, professional risk memo a compliance reviewer could act on directly. The prompt is deliberately constrained to avoid inventing plausible-sounding meanings for anonymized features (e.g., asserting "C1 measures transaction velocity") — a real failure mode of LLM-generated explanations, since sounding confidently wrong about *why* a decision was made is arguably worse than acknowledging the limits of what's known.

## Limitations & Next Steps

- No cost-sensitive decision threshold yet — the model outputs probabilities; a deployed system needs a flag/no-flag rule based on the actual business cost of false positives vs. false negatives
- Device-linkage graph feature has a known precision ceiling at very high degree counts due to generic device-model string collisions
- Trained on a public, static Kaggle dataset rather than live production data — a real deployment would need to account for data drift and adversarial adaptation by fraudsters over time

## Project Structure

```
fraud-detection/
├── data/           # raw/sampled CSVs (gitignored)
├── notebooks/       # exploratory + modeling notebooks
├── src/             # reusable pipeline code
├── models/           # trained model artifacts
├── api/               # serving layer (planned)
├── .env               # API keys (gitignored, never committed)
└── README.md
```

## Stack

Python · pandas · NumPy · XGBoost · NetworkX · SHAP · Anthropic API · Jupyter
