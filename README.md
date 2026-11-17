# Improving Motor Insurance Pricing Through Joint Modelling of Claim Frequency and Claim Severity

## Project overview

This project investigates whether machine-learning methods can improve motor third-party liability (MTPL) pure-premium predictions compared with traditional actuarial generalized linear models (GLMs). Pricing must be set before an insurer knows whether a policyholder will claim or the amount of any claim. Models that miss nonlinear effects and interactions can therefore lead to poor risk segmentation and adverse selection.

The study uses the freMTPL2 French motor insurance dataset to model claim frequency and severity, estimate expected claim cost per unit of exposure (pure premium), and examine model interpretability, calibration, and fairness.

## Research questions

1. Do gradient-boosting models (XGBoost or LightGBM) and neural networks improve on a Poisson GLM for claim frequency, as measured by Poisson deviance?
2. Does a Tweedie model or a separate frequency-and-severity approach produce better pure-premium estimates?
3. Which risk factors explain model predictions, and do model outcomes differ across age groups and regions?
4. How well calibrated are the models across risk deciles, as assessed with lift charts and the Gini index?

## Roadmap

### 1. Audit and prepare the data

- Inspect the frequency and severity files, schemas, record counts, missing values, ranges, and duplicate records.
- Document and resolve known inconsistencies between the two files; define and validate the policy-to-claim matching procedure.
- Preserve the distinction between policy-level frequency and claim-level severity.
- Account for differing policy exposure periods in frequency modeling and evaluation.
- Define a reproducible train/validation/test strategy and avoid leakage between related policy and claim records.
- Record all cleaning decisions and assumptions for the research methodology.

### 2. Explore the portfolio

- Summarize claim counts, zero-claim prevalence, exposure, and claim-cost distributions.
- Examine the distributions and relationships of driver, vehicle, geographic, and bonus-malus variables.
- Visualize potential nonlinearities and interactions, including driver age, vehicle power, and bonus-malus level.
- Build on the exploratory analysis already present in `Data_Analysis/generate_eda_report.py` and `Reports/EDA_Report.html`.

### 3. Establish actuarial baselines

- Fit a Poisson GLM for claim frequency with exposure handled appropriately.
- Fit a severity model on observed claims, selecting a suitable distribution and documenting the choice.
- Combine frequency and severity predictions to create a baseline pure-premium estimate.
- Fit a Tweedie model as an alternative pure-premium baseline.

### 4. Train machine-learning models

- Train and tune gradient-boosting models for frequency, severity, and/or pure premium, as supported by the data and evaluation design.
- Evaluate a neural-network approach if it is feasible within the project scope.
- Use consistent data splits and comparable feature preparation across candidate models.
- Keep hyperparameter selection separate from final test-set evaluation.

### 5. Compare predictive performance and calibration

- Compare frequency models using Poisson deviance.
- Evaluate severity and pure-premium predictions with metrics appropriate to their target distributions.
- Compare Tweedie and combined frequency-times-severity pure premiums.
- Assess calibration by risk decile and report lift charts and the Gini index.
- Present uncertainty and limitations alongside headline results.

### 6. Explain and assess model behavior

- Use SHAP analysis to identify influential predictors and investigate interactions.
- Compare explanations across the GLM and machine-learning models where practical.
- Assess calibration and prediction differences across age groups and regions.
- Discuss fairness findings in context, including the limits of the available data and chosen group metrics.

### 7. Document and reproduce the study

- Record dataset versions, preprocessing, feature definitions, model settings, and evaluation procedures.
- Cite the freMTPL2 data source from the CASdatasets R package (Dutang and Charpentier); verify the full bibliographic reference for the final paper.
- Summarize findings against each research question and clearly state limitations and future work.
- Ensure that the analysis can be rerun from the documented project files and dependencies.

## How to use the data for model training

The project currently contains raw data and exploratory notebooks, not a trained model or a training script. The following is a proposed, reproducible starting point; complete the audit before implementing it.

### 1. Understand the two tables

| Table | One row represents | What it provides |
|---|---|---|
| `Data/freMTPL2freq.csv` | One policy | `ClaimNb`, exposure in years, and policyholder, vehicle and geographic features |
| `Data/freMTPL2sev.csv` | One recorded claim amount | `IDpol` and `ClaimAmount` in euros; a policy may have multiple rows |

Join the tables on `IDpol`, but **do not join raw claims directly to policies when training frequency**: multiple claim rows would duplicate a policy and its exposure. Use the original one-row-per-policy table for frequency. For cost targets, sum claim amounts and count severity records by `IDpol` first, then left-join those summaries to the policy table.

For example, a policy with 0.5 years of exposure, two claims and amounts of EUR 1,000 and EUR 500 has a count of 2, total cost EUR 1,500 and observed cost rate EUR 3,000 per policy-year. A policy with no claims contributes a zero count and zero cost; it has no severity observation. These are training outcomes, not predictors available at pricing time.

### 2. Audit and reconcile before defining targets

Check unique policy IDs, positive exposure, nonnegative integer claim counts, positive claim amounts, missing values, unmatched IDs, and whether each policy's `ClaimNb` equals its number of recorded severity rows. Investigate duplicates rather than assuming equal amounts are duplicate claims. Record all decisions and affected policy and claim counts.

An audit of the CSV files currently in this repository found 678,013 distinct policy IDs, 36,102 claims in `ClaimNb`, and 26,639 severity records. Six severity IDs (195 records) have no matching policy. Among matched policies, 9,117 have a claim-count/record-count mismatch: 9,116 have claims but no severity record, and one has a different positive count. These figures describe these files and should be recomputed if the data changes.

**Missing severity is not a zero-cost claim.** As an initial, explicitly limited analysis:

1. Retain all valid policies for a standalone frequency benchmark; report any exclusions.
2. For the main frequency-times-severity and Tweedie comparison, define a common cost-evaluable cohort of valid policies where `ClaimNb` equals the number of matched severity records. A zero-claim policy with no severity records gets total cost zero; a claimed policy with missing or mismatched amounts is excluded from this cohort, not assigned zero cost. Unmatched severity records cannot be used without policy features.
3. Report the exclusions, exposure, claims and observed costs, and compare the included and excluded policies by risk characteristics. Missing amounts may be systematic, so results on this cohort need not generalize to the full portfolio. Investigate the source of the mismatch and perform sensitivity analyses before making portfolio-wide pricing claims.

### 3. Split policies, then prepare features

- Split **policy IDs**, not claim rows, into training, validation and final test sets (for example 70% / 15% / 15%, with a recorded random seed). Stratify by whether `ClaimNb` is positive where practical. Assign every claim belonging to a policy to that policy's split; use the same split assignments for every model and both cohorts.
- Fit imputers, categorical encoders, scaling and any data-driven capping or transformations on the training set only, then apply them to validation and test. Handle previously unseen categories explicitly. Document any capping and retain an uncapped sensitivity analysis for large losses.
- Use only information available when pricing: `VehPower`, `VehAge`, `DrivAge`, `BonusMalus`, `VehBrand`, `VehGas`, `Area`, `Density` and `Region`, subject to availability and suitability. Do not use `IDpol`, `ClaimNb` or `ClaimAmount` as predictors. Use `Exposure` to account for observation time as described below, not as a substitute for the claim target.
- The files do not provide an explicit time split here, so a seeded random split estimates performance on similar policies, not future-period performance. Reserve the test set for the final comparison.

### 4. Train baselines and combine predictions

Start with interpretable GLMs, then compare gradient boosting or other models using the same targets, cohort and splits.

| Model | Training rows and target | Exposure treatment and prediction |
|---|---|---|
| Frequency | One row per policy, including zero claims; target `ClaimNb` | Fit a Poisson model with `log(Exposure)` as an offset, producing predicted claims per policy-year. Equivalently, for a Poisson estimator supporting sample weights, fit `ClaimNb / Exposure` with weight `Exposure`. |
| Severity | One row per matched recorded claim in the training split; target `ClaimAmount` | Fit a positive-cost model such as a Gamma GLM; predict the mean euros per claim. Do not add zero-claim policies as zero-severity claims. Alternatively, model policy mean claim cost with claim-count weights. |
| Pure premium | Combine the two predictions for each policy | `predicted claims per policy-year × predicted euros per claim` gives expected euros per policy-year. Multiply by policy exposure for expected cost over that policy's observed period. |
| Tweedie comparison | One row per cost-evaluable policy, including zero-cost policies; target total cost divided by exposure | Fit a compound Poisson-Gamma Tweedie model with exposure weights and a power between 1 and 2; predict euros per policy-year. Select the power and other hyperparameters on validation data. |

For the main comparison, fit **both** frequency and severity components using the training portion of the cost-evaluable cohort, and fit Tweedie on the same cohort. A separate frequency model trained on all valid policies can assess whether excluding incomplete cost records changes frequency performance. The severity model estimates a claim-weighted average amount conditional on the features; the product estimates expected cost under the frequency/severity model assumptions, not an observed individual premium.

For illustration, a predicted frequency of 0.08 claims per policy-year and predicted severity of EUR 2,000 per claim give a pure premium of **EUR 160 per policy-year**, or EUR 80 for six months of exposure. This is expected claim cost only: expenses, profit, taxes and other pricing adjustments are not included.

### 5. Validate and report

- Tune preprocessing choices and model hyperparameters on training/validation data only; compare every pure-premium model on the **same cost-evaluable test policies**.
- Evaluate frequency using exposure-weighted mean Poisson deviance of `ClaimNb / Exposure` against predicted annual claim rate. Also compare total observed claims with the sum of predicted annual rates multiplied by exposure.
- Evaluate severity on held-out recorded claims using, for example, mean Gamma deviance for positive predictions and amounts; also compare observed and predicted total claim costs. Examine large losses separately.
- For pure premium, compare observed total cost with predicted rate times exposure, and use an exposure-weighted Tweedie deviance on cost per exposure with the **same fixed evaluation power** for every candidate. Show calibration and lift by risk decile, the Gini index, and results by age group and region; include group sizes, exposure and uncertainty.
- Report results on the cost-evaluable cohort alongside the full-policy frequency benchmark and the reconciliation limitations. Once all choices are fixed, optionally refit on training plus validation data and evaluate once on the untouched test set.

## Current project assets

- `Data/freMTPL2freq.csv` — policy-level frequency and exposure data.
- `Data/freMTPL2sev.csv` — claim-level severity data.
- `Data_Audit_and_Reconcilation/Exploratory_Data_Analysis.ipynb` — ongoing data-quality exploration.
- `Data_Analysis/generate_eda_report.py` — script for generating the exploratory data analysis report.
- `Reports/EDA_Report.html` — generated EDA report.

The files above are present in the repository; the roadmap stages are research tasks, not claims that their analyses are complete.

## Dataset

The freMTPL2 dataset contains approximately 678,000 motor insurance policies, with driver, vehicle, region, bonus-malus, and exposure information. The severity file contains monetary losses for claims. Most policies have no claims, exposure periods vary, and the frequency and severity files have known inconsistencies; these issues must be addressed and documented before modeling.

Available dataset sources:

- [Hugging Face](https://huggingface.co/datasets/mabilton/fremtpl2) — direct CSV files (`freMTPL2freq.csv` and `freMTPL2sev.csv`).
- [OpenML frequency dataset](https://www.openml.org/d/41214).
- [OpenML severity dataset](https://www.openml.org/d/41215).
- [Kaggle dataset](https://www.kaggle.com/datasets/karansarpal/freMTPL2-french-motor-tpl-insurance-claims) — free account required.

The frequency data can also be loaded with `sklearn.datasets.fetch_openml(data_id=41214)`.

### Data Dictionary

#### freMTPL2freq (Policy-level features and claim frequency)

| Column | Type | Range / Values | Description |
|--------|------|-----------------|-------------|
| **IDpol** | Integer | Unique ID | Policy identifier (links to severity dataset) |
| **ClaimNb** | Integer | 0–16 | Number of claims during exposure period |
| **Exposure** | Float | 0–2.01 years | Policy duration in years |
| **VehPower** | Integer | 4–15 | Vehicle power rating |
| **VehAge** | Integer | 0–100 years | Vehicle age in years |
| **DrivAge** | Integer | 18–100 years | Driver age in years |
| **BonusMalus** | Integer | 50–350 | Bonus-malus coefficient (<100 = bonus, ≥100 = malus) |
| **VehBrand** | Categorical | 11 codes (B1, B2, ..., B12) | Car brand (anonymized/grouped) |
| **VehGas** | Categorical | "Diesel", "Regular" | Fuel type |
| **Area** | Categorical | A, B, C, D, E, F | Density zone (A = rural, F = urban center) |
| **Density** | Integer | 1–27,000+ | Population density (inhabitants/km²) |
| **Region** | Categorical | 21 codes | French administrative region (pre-2016) |

#### freMTPL2sev (Claim-level amounts)

| Column | Type | Range / Values | Description |
|--------|------|-----------------|-------------|
| **IDpol** | Integer | Repeated for multiple claims | Policy identifier (links to frequency dataset) |
| **ClaimAmount** | Float | >0 EUR | Individual claim amount in euros |
