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

## Current project assets

- `Data/freMTPL2freq.csv` — policy-level frequency and exposure data.
- `Data/freMTPL2sev.csv` — claim-level severity data.
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
| **IDpol** | Integer | Unique ID | Policy identifier (links to frequency dataset) |
| **ClaimAmount** | Float | >0 EUR | Individual claim amount in euros |
