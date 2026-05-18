# Heart Disease Risk Classification

Comparison of KNN, Logistic Regression, and Random Forest for predicting
heart disease, with a focus on the precision-recall tradeoff and decision-
threshold selection for medical screening.

## Problem

In medical screening, a missed diagnosis (false negative) can be fatal,
while a false alarm only triggers additional non-invasive tests. This
project optimizes for **recall** over accuracy, and shows that for this
dataset, **threshold tuning matters more than model choice**.

## Dataset

UCI Heart Disease Dataset (Cleveland), 303 patients, 13 features. After
dropping 6 rows with missing `ca` or `thal` values: 297 patients. Target
binarized from severity score: 0 = no disease, 1-4 → 1 = disease.

## Methodology

1. **Train/test split** (80/20, stratified, random_state=42)
2. **StandardScaler fit on training set only** to avoid leakage
3. **Three models compared** on the same split: KNN, Logistic Regression,
   Random Forest
4. **KNN tuned** via 5-fold cross-validation on the training set
5. **Decision threshold tuned** for the best model using a cost-aware
   argument

## Results

| Model | Recall (Disease) | Precision | Accuracy | False Negatives |
|---|---|---|---|---|
| KNN (K=5, baseline) | 0.82 | 0.92 | 0.88 | 5 |
| KNN (K=13, tuned) | 0.82 | 0.88 | 0.87 | 5 |
| Logistic Regression | 0.79 | 0.85 | 0.83 | 6 |
| Random Forest (threshold 0.5) | 0.82 | 0.88 | 0.87 | 5 |
| **Random Forest + threshold 0.30** | **0.93** | **0.79** | **0.85** | **2** |

## Key insights

- **Three different algorithms converged on ~0.82 recall** at the default
  threshold, suggesting an information ceiling from these 13 features.
  Model choice mattered less than expected.
- **CV-based tuning didn't beat the KNN baseline.** Cross-validation
  argmax recall picked K=1, which overfit on the test set. A stable
  plateau around K=13-19 was chosen instead and matched the baseline.
- **Threshold tuning was the biggest lever.** Lowering the Random Forest
  threshold from 0.50 → 0.30 cut missed diagnoses by 60% (5 → 2), at the
  cost of 4 additional false alarms. A single line of code outperformed
  every architectural change.
- **LogReg and RF agreed on top predictors.** Both identified `ca`,
  `thal`, `cp`, `oldpeak` as primary risk indicators. RF ranked `thalach`
  much higher than LogReg, consistent with RF's ability to capture
  non-linear effects.

## Threshold selection

Chose threshold = 0.30 by minimizing `5·FN + FP`, treating a missed
diagnosis as 5× the cost of a false alarm — defensible for screening
contexts where misses can be fatal and false alarms only trigger
follow-up tests. The choice is robust across cost ratios from 2× to 5×.

## Tech stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn

## Limitations & next steps

- Test set is only 60 patients; differences within ~2 FN are within
  sampling noise. Repeated stratified CV or a larger held-out set would
  tighten the comparison.
- Cleveland subset only. The UCI archive includes Hungarian, Swiss, and
  VA cohorts — multi-site validation would test generalization.
- Could explore gradient boosting (XGBoost) and probability calibration
  (Platt scaling, isotonic regression).
- Production deployment requires a clinically validated cost ratio, not
  the assumed 5×.

## How to run

```bash
git clone https://github.com/<JubayerRafin>/heart-disease-ml.git
cd heart-disease-ml
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook heart_disease.ipynb
```
