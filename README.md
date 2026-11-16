# McCabe-Thiele Distillation Column: ML-Based Stage Prediction

Predicting the number of theoretical stages and the feed stage location of a binary
distillation column using machine learning, trained on synthetic data generated with the
McCabe-Thiele method.

## Project Overview
The McCabe-Thiele method is a classical chemical engineering technique for finding the number of
equilibrium stages in a binary distillation column, given the relative volatility, reflux ratio and
feed/product compositions. This project:

1. Implements the McCabe-Thiele stepping algorithm in Python
2. Generates a synthetic dataset of 5000 distillation column designs
3. Trains linear regression models to predict the **number of theoretical stages** and the **feed stage**

## Dataset Generation
Random seed fixed (`np.random.seed(42)`). Each row is one column design:

| Variable | Description | Sampling range |
|---|---|---|
| `alpha` | Relative volatility | 1.5 - 10 |
| `xF` | Feed composition | 0.2 - 0.8 |
| `xD` | Distillate composition | sampled above the equilibrium curve at `xF`, capped at 0.99 |
| `R_min` | Minimum reflux ratio | derived from `alpha`, `xF`, `xD` (pinch point) |
| `R` | Actual reflux ratio | `R_factor` x `R_min`, with `R_factor` sampled 1.1 - 1.8 |
| `xB` | Bottoms composition | 0.001 - 0.05 (sampled independently) |

Result: 5000 rows, stage count 5 to 40, feed stage 2 to 17, no non-converged cases.

**Design decisions:**
- `xD` is sampled relative to the equilibrium curve, otherwise the specification can become physically invalid.
- `R` is sampled as a multiple of `R_min`, not independently, which would often give unrealistically low stage counts.
- `xB` is sampled independently of `xF`, in a fixed low range, because tying it to `xF` collapses its variance.
- Rows that fail to converge within 100 stages are flagged and excluded (none occurred in the final dataset).

## Exploratory Findings
Correlation with the targets (full 5000 rows):

| Feature | `stage_numbers` | `feed_stage` |
|---|---|---|
| `alpha` | -0.72 | -0.41 |
| `R` | 0.65 | 0.71 |

`xF`, `xD` and `xB` are weaker. Plots of stage count against `alpha` and `R_factor` show a curved,
inverse-shaped relationship, which is a problem for a purely linear model.

## Current results (linear regression baseline)
Features: 5 raw inputs plus `1/alpha`, `1/R` and their product, standardized with `StandardScaler`.
Metrics are on the 20% held-out test set (1000 rows).

| Target | Train R² | Test R² | Test RMSE |
|---|---|---|---|
| Number of stages | 0.84 | 0.84 | 1.50 |
| Feed stage | 0.69 | 0.71 | 0.87 |

## Limitations
- Synthetic data, constant relative volatility, saturated liquid feed (q=1)
- Only one model so far: linear regression, which can only represent linear relationships. Train and
  test R² are nearly equal for both targets, which suggests underfitting (high bias) rather than
  overfitting. Predictions also vary a lot for the same true value: for example, cases with an actual
  stage count of 12 (51 test rows) were predicted anywhere from 9.4 to 16.5.
- Feed stage R² (0.71) is lower than stage count R² (0.84), likely because the feed stage has a
  narrower spread (std 1.60 vs 3.73), so its RMSE is a larger share of that spread (0.54 vs 0.40).

## Next
Polynomial features + Ridge, neural network, Random Forest, XGBoost, feed stage as a classification
problem, and a comparison of all models.

## How to run
```
pip install -r requirements.txt
jupyter notebook 01_linear_regression_baseline.ipynb
```

## Files
- `01_linear_regression_baseline.ipynb`: dataset generation, EDA, modeling, evaluation

## Author
Built by [iqras-dev](https://github.com/iqras-dev). Chemical Engineering background, transitioning into AI/ML engineering.
