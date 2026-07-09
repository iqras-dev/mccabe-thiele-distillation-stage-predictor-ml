# McCabe-Thiele Distillation Column: ML-Based Stage Prediction

Predicting the number of theoretical stages and feed stage location for a binary
distillation column using machine learning, based on synthetic data generated
from the McCabe-Thiele graphical method.

## Project Overview

The McCabe-Thiele method is a classical chemical engineering technique for
determining the number of equilibrium stages needed in a binary distillation
column, given the relative volatility, reflux ratio, and feed/product
compositions. This project:

1. Implements the McCabe-Thiele stepping algorithm in Python
2. Generates a synthetic dataset of 300 valid distillation column designs
3. Trains linear regression models to predict:
   - **Number of theoretical stages** required
   - **Feed stage location** (which stage the feed should enter)

## Motivation

Rather than working the McCabe-Thiele diagram by hand for every design case,
this project explores whether a data-driven model can approximate the
relationship between column design parameters (relative volatility, reflux
ratio, compositions) and the resulting stage count — combining chemical
engineering domain knowledge with applied machine learning.

## Dataset Generation

Each row represents one distillation column design, sampled as follows:

| Variable | Description | Sampling range |
|---|---|---|
| `alpha` | Relative volatility | 1.5 – 10 |
| `xF` | Feed composition | 0.2 – 0.8 |
| `xD` | Distillate composition | sampled above the equilibrium curve at `xF`, capped at 0.99 |
| `R_min` | Minimum reflux ratio | derived from `alpha`, `xF`, `xD` (pinch-point condition) |
| `R` | Actual reflux ratio | `R_factor × R_min`, with `R_factor` sampled 1.1 – 1.8 |
| `xB` | Bottoms composition | 0.001 – 0.05 (sampled independently) |

**Key design decisions, found through iterative testing:**

- **`xD` is sampled relative to the equilibrium curve**, not just above `xF` —
  otherwise the specification can become physically invalid (asking for a
  distillate purity lower than what a single equilibrium stage already
  provides at the feed).
- **`R` is sampled as a multiple of `R_min`**, not independently — an
  independently sampled `R` is very often far above what's needed, producing
  unrealistically low stage counts (2-4 stages) that don't reflect real
  column designs.
- **`xB` is sampled independently of `xF`**, in a fixed, realistic low range —
  tying it to `xF` (e.g. `xF × 0.05`) collapses its variance to near-zero for
  low `xF` values.
- Rows that fail to converge within 100 stages (a near-pinch condition with
  very low `alpha` and `R_factor` close to 1.1) are flagged and excluded.

Random seed fixed (`np.random.seed(42)`) for full reproducibility.

## Exploratory Findings

Correlation analysis showed:

| Feature | `stage_numbers` | `feed_stage` |
|---|---|---|
| `alpha` | strong (-0.71) | moderate (-0.46) |
| `R` | strong (0.70) | strong (0.74) |
| `xF` | weak | weak |
| `xD` | weak | moderate |
| `xB` | weak | weak |

Plotting `stage_numbers` against `alpha` and `R_factor` revealed a clearly
**nonlinear, inverse-shaped relationship** (steep near the pinch point,
flattening at higher values) — consistent with the physical behavior of a
McCabe-Thiele diagram near minimum reflux.

## Modeling Approach

Two separate `LinearRegression` models were trained (features standardized
with `StandardScaler`, 80/20 train-test split):

**Feature engineering:** based on the observed inverse-shaped relationships,
`1/alpha`, `1/R`, and their interaction term were added as additional
features alongside the five raw inputs.

### Results

| Target | RMSE | R² |
|---|---|---|
| `stage_numbers` | 1.25 stages | 0.80 |
| `feed_stage` | 0.73 stages | 0.57 |

**Stage number model:** feature engineering improved R² from 0.74 → 0.80.
`1/alpha` was the single most influential feature. Residual analysis shows
the model is reliable for easier separations (low stage counts) but less
consistent for harder, near-pinch cases — a real limitation of fitting a
linear model to an inherently curved relationship, not a data or code issue.

**Feed stage model:** weaker fit, explained by `feed_stage` being a discrete,
"staircase-like" target (only takes values 2-11) rather than a smoothly
continuous quantity — the same absolute error matters proportionally more
against such a narrow range.

## Planned Next Step

Treat `feed_stage` prediction as a **multiclass classification** problem
(e.g. softmax regression) rather than regression, since it's fundamentally a
discrete category rather than a continuous value. This is planned once
covered in coursework (currently progressing through Andrew Ng's Machine
Learning Specialization).

## Tech Stack

- Python, NumPy, Pandas
- scikit-learn (`LinearRegression`, `StandardScaler`, `train_test_split`)
- Matplotlib

## Files
- `LinearRegressionforMcCabeTheileMethod (2).ipynb` — full notebook: dataset generation, EDA, modeling, evaluation

## Author

Built by [iqras-dev](https://github.com/iqras-dev) — Chemical Engineering
background, transitioning into AI/ML engineering.
