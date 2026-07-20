# Bike Sharing Demand Prediction — Multiple Linear Regression

A multiple linear regression model that predicts daily demand for shared bikes using **730 days of rental data (2018–2019)** with weather, seasonal, and calendar features. Built for BoomBikes, a bike-sharing provider looking to understand the factors driving demand and prepare a post-pandemic recovery strategy.

## Business Objective

BoomBikes wants to know **which variables significantly predict bike demand** and **how well those variables explain it**, so management can tune business strategy — pricing, fleet planning, and marketing — around the true drivers of ridership.

## Dataset

| Property | Detail |
|---|---|
| Records | 730 daily observations × 16 columns |
| Target | `cnt` — total daily rentals (mean ≈ 4,508; range 22 – 8,714) |
| Features | Season, year, month, holiday, weekday, working day, weather situation, temperature, feeling temperature, humidity, windspeed |
| Data quality | No missing values, no duplicates |

## Methodology

1. **Data Preparation** — mapped encoded categoricals (season, month, weekday, weathersit) to readable labels; dropped non-predictive columns (`instant`, `dteday`, `casual`, `registered`)
2. **EDA** — boxplots and bar plots of categorical drivers; pair-plots of numeric features (temp/atemp show the strongest correlation with demand, r ≈ 0.63)
3. **Dummy Encoding** — `drop_first=True` to avoid multicollinearity → 32 model-ready columns
4. **Train–Test Split** — 70/30 (511 train / 219 test), `MinMaxScaler` applied to numeric features
5. **Feature Selection** — RFE to shortlist 15 features, then manual backward elimination using p-values and VIF via `statsmodels` OLS
6. **Assumption Validation** — residual normality, homoscedasticity, linearity, independence (Durbin–Watson = 2.16), and multicollinearity checks

## Final Model Performance

| Metric | Value |
|---|---|
| R² (train) | 0.816 |
| Adjusted R² (train) | 0.812 |
| R² (test) | 0.813 |
| F-statistic | 246.5 (p ≈ 9.7e-178) |
| Durbin–Watson | 2.16 |

The near-identical train and test R² (gap of only 0.3 percentage points) indicates a well-generalized model with no overfitting.

## Key Demand Drivers (final 9-feature model)

| Feature | Coefficient | Interpretation |
|---|---|---|
| `temp` | +3,984 | Strongest positive driver — warm days boost rentals |
| `yr` | +1,995 | Demand grew strongly year over year (avg 3,406 → 5,610) |
| `weathersit_bad` | −2,256 | Rain/snow days cut demand sharply |
| `season_spring` | −1,168 | Spring underperforms other seasons |
| `weathersit_moderate` | −662 | Misty/cloudy weather dampens demand |
| `mnth_jul` | −515 | July dip despite summer |
| `season_winter` | +497 | Winter outperforms baseline |
| `mnth_sept` | +466 | September peak month |

**Top 3 drivers of demand: temperature, year, and season.**

## Recommendations

- Concentrate fleet capacity and marketing spend in warm months, especially September
- Plan for weather-driven dips with dynamic pricing or promotions on bad-weather days
- Ride the year-over-year growth trend with capacity expansion

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` (RFE, MinMaxScaler, train_test_split, r2_score) · `statsmodels` (OLS, VIF) · `seaborn` / `matplotlib`

## Repository Structure

```
├── Bike_Sharing_Linear_Regression.ipynb   # Full analysis notebook
├── day.csv                                # Dataset (730 daily records)
├── Solution_PPT.pdf                       # Subjective Q&A presentation
└── README.md
```

## How to Run

```bash
pip install pandas numpy scikit-learn statsmodels seaborn matplotlib
jupyter notebook Bike_Sharing_Linear_Regression.ipynb
```

---

*Author: Kailas Wadje — MSc Data Science & AI, University of Liverpool*
