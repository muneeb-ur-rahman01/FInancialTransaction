# Financial Transaction Analysis & Prediction System

An end-to-end Python data science project on the **Online Retail** dataset (541,909 transactions of a UK
wholesale gift retailer, Dec 2010 – Dec 2011). It cleans and analyses the data, tests what is statistically
real, and trains, evaluates and serves four models. Results are available on the command line and in an
interactive Streamlit dashboard.

| Question | Approach | Result on unseen data |
|---|---|---|
| How much will we sell in the next trading days? | Random Forest, recursive multi-day forecast | MAE £12,230/day, 3.7% better than the best simple rule |
| How many units will each product sell next week? | 4-week moving average (beat every ML model fairly) | WAPE 61.8%, product-level bias 19.5% |
| Which customer groups exist? | K-Means on log-scaled RFM, k = 4 | Silhouette 0.335 (rule-based RFM: 0.141), stability ARI 0.92 |
| Which order lines are keying errors? | Line-level rule, top 1% alert queue | Catches 25% of known errors at 24× the random rate |

---

## 1. Setup

Requires **Python 3.14** (tested with 3.14.7) on Windows, macOS or Linux.

```powershell
# from the project folder
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place the raw data file at **`data/raw/Online_Retail.xlsx`** (UCI Machine Learning Repository, "Online Retail").
The original file is never modified.

## 2. Quick start

```powershell
python main.py pipeline        # build everything (about 10 minutes)
python main.py dashboard       # open the dashboard at http://localhost:8501
python main.py test            # run the test suite
```

### All commands

| Command | What it does |
|---|---|
| `python main.py pipeline` | Data quality → cleaning → features → EDA → statistics → 4 models → evaluation report |
| `python main.py pipeline --from models` | Restart from a later step (`quality`, `clean`, `features`, `eda`, `stats`, `models`, `report`) |
| `python main.py forecast-sales --days 12` | Next 12 trading days of net sales with an 80% range |
| `python main.py forecast-demand --top 20` | Next week's units for the 20 biggest products |
| `python main.py forecast-demand --code 85123A 22423` | Next week's units for specific products |
| `python main.py segment --customer 12347` | A customer's segment and the recommended action |
| `python main.py segment --recency 10 --frequency 8 --monetary 2500` | Segment for a what-if profile |
| `python main.py check-line --code 85123A --quantity 5000 --price 2.55` | Is this order line suspicious, and why? |
| `python main.py evaluate` | Regenerate the consolidated evaluation report |

Add `--save` to a forecast command to write a CSV to `reports/predictions/`.

Each step can also be run on its own, for example `python -m src.models.sales_forecasting`
or `python notebooks/04_eda.py`.

## 3. Dashboard

`python main.py dashboard` (or `streamlit run dashboard/app.py`) opens eight pages:

| Page | Contents |
|---|---|
| Overview | KPIs, monthly sales, top countries and products |
| Transaction Analysis | Transaction types, data quality, order sizes, weekday × hour heatmap |
| Financial Trends | Daily trend, sales per trading day, return rate, seasonality, customer concentration |
| Sales Forecasting | Next 6–24 trading days with an 80% range; test-period accuracy |
| Product Demand | Next-week forecast for 2,237 products with search; per-product history |
| Customer Segmentation | Segments, profiles and actions, customer lookup, what-if, downloadable win-back list |
| Anomaly Detection | Line checker, review queue (CSV), confusion matrix, precision–recall |
| Model Performance | Evaluation of every model and the cross-task overview |

Date and country filters in the sidebar apply to the analysis pages. The dashboard uses a dark theme
(`.streamlit/config.toml`); set `base = "light"` there to switch.

## 4. Project structure

```
data/raw/                 original Excel file (read-only)
data/processed/           clean transactions, feature tables, segments, anomaly scores (parquet)
notebooks/                analysis scripts: 01 understanding, 02 data quality, 04 EDA, 05 statistics
src/data_processing/      cleaning.py (Module 3), features.py (Module 6)
src/analysis/             eda.py (Module 4), statistical_tests.py (Module 5)
src/models/               sales_forecasting, demand_forecasting, customer_segmentation, anomaly_detection (7A-D),
                          evaluation.py + evaluation_report.py (Module 8), predict.py (Module 9), estimators.py
src/utils/                paths.py, plotting.py
artifacts/models/         saved models (joblib)
artifacts/preprocessors/  saved preprocessing (RFM transform)
dashboard/                app.py, pages/, components/
reports/                  text findings, figures/, model_reports/, predictions/
tests/                    pytest suite (unit tests, leakage tests, dashboard smoke tests)
config/                   reserved for configuration
main.py                   single entry point
```

## 5. Methodology

### Data quality and cleaning (Modules 2–3)
**No row is deleted.** Every row gets a `RowID` linking it back to the raw file, plus labels, and each analysis
picks the rows it needs.

- `TransactionType`: sale, cancellation (`C` invoices), stock adjustment (price 0: damages, write-offs),
  accounting adjustment (`A` invoices: bad debt).
- `IsProduct` separates merchandise from postage, fees and manual adjustments.
- `IsDuplicate` flags 5,268 repeated lines (kept, not dropped).
- `IsEntryCorrection`: a sale cancelled with identical quantity and price **within one hour**. This is an
  order-entry mistake that never really happened (452 pairs, £292K). Slower exact reversals are genuine returns.
- Missing `CustomerID` (25% of rows) means anonymous sales. They are kept for sales and demand, and excluded
  only for customer analysis.
- **Revenue = net product sales** (sales minus returns): £9,792,310.

### Exploratory and statistical analysis (Modules 4–5)
- **Seasonality:** sales per trading day are 2.6× higher in November than in February.
- **Weekly pattern:** there is no Saturday trading at all.
- **Concentration:** the UK is 85% of sales; the top 20% of customers bring in 74% of revenue.
- **Non-parametric tests**, because order values have skewness 17: Mann-Whitney, Kruskal-Wallis with Holm
  correction, Spearman, chi-square with Cramér's V. **Every test reports an effect size.**
- **Time-series diagnostics for forecasting:** ADF and KPSS agree daily sales are non-stationary; there is a
  weekly cycle at a lag of 6 trading days.

### Feature engineering (Module 6)
There are four feature tables, one per model: daily, product × week, customer (RFM), and invoice/line. Every
lag uses `shift(k ≥ 1)`, rolling statistics are computed on the shifted series, and RFM uses only
transactions before a snapshot date. Tests prove that changing a future value leaves past features unchanged.

### Machine learning and evaluation (Modules 7–8)
Every task follows the same protocol:
1. **Split by time.** Train on the past; test on the most recent weeks, scored once.
2. **Baselines first.** Simple rules are scored alongside the models, and complexity must earn its place.
3. **Select on cross-validation or a validation period only**, never on the test period.
4. **Fit preprocessing and norms inside the training data** (no leakage).
5. **Use metrics that fit the problem.**
   - Forecasting: MAE, RMSE, R², WAPE and bias.
   - Demand: WAPE + product-level bias, because WAPE alone rewards under-forecasting skewed counts.
   - Clusters: silhouette + bootstrap stability.
   - Anomalies: precision, recall, F1, confusion matrix and average precision. Accuracy is meaningless when
     99.9% of lines are normal.

The full comparison is in `reports/model_reports/model_evaluation_summary.md` (and the dashboard's Model
Performance page).

**Segments (K-Means, k = 4)**

| Segment | Customers | Revenue share | Recommended action |
|---|---|---|---|
| Champions | 690 | 64% | Reward and retain |
| Loyal regulars | 1,189 | 24% | Cross-sell, loyalty pricing |
| New customers | 821 | 5% | Onboarding, second-order incentive |
| Hibernating | 1,633 | 7% | Low-cost reactivation only |

269 Champions and Loyal regulars with no order for 90+ days are flagged as **lapsing** (a win-back list
worth £489K in lifetime spend).

### Prediction system (Module 9)
`src/models/predict.py` serves every saved model and is used by both `main.py` and the dashboard, so the
command line and the dashboard always agree.

- **Sales:** recursive multi-day forecasts. A test checks that the feature row built at prediction time is
  identical to the training features.
- **Demand:** next-week forecasts from features built on an extended week grid.
- **Segments:** customer lookup and what-if profiles.
- **Anomalies:** line scoring with plain-language reasons.

## 6. Testing

```powershell
python -m pytest -q
```

The suite covers:
- cleaning rules;
- every aggregation and statistical test, on data with known answers;
- **leakage tests** for the feature tables;
- each model's selection logic, with end-to-end runs on synthetic data;
- the prediction system against the saved models;
- smoke tests that run every dashboard page headlessly.

Tests that need pipeline outputs are skipped automatically until `python main.py pipeline` has run.

## 7. Limitations

- **One year of data.** No model has seen a previous autumn, so every model under-forecasts the Q4 ramp-up.
  A second year of data would help most.
- **Holidays are unknown to the forecaster.** Last year the business closed 24 Dec – 3 Jan.
- **Demand depends on sales history alone**, which cannot anticipate promotions or launches. Products with
  under 4 weeks of history are not forecast.
- **The anomaly answer key is partial.** It mixes keying errors with quick customer cancellations of normal
  lines, so measured precision is a lower bound.
- **Customer segments** are a snapshot at 10 Dec 2011 and should be refreshed regularly.
