# Autonomous Vehicle Journey Risk Analysis

> Would you trust the ride? An analysis of 2,500 autonomous vehicle journeys to find what drives problems, test whether they can be predicted, and decide whether the technology can be trusted.

**Live app:** _add your Streamlit Cloud link here_

---

## Contents

- [Overview](#overview)
- [Data](#data)
- [Key findings](#key-findings)
- [Prediction model](#prediction-model)
- [Would we trust the ride?](#would-we-trust-the-ride)
- [The Streamlit app](#the-streamlit-app)
- [Running the project](#running-the-project)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Ethical considerations](#ethical-considerations)

---

## Overview

Autonomous vehicles rely on sensors, data and AI instead of a human driver. This project answers five questions:

1. Can we identify patterns that humans might miss?
2. Which factors have the biggest impact on performance?
3. Can we predict when problems might occur?
4. How could AI improve future journeys?
5. Based on the analysis, would we trust the ride?

The target variable is `problem_occurred`. Across all journeys, about **20%** had a problem.

## Data

Four tables are joined with left merges on their ID columns:

| Table | Rows | Join key | Contents |
|---|---|---|---|
| `journeys` | 2,500 | - | Date, time, conditions, sensor readings, outcomes |
| `vehicle` | 10 | `vehicle_profile_id` | Model, software version, sensor suite, age |
| `roads` | 30 | `road_segment_id` | Road type, junctions, markings, lighting, infrastructure |
| `weather` | 8 | `weather_code` | Weather name, risk multiplier, visibility factor |

```python
data = (journeys
        .merge(vehicle, on='vehicle_profile_id', how='left', validate='m:1')
        .merge(roads, on='road_segment_id', how='left', validate='m:1', suffixes=('', '_road'))
        .merge(weather, on='weather_code', how='left', validate='m:1'))
# Result: 2,500 rows x 44 columns
```

**Data quality notes:**

- `road_surface` is largely determined by weather: snow always means a snow-covered road, and rain always means a wet road. Only "Uneven" carries independent information, so the model uses an `uneven_surface` flag instead.
- Urban streets are the only roads with high junction complexity, so their effects cannot be separated.
- Traffic level and visibility are independent of each other.

## Key findings

### 1. Difficult conditions compound

Counting independent difficult conditions (bad weather, high traffic, low visibility):

| Difficult conditions | Problem rate | Journeys |
|---|---|---|
| 0 | 13.5% | 1,043 |
| 1 | 21.7% | 931 |
| 2 | 28.2% | 408 |
| 3 | **36.4%** | 118 |

Each extra condition adds roughly 7–8 percentage points. Journeys with all three are **2.7 times** as likely to have a problem.

### 2. Snow is the single biggest risk

Snow journeys have a **47.7%** problem rate (n=86), compared with **15.7%** in clear weather (n=1,213).

### 3. Sensors show warning signs

- Journeys below 75% sensor confidence have a **41.6%** problem rate; journeys at or above 90% have **15.9%**.
- Journeys with 5+ sensor alerts have a **95.6%** problem rate, and journeys with none have **0%**. This near-perfect split suggests alerts are part of how problems are recorded, so the alert count is excluded from the prediction model.

### 4. Other factors

| Factor | Worst | Best |
|---|---|---|
| Unexpected obstacle | Yes: 29.9% | No: 18.8% |
| Traffic level | High: 24.9% | Low: 14.3% |
| Visibility | Lowest 25%: 24.7% | Highest 25%: 14.3% |
| Sensor suite | Standard: 23.7% | Advanced: 17.0% |
| Road markings | Poor: 23.8% | Good: 17.4% |
| Road type | Urban street: 22.4% | A road: 17.1% |
| Software version | 4.1: 22.5% | 4.3: 18.1% |

### 5. A misleading pattern: vehicle age

Newer vehicles initially appeared riskier (21.9% vs 16.8%). The cause was that 59% of the newest vehicles carry the Standard sensor suite, while the oldest are all Enhanced. Within the same sensor suite, age has little effect. **Sensor suite, not age, drives the difference.**

## Prediction model

### Approach

- **Features:** only information known *before* the journey starts: weather, traffic, time of day, visibility, distance, road characteristics and vehicle profile.
- **Excluded to prevent leakage:** `safety_incident`, `emergency_stop`, `human_intervention`, `journey_completed`, `delay_min`, `actual_time_min`, `risk_level` and `sensor_alert_count`. These are outcomes or in-journey records, not causes.
- **Split:** 75% train / 25% test, stratified on the target.
- **Class imbalance:** `class_weight='balanced'`, so the model prioritises catching problems.

### Results

| Model | CV ROC AUC | Test accuracy | Precision | Recall |
|---|---|---|---|---|
| **Logistic regression** | **0.624 ± 0.021** | 0.630 | 0.275 | 0.520 |
| Decision tree (depth 4) | 0.567 ± 0.046 | 0.474 | 0.248 | 0.800 |
| Baseline (always "no problem") | 0.500 | 0.800 | - | 0.000 |

Adding in-journey sensor confidence and obstacle detection did not meaningfully help (test AUC 0.599 → 0.611). Sensor confidence largely reflects conditions the model already has, such as weather and visibility.

### Choosing the model

**Logistic regression was selected.** It had the higher and more stable cross-validated AUC. The decision tree scored better on the single test split, but cross-validation showed that was due to the particular split. Its higher recall came from a more cautious default threshold, not from better knowledge.

The deployed model uses a **0.4 threshold** instead of 0.5, to favour catching problems:

| Threshold | Recall | Precision | Journeys flagged |
|---|---|---|---|
| 0.50 | 52% | 28% | 38% |
| **0.40** | **75%** | **22%** | **68%** |
| 0.35 | 88% | 22% | 81% |
| 0.30 | 96% | 21% | 92% |

**Conclusion:** the patterns describe risk across *groups* of journeys, but the available data cannot reliably predict *individual* problems. Catching nearly all problems means flagging nearly all journeys, and precision stays close to the 20% base rate.

## Would we trust the ride?

**Not yet, at least not for an important journey without human oversight.**

Even in the best conditions, 13.5% of journeys had a problem, and the model cannot reliably warn passengers in advance which journeys will go wrong.

**Recommendations:**

1. Restrict operation in snow and when two or more difficult conditions combine.
2. Phase out the Standard sensor suite.
3. Require remote human oversight when sensor confidence drops below 75%.
4. Collect richer data: more vehicles, a clear definition of "problem", and second-by-second sensor data.

## The Streamlit app

The app scores a planned journey before departure.

1. **Route and vehicle:** choose a road segment and a vehicle profile. Their characteristics fill in automatically.
2. **Conditions:** set weather, traffic, time of day, visibility and distance.
3. **Risk assessment:** shows three things:
   - the risk score compared with the 0.4 alert threshold
   - the journey's percentile among past journeys
   - how often past journeys with similar scores actually had a problem
4. **Why this score?:** a chart of the factors raising or lowering this journey's risk, using the logistic regression coefficients.

The score is not shown as a percentage chance. Balanced class weights inflate the model's probabilities, so the app reports the observed problem rate of similar past journeys instead.

## Running the project

### Files

| File | Purpose |
|---|---|
| `app_6.py` | Streamlit app |
| `av_risk_model.joblib` | Saved model bundle (pipeline, threshold, lookup tables, metrics) |
| `requirements.txt` | Python dependencies, pinned to the versions used for training |
| Analysis notebook | Data merging, exploration, modelling and evaluation |

### Run locally

```bash
pip install -r requirements.txt
```

```bash
streamlit run app_6.py
```

The app opens at `http://localhost:8501`.

### Recreate the model file

Run the save cell at the end of the notebook. It refits the chosen pipeline on all 2,500 journeys and writes `av_risk_model.joblib`, then prints the scikit-learn version, which must match `requirements.txt`.

## Deployment

The app runs on [Streamlit Community Cloud](https://share.streamlit.io):

1. Push `app_6.py`, `av_risk_model.joblib` and `requirements.txt` to GitHub.
2. On share.streamlit.io, click **Create app** and choose the repository and `app_6.py`.
3. Under **Advanced settings**, set the Python version to match the one used in the notebook.

## Troubleshooting

**`AttributeError` inside `joblib.load` / `pickle.find_class`**

The model file was saved with different package versions from the ones the server installed. Fix it as follows:

1. In the notebook, print the exact versions:
   ```python
   import sys, sklearn, pandas, numpy, joblib
   print(sys.version.split()[0], sklearn.__version__, pandas.__version__, numpy.__version__, joblib.__version__)
   ```
2. Pin those versions in `requirements.txt` (for example `scikit-learn==X.Y.Z`).
3. If the Python version differs, recreate the app on Streamlit Cloud with the matching Python version, which can only be set when creating an app.
4. Reboot the app.

## Limitations

- **Only 10 vehicle profiles:** vehicle-level findings rest on very few distinct vehicles.
- **Overlapping fields:** road surface follows weather, and urban streets are the only high-complexity junctions.
- **Undefined target:** what counts as a "problem" isn't documented, and sensor alerts appear to be part of its definition.
- **Weak prediction:** ROC AUC is about 0.62, so the model is a screening aid, not a decision-maker.
- **Association, not causation:** the data is observational, and confounding (as with vehicle age) can mislead.

## Ethical considerations

- **Bias:** urban streets and roads with poor markings show higher problem rates. Restricting service where risk is higher could leave areas with poorer infrastructure underserved.
- **Explainability:** the vehicle-age example shows how easily patterns mislead. Safety decisions need models whose reasoning can be inspected.
- **Privacy:** journey dates, times and locations reveal people's movements, and vehicle cameras record bystanders. Data should be minimised, anonymised and retained only for a limited time.
- **Safety:** the model misses about a quarter of problems even at a cautious threshold. Society must decide what level of risk is acceptable before widespread use.
- **Human oversight:** the data records human interventions, so humans are still part of the system. Clear rules are needed on when a human takes over and who is accountable.
