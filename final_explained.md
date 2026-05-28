# Delhi Climate Forecasting — Code Walkthrough

---

## 1. Dataset Description

**File:** `DailyDelhiClimateTrain.csv`  
**Source:** Kaggle — Delhi Climate Dataset

This dataset contains daily weather observations recorded in **New Delhi, India**. Each row represents one day, with the following columns:

| Column | Unit | Description |
|--------|------|-------------|
| `date` | — | Calendar date (YYYY-MM-DD) |
| `meantemp` | °C | Average daily temperature — **the target variable we forecast** |
| `humidity` | % | Average daily relative humidity |
| `wind_speed` | km/h | Average daily wind speed |
| `meanpressure` | hPa | Average daily atmospheric pressure |

**Date range:** January 1, 2013 → April 24, 2017 (~1,462 daily rows)

Delhi has a strong annual temperature cycle — cold winters (~10°C) and scorching summers (~40°C) — making it an interesting and challenging series for time-series forecasting.

---

## 2. Package Installation

```python
pip_packages = ["torch", "chronos-forecasting", "momentfm", "uni2ts", ...]
subprocess.run(["pip", "install", "-q", "-U", ...] + pip_packages, check=False)
```

**What it does:** Installs all required libraries in a single command before the notebook continues.

**Why a single `pip install` call?**  
Running multiple `pip install` calls separately can cause dependency conflicts — one package might install a version of another that a later install then downgrades. Bundling everything into one call lets pip resolve all dependencies simultaneously and pick compatible versions.

**Key packages installed:**
| Package | Purpose |
|---------|---------|
| `torch` / `torchvision` / `torchaudio` | PyTorch — the deep learning framework all models run on |
| `transformers` | HuggingFace Transformers — loads pre-trained model weights |
| `accelerate` | Speeds up HuggingFace model loading and inference |
| `chronos-forecasting` | Amazon's Chronos time-series foundation models |
| `momentfm` | MOMENT foundation model for time series |
| `uni2ts` / `gluonts` | Moirai model and its GluonTS data utilities |
| `google-generativeai` | Google Gemini API client |
| `lag-llama` | Lag-Llama foundation model |
| `tsfm-public` | IBM TinyTimeMixer model |
| `huggingface_hub` | Downloading models from HuggingFace Hub |

---

## 3. Imports & Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_squared_error, r2_score
import torch
import warnings
warnings.filterwarnings("ignore")
```

**What each import does:**

| Import | Purpose |
|--------|---------|
| `pandas` | Load, index, and manipulate the time-series DataFrame |
| `numpy` | Numerical arrays, math operations, metrics computation |
| `matplotlib.pyplot` | Plotting charts — STL decomposition, forecast vs actual |
| `STL` from `statsmodels` | Seasonal-Trend decomposition (EDA) |
| `mean_squared_error`, `r2_score` from `sklearn` | Compute MSE and R² evaluation metrics |
| `torch` | PyTorch — runs all deep learning models on CPU/GPU |
| `warnings.filterwarnings("ignore")` | Suppresses noisy deprecation warnings that clutter Colab output — doesn't hide real errors |

---

## 4. Data Loading & Preparation

```python
df = pd.read_csv('DailyDelhiClimateTrain.csv')
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)
df = df.asfreq('D')
```

**Step by step:**

- **`pd.read_csv`** — reads the CSV file into a DataFrame with one row per day.
- **`pd.to_datetime`** — converts the `date` column from a plain string (e.g., `"2013-01-01"`) to a proper datetime object that pandas can sort, slice, and index.
- **`set_index("date")`** — makes the date the row label instead of a plain column. This is the standard format for time-series work — you can then slice with `df["2015"]` or `df["2014-06":"2014-08"]`.
- **`asfreq('D')`** — tells pandas the series has a **daily** frequency. If any dates are missing in the CSV (e.g., a day was not recorded), pandas inserts that date with `NaN` values. This is critical because forecasting models expect a contiguous, gap-free time index.

---

## 5. STL Decomposition (Exploratory Data Analysis)

```python
stl = STL(df['meantemp'].dropna(), period=365, robust=True)
stl_fit = stl.fit()
stl_fit.plot()
```

**What is STL?**  
STL stands for **Seasonal-Trend decomposition using LOESS** (Locally Estimated Scatterplot Smoothing). It breaks any time series into three independent components:

```
Original series = Trend + Seasonality + Residual
```

| Component | What it means |
|-----------|---------------|
| **Trend** | The long-term direction of temperature — is Delhi getting hotter over the years? |
| **Seasonality** | The repeating annual pattern — summers hot, winters cold |
| **Residual** | Whatever is left after removing trend and seasonality — random noise, anomalies |

**Why `period=365`?**  
The data is daily, and Delhi's climate repeats on a yearly cycle (~365 days). Setting `period=365` tells STL to look for annual seasonality.

**Why `robust=True`?**  
Standard LOESS fitting is sensitive to outliers (e.g., an unusually extreme day can distort the estimated trend). `robust=True` uses iteratively re-weighted fitting so that outliers have less influence on the decomposition — it uses a median-like approach internally.

**Why run this BEFORE the train/test split?**  
STL decomposition is purely **exploratory** — it helps you understand the data's structure. It is not used in modeling at all. Running it on the full dataset gives you the most accurate picture of the overall trend and seasonal shape, which you'd lose if you only looked at the training portion.

**What the printed output tells you:**
- `Trend range`: the minimum and maximum of the slow-moving underlying temperature — e.g., `10.5 – 32.1 °C` would mean the trend oscillates by ~22°C over the years.
- `Seasonal range`: how much the seasonal pattern swings — e.g., `-12.0 – 11.5 °C` means seasonality alone accounts for a ~23°C swing.
- `Residual std`: how noisy the leftover component is — a small residual std means the trend + seasonality explain most of the variation.

---

## 6. ADF Stationarity Test

```python
from statsmodels.tsa.stattools import adfuller

def run_adf(series, label):
    result = adfuller(series.dropna(), autolag='AIC')
    stat, p, lags, _, crit, _ = result
    is_stationary = p < 0.05
    ...
```

**What is stationarity?**  
A time series is **stationary** if its statistical properties (mean, variance) do not change over time — the series fluctuates around a constant average with no upward or downward drift.

**What does the ADF test do?**  
The **Augmented Dickey-Fuller (ADF) test** is a statistical hypothesis test:
- **Null hypothesis (H₀):** The series has a unit root → it is **non-stationary** (has a trend or drift).
- **Alternative hypothesis (H₁):** The series is **stationary**.

A low **p-value (< 0.05)** means we reject H₀ → the series is stationary.

**Why `autolag='AIC'`?**  
The ADF test must include extra lag terms to account for autocorrelation in the series. `autolag='AIC'` automatically selects the optimal number of lags by minimizing the Akaike Information Criterion — it picks just enough lags without overfitting.

**What happens if non-stationary?**  
The code applies **1st-order differencing** — subtracting each value from the previous one:
```
diff[t] = temperature[t] - temperature[t-1]
```
Differencing removes trends and often makes a series stationary. The ADF test is then re-run on the differenced series to confirm.

**Why does it matter?**  
Classical models like ARIMA require stationarity. Foundation models like Chronos handle non-stationarity internally (they normalize inputs), but running the ADF test is good practice — it tells you whether the raw series is safe to use or needs transformation.

---

## 7. Train / Test Split

```python
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx].copy()
test  = df.iloc[split_idx:].copy()
```

**80/20 time-based split** — the first 80% of days go into training (~1,170 days), the last 20% into testing (~292 days).

**Why no shuffling?**  
This is a **time series** — order matters. Shuffling would let the model see future data during training (e.g., training on day 1000, then evaluating on day 500), which artificially inflates performance. The split is always chronological.

**Why `.copy()`?**  
Without `.copy()`, `train` and `test` are views into the same DataFrame. Modifying one would silently modify the other, causing subtle bugs during the outlier-handling step.

**Why split BEFORE any preprocessing?**  
This is the most important rule in time-series ML: **no data leakage**. Any statistics derived from the data (IQR bounds, rolling means, normalization parameters) must be computed from the training set only. If you preprocessed first and split later, test statistics would contaminate training, giving an over-optimistic evaluation.

---

## 8. Outlier Handling

### 8a. Mean Pressure — Domain Threshold

```python
mask = (split_df["meanpressure"] < 900) | (split_df["meanpressure"] > 1100)
split_df.loc[mask, "meanpressure"] = np.nan
split_df["meanpressure"] = split_df["meanpressure"].interpolate().ffill().bfill()
```

**Why 900–1100 hPa?**  
Sea-level atmospheric pressure on Earth ranges roughly 870–1085 hPa in extreme weather. Values outside 900–1100 hPa in Delhi are physically impossible — they indicate sensor errors or data entry mistakes. This threshold is safe to apply to both train and test because it's based on domain knowledge, not computed from the data.

### 8b. Temperature, Humidity, Wind Speed — IQR Method

```python
Q1, Q3 = train[col].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
```

**What is the IQR method?**  
The **Interquartile Range** method defines outlier boundaries:
- **Lower bound** = 25th percentile − 1.5 × IQR
- **Upper bound** = 75th percentile + 1.5 × IQR

Any value outside this range is replaced with `NaN` then linearly interpolated.

**Why compute IQR from train only?**  
If you computed IQR from the full dataset (train + test combined), the test set's distribution would influence the bounds — that's data leakage. The bounds are computed strictly from `train` and then applied to both splits.

**`interpolate().ffill().bfill()`** — fills NaN values using linear interpolation between neighboring valid values. `.ffill()` and `.bfill()` handle any NaNs at the very start or end of the series that `interpolate()` can't fill.

---

## 9. Feature Engineering

```python
def add_features(df_split, train_tail=None):
    combined['month']       = combined.index.month
    combined['day_of_year'] = combined.index.day_of_year
    combined['sin_doy']     = np.sin(2 * np.pi * combined['day_of_year'] / 365)
    combined['cos_doy']     = np.cos(2 * np.pi * combined['day_of_year'] / 365)
    for lag in [1, 7, 14]:
        combined[f'temp_lag_{lag}'] = combined['meantemp'].shift(lag)
    for window in [7, 30]:
        combined[f'temp_rollmean_{window}'] = combined['meantemp'].shift(1).rolling(window).mean()
        combined[f'temp_rollstd_{window}']  = combined['meantemp'].shift(1).rolling(window).std()
```

**Why do feature engineering after the split?**  
Features like lag values and rolling means use past temperature values. If you created features before splitting, a test row could inadvertently have its features computed using values from future test rows — leakage.

**Feature explanations:**

| Feature | Why it's useful |
|---------|----------------|
| `month` | Captures monthly seasonal variation (1–12) |
| `day_of_year` | More granular than month — captures where we are in the year (1–365) |
| `sin_doy`, `cos_doy` | **Cyclical encoding** — day 1 and day 365 are neighboring days, but numerically 1 and 365 are far apart. Encoding as sin/cos maps the year onto a circle, so day 365 → day 1 has no artificial discontinuity |
| `temp_lag_1` | Yesterday's temperature — strongest single predictor |
| `temp_lag_7` | Temperature one week ago — captures weekly patterns |
| `temp_lag_14` | Temperature two weeks ago — captures slower trends |
| `temp_rollmean_7` | 7-day rolling mean — smoothed recent trend |
| `temp_rollmean_30` | 30-day rolling mean — captures monthly climate |
| `temp_rollstd_7/30` | Rolling standard deviation — how volatile temperature has been recently |

**Why `.shift(1)` before rolling?**  
Shifting by 1 day before computing the rolling window ensures that today's value is never included in its own rolling feature — otherwise you'd be using future information to predict the present.

**`train_tail` / `SEED_ROWS = 30`:**  
Lag and rolling features for the first few test rows depend on the last few training rows. The function accepts `train_tail` (the last 30 training rows) and prepends them to the test set so rolling features can be computed correctly, then strips them off before returning.

---

## 10. Evaluation Metrics

Four metrics are commonly used to evaluate time-series forecasts. This project uses **RMSE** and **sMAPE**.

---

### RMSE — Root Mean Squared Error

**Formula:**
```
RMSE = √( mean( (actual - predicted)² ) )
```

**What it measures:** The square root of the average squared error. Because errors are squared before averaging, large mistakes are penalized much more heavily than small ones.

**Unit:** Same as the target — here, **°C**. An RMSE of 2.5 means forecasts are off by roughly 2.5°C on average (with large errors weighted more).

**Pros:**
- Same unit as the target — easy to interpret
- Differentiable — useful for training neural networks

**Cons:**
- Sensitive to outliers — one very bad prediction can dominate the score
- Not comparable across different datasets or targets with different scales

---

### MAE — Mean Absolute Error *(not used in this code, but commonly compared)*

**Formula:**
```
MAE = mean( |actual - predicted| )
```

**What it measures:** The average absolute difference between actual and predicted values. Every error counts equally — a 5°C miss is exactly 5× worse than a 1°C miss.

**Unit:** Same as the target — **°C**.

**Pros:**
- Robust to outliers — large errors do not dominate
- Very easy to interpret: "on average, predictions are off by X degrees"

**Cons:**
- Not differentiable at zero — harder to use as a training loss
- Treats all errors equally, even when large errors should be penalized more

**RMSE vs MAE:**  
If RMSE >> MAE, the model has a few very large errors. If RMSE ≈ MAE, errors are consistently sized. RMSE always ≥ MAE.

---

### MAPE — Mean Absolute Percentage Error *(not used in this code)*

**Formula:**
```
MAPE = mean( |actual - predicted| / |actual| ) × 100
```

**What it measures:** The average percentage error. Scale-free — a MAPE of 10% means predictions are off by 10% of the actual value on average.

**Pros:**
- Scale-free — comparable across different datasets
- Intuitive: expressed as a percentage

**Cons:**
- **Breaks when actual = 0** (division by zero) — dangerous for temperature data where winter values may be near 0°C
- **Asymmetric:** a 10°C over-prediction and a 10°C under-prediction on a 20°C day give different MAPE values. Intuitively they should be equal — but they're not.
- Penalizes under-forecasting more than over-forecasting

---

### sMAPE — Symmetric MAPE *(used in this code)*

**Formula:**
```
sMAPE = mean( |actual - predicted| / ((|actual| + |predicted|) / 2) ) × 100
```

**What it measures:** Like MAPE, but the denominator is the average of the actual and predicted values — making it symmetric and more stable near zero.

**Why it's used here:**  
Delhi's winter temperatures can be close to 0°C. Regular MAPE would produce extreme or undefined values in those cases. sMAPE's denominator adds the predicted value as a buffer, preventing division by near-zero.

**Pros:**
- **Symmetric:** over-prediction and under-prediction are penalized equally
- **Handles near-zero** values much better than MAPE
- Scale-free percentage — easy to compare across series

**Cons:**
- Still not perfectly intuitive (the denominator shifts depending on both values)
- Can still behave oddly when both actual and predicted are near zero

---

### Summary Comparison

| Metric | Unit | Outlier Sensitive | Handles Near-Zero | Symmetric | Used in Code |
|--------|------|:-----------------:|:-----------------:|:---------:|:------------:|
| RMSE   | °C   | Yes (strongly)    | Yes               | Yes       | Yes |
| MAE    | °C   | No                | Yes               | Yes       | No  |
| MAPE   | %    | No                | **No**            | **No**    | No  |
| sMAPE  | %    | No                | Yes               | Yes       | Yes |

Also computed: **MSE** (RMSE² — raw squared error) and **R²** (proportion of variance explained; 1.0 = perfect, 0.0 = no better than predicting the mean, negative = worse than the mean).

---

## 11. Rolling Forecast Function

```python
def rolling_forecast(model_fn, ctx_full, test_vals_arr, step=14, ctx_cap=512):
    preds = []
    ctx = list(ctx_full)
    for i in range(0, len(test_vals_arr), step):
        horizon = min(step, len(test_vals_arr) - i)
        chunk = model_fn(np.array(ctx[-ctx_cap:], dtype=np.float32), horizon)
        preds.extend(chunk[:horizon])
        ctx.extend(test_vals_arr[i : i + horizon])
    return np.array(preds, dtype=np.float32)
```

**What is walk-forward (rolling) forecasting?**  
Instead of predicting all test days at once from the training context, the function predicts **14 days at a time**, then **updates the context with the actual values** before predicting the next 14.

```
[Training context] → predict days 1–14
[Training context + actual days 1–14] → predict days 15–28
[Training context + actual days 1–28] → predict days 29–42
...
```

**Why rolling?**  
Foundation models have a **maximum context window** (e.g., 512 or 2048 tokens). If the test period is 292 days, predicting all 292 days at once from only the training context would:
1. Exceed the context limit (for some models)
2. Force the model to forecast very far into the future from a fixed starting point, accumulating error

Rolling forecasting keeps the context fresh with real data, reducing compounding error and staying within context limits.

**`ctx_cap`** — each model has a different maximum context size. `ctx[-ctx_cap:]` takes the most recent `ctx_cap` values from the ever-growing context list.

**`step=14`** — predict 14 days at a time. This balances accuracy (more frequent updates) against computation cost (fewer model calls).

---

## 12. Foundation Models

Foundation models for time series are large neural networks **pre-trained on millions of diverse time series from many domains** (finance, energy, weather, retail, etc.). They work **zero-shot** — you do not fine-tune them on your specific dataset. You simply provide a historical context and ask them to predict the future.

All three models in this code are from **Amazon's Chronos family**.

---

### Model 1: Chronos-T5-Tiny

```python
chronos_tiny = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-tiny",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16
)
```

**Architecture:** T5 (Text-to-Text Transfer Transformer) — an encoder-decoder transformer architecture originally developed by Google for NLP, adapted by Amazon for time-series.

**Size:** ~8 million parameters — the smallest and fastest Chronos variant.

**Context cap:** 512 tokens (days of history the model sees at once).

**`torch_dtype=torch.bfloat16`:** Uses 16-bit brain-float precision instead of 32-bit. Halves memory usage — critical on free Colab GPUs with limited VRAM. Accuracy loss is negligible.

**`num_samples=20`:** Chronos is **probabilistic** — it generates 20 different possible futures (sample paths). The code takes the **median** of these 20 paths as the point forecast. This is more robust than a single deterministic prediction.

**Normalization:**
```python
mu, sigma = ctx_window.mean(), ctx_window.std() + 1e-8
ctx_norm = torch.tensor((ctx_window - mu) / sigma)
fc = chronos_tiny.predict(ctx_norm, ...)
return fc.median(...).numpy() * sigma + mu
```
The input is normalized to zero-mean, unit-variance before being passed to the model, then denormalized afterward. Chronos was trained on normalized data — skipping this step would degrade predictions significantly. The `+ 1e-8` prevents division by zero if the context window is constant.

---

### Model 2: Chronos-T5-Small

```python
chronos_small = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16
)
SMALL_CTX_CAP = 2048
```

**Architecture:** Same T5 encoder-decoder as Tiny, but larger.

**Size:** ~46 million parameters — ~6× more than Tiny.

**Context cap: 2048** — can see ~4 years of daily history at once (vs. 512 days for Tiny). For this dataset where training spans ~4 years, this means Small can potentially see the entire training history in one shot.

**Same probabilistic approach:** `num_samples=20`, median aggregation, same normalization pattern.

**Trade-off vs Tiny:** Slower inference, more GPU memory, but generally more accurate due to larger capacity and longer context.

---

### Model 3: Chronos-Bolt-Small

```python
from chronos import BaseChronosPipeline

chronos_bolt = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-small",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16
)
BOLT_CTX_CAP = 512
```

**Architecture:** A **distilled and optimized** variant of Chronos. "Bolt" refers to its speed — it uses a more efficient inference mechanism than the T5 encoder-decoder pipeline.

**Key difference from T5 variants:** Uses `BaseChronosPipeline` instead of `ChronosPipeline`. The API is slightly different — notably, `chronos_bolt.predict()` does **not** take a `num_samples` parameter, because Bolt produces a **deterministic** (single) forecast rather than a probabilistic distribution.

**Size:** ~46M parameters (same as Small in terms of parameter count, but architecturally optimized for speed).

**Context cap:** 512 — same as Tiny.

**When to prefer Bolt over T5-Small?**  
When inference speed matters more than squeezing out the last fraction of accuracy — Bolt is designed for production latency-sensitive settings.

---

### Model Comparison Table

| Model | Params | Context Cap | API Class | Probabilistic | Relative Speed |
|-------|--------|-------------|-----------|:-------------:|:--------------:|
| Chronos-T5-Tiny | ~8M | 512 days | `ChronosPipeline` | Yes (20 samples → median) | Fastest |
| Chronos-T5-Small | ~46M | 2048 days | `ChronosPipeline` | Yes (20 samples → median) | Slowest |
| Chronos-Bolt-Small | ~46M | 512 days | `BaseChronosPipeline` | No (deterministic) | Fast |

**Memory management:** After each model is done, `del chronos_xxx` is called immediately. This frees GPU VRAM before loading the next model — critical on Colab where GPU memory is limited. Without this, you'd likely hit an out-of-memory error when loading the second model.

---

## 13. Results Comparison & Visualization

```python
comparison_df = pd.DataFrame({
    'Model':     list(results.keys()),
    'MSE':       [...],
    'RMSE':      [...],
    'sMAPE (%)': [...],
    'R2':        [...],
}).sort_values('RMSE').reset_index(drop=True)
```

**What this section does:**
- Builds a single DataFrame with all models and all metrics side-by-side, sorted by RMSE ascending so the best-performing model appears first.
- Produces two sets of plots:
  1. **Combined plot** — all model forecasts overlaid on the actual temperature, so you can see at a glance which ones track the actual curve most closely.
  2. **Individual subplots** — one chart per model, with its RMSE score in the title. Easier for direct per-model evaluation.

**`del model` after each run** — frees GPU VRAM before the next model loads.

---

## 14. Gemini LLM Narrative Analysis

```python
prompt = f"""You are a climate data analyst...
Explain what this forecast tells us about Delhi's temperature during the test period:
1. What is the overall temperature trend — rising, falling, or stable?
2. Are there noticeable fluctuations or spikes? When do they occur?
3. How closely does the forecast follow the actual temperature pattern?
4. What would someone reading this forecast conclude about Delhi's climate during this period?
Speak as if explaining to a city planner or journalist — no technical jargon, no model names, no metric scores.
"""
response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
```

**What it does:** Sends the actual and predicted temperature values, along with the dates, to Google's **Gemini 2.5 Flash** language model. Gemini then generates a natural-language narrative about what the forecast reveals.

**Why the best model is chosen:**  
`comparison_df.iloc[0]` is the model with the lowest RMSE — the most accurate forecast — whose predictions are sent to Gemini for analysis.

**Why this step?**  
Numbers (RMSE, sMAPE) measure accuracy, but they don't explain the *story* of the forecast. Gemini translates raw numbers into human-readable insight:
- "Delhi experienced a sharp cooling in December, with temperatures dropping from 25°C to 10°C over two weeks..."
- "The forecast closely tracked the actual summer peak in May, correctly anticipating the heatwave..."

The prompt explicitly instructs Gemini to avoid mentioning model names or metric scores — the output is intended for a non-technical audience (city planners, journalists, policymakers).
