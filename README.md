# 🌡️ Delhi Climate AI Forecasting Dashboard

> Zero-shot time-series forecasting of Delhi's daily mean temperature using 8 state-of-the-art AI foundation models, with an interactive Streamlit dashboard and a Gemini LLM-powered narrative analysis.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

---

## 📌 Overview

This project benchmarks 8 modern **AI foundation models for time-series forecasting** on Delhi's daily mean temperature (2013–2017). All models run in **zero-shot mode** — no fine-tuning, no training on Delhi data — making it a direct test of out-of-the-box forecasting capability.

The project has two components:

| Component | File | Description |
|-----------|------|-------------|
| Benchmarking script | `final.py` | Runs all 8 models, compares metrics, generates Gemini narrative |
| Interactive dashboard | `app.py` | Streamlit app using the best model (Chronos Bolt Small) |

---

## 🗂️ Dataset

| Property | Value |
|----------|-------|
| File | `DailyDelhiClimateTrain.csv` |
| Period | January 2013 – April 2017 |
| Observations | 1,462 daily records |
| Target | `meantemp` — daily mean temperature (°C) |
| Features | `humidity`, `wind_speed`, `meanpressure` |
| Train / Test split | 80 / 20 (1,169 train days / 293 test days) |

Preprocessing: IQR-based outlier clipping + linear interpolation for missing values.

---

## 🤖 Foundation Models

All 8 models are loaded from HuggingFace and run zero-shot:

| Model | Organization | Notes |
|-------|-------------|-------|
| Chronos-T5-Tiny | Amazon | 8M parameters, 512-day context |
| Chronos-T5-Small | Amazon | 46M parameters, 2048-day context |
| **Chronos-Bolt-Small** | Amazon | Speed-optimized, 512-day context — **used in dashboard** |
| MOMENT | CMU / Databricks | Masked autoencoder |
| Moirai | Salesforce | Universal time-series transformer |
| TimesFM | Google | Large-scale forecasting foundation model |
| Lag-Llama | HuggingFace | LLM-based probabilistic forecaster |
| TinyTimeMixer | IBM | Lightweight MLP-Mixer architecture |

Evaluation metrics: **MAE**, **RMSE**, **sMAPE**. Models are ranked and the best is selected automatically.

---

## 🧠 Gemini LLM Integration

After benchmarking all models, `final.py` uses **Google Gemini 2.5 Flash** to convert the raw numbers into a human-readable narrative about Delhi's climate.

### What it does

The best model's actual vs. forecasted temperatures (with dates) are sent to Gemini with a structured prompt. Gemini is instructed to act as a **climate data analyst** and explain:

1. The overall temperature trend — rising, falling, or stable
2. Any noticeable fluctuations or spikes, and when they occur
3. How closely the forecast tracks the actual temperatures
4. What a city planner or journalist would conclude about Delhi's climate in this period

The output is written in **plain English** — no model names, no metric scores — just a readable story about what the forecast reveals.

### Example output

```
Delhi's temperatures during the forecast period followed a clear seasonal arc.
Starting near 18 °C in early January, temperatures rose steadily through spring,
peaking around 38–40 °C in May. The forecast tracked this pattern closely,
with slight deviations during rapid weather shifts in March...
```

### Setup

Get a free API key at [Google AI Studio](https://aistudio.google.com/), then open `final.py` and set it on line 352:

```python
GEMINI_API_KEY = "your-key-here"
```

---

## 🖥️ Streamlit Dashboard (`app.py`)

Runs locally using **Chronos Bolt Small**. Click **⚡ Run Forecast** in the sidebar to generate predictions.

### Tabs

| Tab | Content |
|-----|---------|
| 🌡️ Raw Data | Full historical series + rolling forecast with 80% confidence band, date range selector |
| 🔬 Stationarity | ADF & KPSS test results with pass/fail badges and automated interpretation |
| 📊 ACF & PACF | Autocorrelation plots with adjustable lag slider, significant lag count |
| 🔮 Forecast & Metrics | MSE, RMSE, sMAPE, R² + error histogram + actual vs forecast scatter plot |
| 🌀 Decomposition | STL decomposition into trend, seasonal, and residual components with insights card |

---

## ⚙️ Setup & Run

### Dashboard

```bash
cd "forecasting 2"
pip install -r requirements.txt
streamlit run app.py
```

Opens at **http://localhost:8501**

### Full Benchmark (recommended: Google Colab with GPU)

1. Upload `DailyDelhiClimateTrain.csv` to the Colab session
2. Set your `GEMINI_API_KEY` in `final.py` (line 352)
3. Run all cells — packages install automatically

The script prints a model comparison table and ends with the Gemini narrative analysis.

**Optional — Generate a PDF report**
```bash
python generate_report.py
```

---

## 📁 Project Structure

```
forecasting 2/
├── app.py                                  # Streamlit dashboard (entry point)
├── final.py                                # Full benchmark script — all 8 models + Gemini
├── final (2).ipynb                         # Jupyter notebook version
├── data_bf_with_all_foundation_models.py   # Data preprocessing
├── generate_report.py                      # PDF report generator
├── final_explained.md                      # Line-by-line code walkthrough
├── Delhi_Climate_Forecasting_Report.pdf    # Generated project report
├── DailyDelhiClimateTrain.csv              # Dataset
└── requirements.txt
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Dashboard | Streamlit |
| Deep Learning | PyTorch |
| Data | pandas, numpy |
| Visualization | Plotly |
| Statistical Analysis | statsmodels (STL, ADF, KPSS, ACF/PACF) |
| Evaluation | scikit-learn |
| Foundation Models | chronos-forecasting, momentfm, uni2ts, lag-llama, timesfm, tsfm-public |
| LLM Narrative | Google Gemini 2.5 Flash (`google-generativeai`) |
