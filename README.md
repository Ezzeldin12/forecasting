# 🌡️ Forecasting 2 — Delhi Climate AI Forecasting

> A time-series forecasting app that predicts Delhi's daily mean temperature using 6 state-of-the-art AI foundation models with zero-shot inference, compared side-by-side in an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

---

## 📌 Overview

This project evaluates 6 modern **AI foundation models for time-series forecasting** on a real climate dataset — Delhi's daily mean temperature from 2013 to 2017. All models run in **zero-shot mode** (no fine-tuning required), making it a direct benchmark of out-of-the-box forecasting capability.

The app includes **STL decomposition** to separate trend, seasonality, and residuals, along with **ACF/PACF analysis** to understand autocorrelation. All 6 model predictions are visualized together in an interactive Streamlit dashboard so their performance can be compared visually and numerically.

---

## ✨ Features

- 🤖 **6 Foundation Models** — benchmark state-of-the-art time-series models in one app
- 🔮 **Zero-Shot Inference** — no fine-tuning needed; models generalize directly
- 📉 **STL Decomposition** — separate trend, seasonality, and residual components
- 📊 **ACF/PACF Analysis** — autocorrelation and partial autocorrelation visualizations
- 🆚 **Model Comparison** — side-by-side forecast plots and error metrics (MAE, RMSE, MAPE)
- 🖥️ **Streamlit Dashboard** — fully interactive web app
- 📄 **PDF Report** — generated summary report of all model results

---

## 🤖 Foundation Models

| Model | Organization | Type |
|-------|-------------|------|
| **Chronos** | Amazon | Probabilistic transformer |
| **MOMENT** | Carnegie Mellon / Databricks | Masked autoencoder |
| **Moirai** | Salesforce (uni2ts) | Universal time-series model |
| **Lag-Llama** | HuggingFace | LLM-based forecaster |
| **TimesFM** | Google | Foundation model for forecasting |
| **TinyTimeMixer** | IBM | Lightweight mixer architecture |

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| App Framework | Streamlit | Interactive web dashboard |
| Deep Learning | PyTorch | Model inference backbone |
| Data Processing | pandas, numpy | Data manipulation |
| Visualization | Plotly | Interactive charts |
| Statistical Analysis | statsmodels | STL decomposition, ACF/PACF |
| Evaluation | scikit-learn | MAE, RMSE, MAPE metrics |
| Models | chronos, momentfm, uni2ts, lag-llama, timesfm, tsfm | Foundation models |

---

## 📁 Project Structure

```
forecasting 2/
├── app.py                                  # Streamlit dashboard — entry point
├── final.py                                # Core forecasting engine
├── final.ipynb (2)                         # Jupyter notebook version
├── data_bf_with_all_foundation_models.py   # Data preprocessing
├── generate_report.py                      # PDF report generator
├── final_explained.md                      # Code documentation & methodology
├── Delhi_Climate_Forecasting_Report.pdf    # Final project report
├── DailyDelhiClimateTrain.csv              # Dataset
└── requirements.txt
```

---

## 🗂️ Dataset

| Property | Value |
|----------|-------|
| File | `DailyDelhiClimateTrain.csv` |
| Rows | 1,462 daily observations |
| Date Range | Jan 1, 2013 → Apr 24, 2017 |
| Target | `meantemp` (daily mean temperature in °C) |
| Features | `date`, `meantemp`, `humidity`, `wind_speed`, `meanpressure` |
| Source | Kaggle — Delhi Climate Dataset |

---

## ⚙️ Setup & Run

**1. Navigate to the project folder**
```bash
cd "forecasting 2"
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

> Note: Foundation model libraries (chronos, momentfm, uni2ts, etc.) are large. Installation may take a few minutes.

**3. Launch the Streamlit dashboard**
```bash
streamlit run app.py
```

The app opens in your browser at **http://localhost:8501**

**Optional — Generate a PDF report**
```bash
python generate_report.py
```

---

## 📸 Screenshots

> _Add screenshots of the dashboard forecast comparison and STL decomposition here_
