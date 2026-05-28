"""
Generate a professional academic PDF report for the
Delhi Climate Forecasting project.

Run:  python generate_report.py
Output: Delhi_Climate_Forecasting_Report.pdf
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether, NextPageTemplate
)
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate
import datetime

# ─── Colours ────────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#1a237e")
GOLD   = colors.HexColor("#f9a825")
LIGHT  = colors.HexColor("#e8eaf6")
WHITE  = colors.white
DARK   = colors.HexColor("#000000")
GREY   = colors.HexColor("#424242")
ROW_A  = colors.HexColor("#e8eaf6")
ROW_B  = colors.white
ACCENT = colors.HexColor("#283593")

W, H = A4   # 595.27 x 841.89 pts

# ─── Page callbacks ─────────────────────────────────────────────────────────
def add_header_footer(canvas_obj, doc):
    canvas_obj.saveState()
    # Header bar
    canvas_obj.setFillColor(NAVY)
    canvas_obj.rect(0, H - 1.2*cm, W, 1.2*cm, fill=1, stroke=0)
    canvas_obj.setFillColor(GOLD)
    canvas_obj.setFont("Helvetica-Bold", 8)
    canvas_obj.drawString(1.5*cm, H - 0.85*cm,
                          "Delhi Climate Forecasting  |  Technical Report")
    canvas_obj.setFillColor(WHITE)
    canvas_obj.drawRightString(W - 1.5*cm, H - 0.85*cm,
                               "Foundation Time Series Models")

    # Footer bar
    canvas_obj.setFillColor(NAVY)
    canvas_obj.rect(0, 0, W, 1.0*cm, fill=1, stroke=0)
    canvas_obj.setFillColor(GOLD)
    canvas_obj.setFont("Helvetica", 8)
    canvas_obj.drawString(1.5*cm, 0.35*cm, "Kaggle — Daily Delhi Climate Dataset")
    canvas_obj.setFillColor(WHITE)
    canvas_obj.drawCentredString(W/2, 0.35*cm, f"Page {doc.page}")
    canvas_obj.drawRightString(W - 1.5*cm, 0.35*cm,
                               "New Delhi, India  |  2013 – 2017")
    canvas_obj.restoreState()


def title_page_bg(canvas_obj, doc):
    """Full-bleed navy background for page 1 only."""
    canvas_obj.saveState()
    canvas_obj.setFillColor(NAVY)
    canvas_obj.rect(0, 0, W, H, fill=1, stroke=0)
    # Gold accent strip
    canvas_obj.setFillColor(GOLD)
    canvas_obj.rect(0, H * 0.38, W, 0.4*cm, fill=1, stroke=0)
    canvas_obj.rect(0, H * 0.38 - 0.15*cm, W, 0.08*cm, fill=1, stroke=0)
    canvas_obj.restoreState()


# ─── Styles ─────────────────────────────────────────────────────────────────
def make_styles():
    base = getSampleStyleSheet()
    s = {}

    s["title_main"] = ParagraphStyle(
        "title_main", parent=base["Title"],
        fontSize=26, textColor=WHITE, fontName="Helvetica-Bold",
        alignment=TA_CENTER, spaceAfter=6, leading=32
    )
    s["title_sub"] = ParagraphStyle(
        "title_sub", parent=base["Normal"],
        fontSize=13, textColor=GOLD, fontName="Helvetica-BoldOblique",
        alignment=TA_CENTER, spaceAfter=4
    )
    s["title_meta"] = ParagraphStyle(
        "title_meta", parent=base["Normal"],
        fontSize=10, textColor=colors.HexColor("#b0bec5"),
        fontName="Helvetica", alignment=TA_CENTER, spaceAfter=2
    )
    s["section_h"] = ParagraphStyle(
        "section_h", parent=base["Heading1"],
        fontSize=15, textColor=WHITE, fontName="Helvetica-Bold",
        alignment=TA_LEFT, spaceBefore=14, spaceAfter=6,
        backColor=NAVY, leftIndent=-0.5*cm, rightIndent=-0.5*cm,
        borderPad=(4, 8, 4, 8)
    )
    s["subsection_h"] = ParagraphStyle(
        "subsection_h", parent=base["Heading2"],
        fontSize=12, textColor=NAVY, fontName="Helvetica-Bold",
        spaceBefore=10, spaceAfter=4, borderColor=GOLD,
        borderWidth=0, leftIndent=0
    )
    s["body"] = ParagraphStyle(
        "body", parent=base["Normal"],
        fontSize=11, textColor=DARK, fontName="Helvetica",
        leading=17, spaceAfter=6, alignment=TA_JUSTIFY
    )
    s["bullet"] = ParagraphStyle(
        "bullet", parent=base["Normal"],
        fontSize=11, textColor=DARK, fontName="Helvetica",
        leading=16, spaceAfter=3, leftIndent=14, bulletIndent=4
    )
    s["code"] = ParagraphStyle(
        "code", parent=base["Code"],
        fontSize=9, textColor=colors.HexColor("#1565c0"),
        fontName="Courier", backColor=colors.HexColor("#f5f5f5"),
        borderColor=colors.HexColor("#bdbdbd"), borderWidth=0.5,
        borderPad=4, leading=13, spaceAfter=4
    )
    s["table_h"] = ParagraphStyle(
        "table_h", parent=base["Normal"],
        fontSize=10, textColor=WHITE, fontName="Helvetica-Bold",
        alignment=TA_CENTER
    )
    s["table_c"] = ParagraphStyle(
        "table_c", parent=base["Normal"],
        fontSize=10, textColor=DARK, fontName="Helvetica",
        alignment=TA_LEFT, leading=14
    )
    s["table_cc"] = ParagraphStyle(
        "table_cc", parent=base["Normal"],
        fontSize=10, textColor=DARK, fontName="Helvetica",
        alignment=TA_CENTER, leading=14
    )
    s["caption"] = ParagraphStyle(
        "caption", parent=base["Normal"],
        fontSize=9, textColor=GREY, fontName="Helvetica-Oblique",
        alignment=TA_CENTER, spaceAfter=8
    )
    s["note"] = ParagraphStyle(
        "note", parent=base["Normal"],
        fontSize=10, textColor=NAVY,
        fontName="Helvetica-Bold", leading=15,
        backColor=colors.HexColor("#e3f2fd"), borderPad=6,
        borderColor=NAVY, borderWidth=0.8,
        spaceAfter=8
    )
    return s

# ─── Helper builders ─────────────────────────────────────────────────────────
def section_header(text, s):
    return Paragraph(f"&nbsp;&nbsp;{text}", s["section_h"])


def hr():
    return HRFlowable(width="100%", thickness=0.5, color=GOLD, spaceAfter=6)


def build_table(headers, rows, s, col_widths=None):
    h_row = [Paragraph(h, s["table_h"]) for h in headers]
    data = [h_row]
    for i, row in enumerate(rows):
        bg = ROW_A if i % 2 == 0 else ROW_B
        data.append([Paragraph(str(c), s["table_c"]) for c in row])

    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [ROW_A, ROW_B]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9fa8da")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 0), (-1, 0), [NAVY]),
    ])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(style)
    return t


# ─── Content builders ───────────────────────────────────────────────────────

def title_page(s):
    story = []
    story.append(Spacer(1, 5.5*cm))
    story.append(Paragraph(
        "Delhi Daily Temperature Forecasting", s["title_main"]))
    story.append(Paragraph(
        "Using Foundation Time Series Models", s["title_main"]))
    story.append(Spacer(1, 0.6*cm))
    story.append(Paragraph(
        "Data Preprocessing · Feature Engineering · Model Evaluation",
        s["title_sub"]))
    story.append(Spacer(1, 3.5*cm))
    story.append(Paragraph("Technical Report", s["title_meta"]))
    story.append(Paragraph(
        f"Generated: {datetime.date.today().strftime('%B %d, %Y')}",
        s["title_meta"]))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Dataset: Kaggle — Daily Delhi Climate Dataset",
                            s["title_meta"]))
    story.append(Paragraph(
        "Temporal Coverage: January 1, 2013 — April 24, 2017",
        s["title_meta"]))
    story.append(Paragraph("Location: New Delhi, India", s["title_meta"]))
    story.append(PageBreak())
    return story


def section_dataset(s):
    story = []
    story.append(section_header("1.  Dataset Description", s))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph(
        "The dataset is the <b>Daily Delhi Climate</b> dataset sourced from Kaggle. "
        "It records four meteorological variables measured daily in New Delhi, India, "
        "spanning <b>January 1, 2013 to April 24, 2017</b> — a total of approximately "
        "<b>1,462 daily observations</b> across 5 columns.",
        s["body"]))

    story.append(Paragraph("1.1  Column Summary", s["subsection_h"]))
    headers = ["Column", "Data Type", "Unit", "Role", "Description"]
    rows = [
        ["date",         "DateTime", "—",    "Index",   "Calendar date (YYYY-MM-DD). Used as time index."],
        ["meantemp",     "Float",    "°C",   "TARGET",  "Mean daily temperature. Primary variable to forecast."],
        ["humidity",     "Float",    "%",    "Feature", "Mean daily relative humidity."],
        ["wind_speed",   "Float",    "km/h", "Feature", "Mean daily wind speed."],
        ["meanpressure", "Float",    "hPa",  "Feature", "Mean daily atmospheric pressure."],
    ]
    story.append(build_table(headers, rows, s,
                             col_widths=[2.6*cm, 1.8*cm, 1.4*cm, 1.6*cm, 8.0*cm]))
    story.append(Paragraph(
        "Table 1 — Dataset columns, units, and roles.", s["caption"]))

    story.append(Paragraph("1.2  Climate Context", s["subsection_h"]))
    story.append(Paragraph(
        "New Delhi experiences a strong semi-arid climate with a marked annual temperature cycle. "
        "Winter temperatures can drop to approximately 8–10 °C (December–January), while peak "
        "summer temperatures exceed 38–40 °C (May–June). This wide seasonal swing "
        "creates a challenging yet well-structured time-series forecasting scenario with "
        "clear annual periodicity.",
        s["body"]))

    story.append(Paragraph("1.3  Dataset Statistics", s["subsection_h"]))
    headers2 = ["Attribute", "Value"]
    rows2 = [
        ["Total rows",            "~1,462 daily observations"],
        ["Total columns",         "5 (1 index + 4 features)"],
        ["Time range",            "2013-01-01 → 2017-04-24"],
        ["Temporal resolution",   "Daily (enforced via asfreq('D'))"],
        ["Target variable",       "meantemp (°C)"],
        ["Approximate temp range","8 °C (winter) — 40 °C (summer)"],
        ["Source",                "Kaggle — Daily Delhi Climate Dataset"],
    ]
    story.append(build_table(headers2, rows2, s,
                             col_widths=[6*cm, 9.5*cm]))
    story.append(Paragraph("Table 2 — High-level dataset statistics.", s["caption"]))

    return story


def section_preprocessing(s):
    story = []
    story.append(section_header("2.  Preprocessing Pipeline", s))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "Preprocessing was applied in strict chronological order to prevent "
        "data leakage. All statistics (IQR bounds, etc.) are derived "
        "<b>exclusively from the training split</b> and then applied to both splits.",
        s["body"]))

    steps = [
        ("Step 1 — Data Loading & Frequency Enforcement",
         "The CSV file is read with <font name='Courier'>pd.read_csv()</font>. "
         "The <b>date</b> column is parsed with <font name='Courier'>pd.to_datetime()</font> "
         "and set as the DataFrame index. "
         "<font name='Courier'>.asfreq('D')</font> is then applied to enforce a "
         "daily frequency, inserting <font name='Courier'>NaN</font> for any "
         "missing calendar dates to guarantee a contiguous time series."),

        ("Step 2 — STL Decomposition (Exploratory)",
         "Seasonal-Trend decomposition using LOESS (STL) is applied to the "
         "full series (<font name='Courier'>period=365, robust=True</font>) to "
         "visualise and understand the underlying structure before any split. "
         "The decomposition separates the series into three additive components: "
         "<b>Trend</b> (long-term direction), <b>Seasonality</b> (365-day annual pattern), "
         "and <b>Residual</b> (noise). This step is exploratory only — no transformation "
         "is applied to the data."),

        ("Step 3 — Augmented Dickey–Fuller (ADF) Stationarity Test",
         "The ADF test (<font name='Courier'>autolag='AIC'</font>) checks the null "
         "hypothesis H₀: the series contains a unit root (non-stationary). "
         "Decision rule: p-value &lt; 0.05 → reject H₀ → stationary. "
         "If the raw series is non-stationary, <b>first-order differencing</b> "
         "(subtracting each value from the previous) is applied and the test is "
         "repeated. Output includes the ADF statistic, p-value, number of lags "
         "used, and critical values at 1%, 5%, and 10%."),

        ("Step 4 — Chronological Train / Test Split (80 / 20)",
         "The dataset is split into training (first 80 % ≈ 1,170 rows) and "
         "test (last 20 % ≈ 292 rows) sets using index-based slicing "
         "(<font name='Courier'>.iloc[:split_idx]</font> / "
         "<font name='Courier'>.iloc[split_idx:]</font>). "
         "<b>No shuffling</b> is performed — temporal order is strictly preserved. "
         "<font name='Courier'>.copy()</font> is called on both slices to "
         "prevent silent aliasing. This split is performed <b>before</b> any "
         "statistics are computed, ensuring zero data leakage."),

        ("Step 5 — Outlier Detection & Imputation",
         "Two strategies are used depending on the feature:"),
    ]

    for title, body in steps:
        story.append(Paragraph(title, s["subsection_h"]))
        story.append(Paragraph(body, s["body"]))

    # Outlier sub-table
    story.append(Paragraph(
        "<b>5a — meanpressure: Domain-Knowledge Threshold</b>", s["bullet"]))
    story.append(Paragraph(
        "Values outside the physically possible atmospheric range "
        "(&lt; 900 hPa or &gt; 1,100 hPa) are flagged as outliers. "
        "This threshold is based on meteorological domain knowledge and "
        "is therefore safe to apply before the split without leakage.",
        s["body"]))

    story.append(Paragraph(
        "<b>5b — meantemp, humidity, wind_speed: IQR Method</b>", s["bullet"]))
    story.append(Paragraph(
        "Inter-Quartile Range bounds are calculated <b>from the training set only</b>:",
        s["body"]))

    headers = ["Formula", "Description"]
    rows = [
        ["Q1 = train[col].quantile(0.25)", "25th percentile of training values"],
        ["Q3 = train[col].quantile(0.75)", "75th percentile of training values"],
        ["IQR = Q3 − Q1",                 "Spread of the middle 50 %"],
        ["Lower = Q1 − 1.5 × IQR",        "Lower fence — anything below is an outlier"],
        ["Upper = Q3 + 1.5 × IQR",        "Upper fence — anything above is an outlier"],
    ]
    story.append(build_table(headers, rows, s,
                             col_widths=[7.5*cm, 8.0*cm]))
    story.append(Paragraph("Table 3 — IQR outlier detection formulas.", s["caption"]))

    story.append(Paragraph(
        "Detected outliers in both train and test sets are replaced with "
        "<font name='Courier'>NaN</font>. Gaps are then filled using a three-step "
        "imputation chain: "
        "<font name='Courier'>.interpolate()</font> (linear interpolation between "
        "neighbours) → "
        "<font name='Courier'>.ffill()</font> (forward fill for trailing edges) → "
        "<font name='Courier'>.bfill()</font> (backward fill for leading edges). "
        "The result is a clean, gap-free time series in both splits.",
        s["body"]))

    return story


def section_feature_engineering(s):
    story = []
    story.append(section_header("3.  Feature Engineering", s))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "Feature engineering expands the original 4 columns into 15 informative "
        "predictors by adding temporal, autoregressive, and statistical context. "
        "All features are computed <b>after</b> the train/test split and outlier "
        "handling. A dedicated function "
        "<font name='Courier'>add_features(df_split, train_tail=None)</font> handles "
        "both splits to avoid leakage.",
        s["body"]))

    story.append(Paragraph(
        "<b>Leakage prevention:</b> The test set is seeded with the last 30 rows of "
        "the training set (<font name='Courier'>train_tail=train.iloc[-30:]</font>). "
        "These seed rows are dropped after feature computation, so only true "
        "test-period rows remain with fully valid rolling/lag features.",
        s["note"]))

    story.append(Paragraph(
        "Summary: 4 original features + 11 engineered features = <b>15 total features</b>.",
        s["body"]))

    # ── Group 1: Temporal ──
    story.append(Paragraph("3.1  Temporal / Cyclical Features  (4 features)", s["subsection_h"]))
    story.append(Paragraph(
        "These features encode the position of each observation within the calendar year. "
        "Cyclical (sin/cos) encoding is preferred over raw integers because it preserves "
        "the circular nature of time: day 365 and day 1 are adjacent, not 364 apart.",
        s["body"]))

    headers = ["Feature", "Formula", "Range", "Rationale"]
    rows = [
        ["month",     "df.index.month",
         "1 – 12",    "Coarse monthly label; captures month-level seasonality (e.g., April vs. December)."],
        ["day_of_year", "df.index.day_of_year",
         "1 – 365",   "Granular yearly position; finer resolution than month alone."],
        ["sin_doy",   "sin(2π × day_of_year / 365)",
         "[-1, 1]",   "Projects annual cycle onto unit circle. Ensures day 365 and day 1 are numerically close (no discontinuity at year boundary)."],
        ["cos_doy",   "cos(2π × day_of_year / 365)",
         "[-1, 1]",   "Complement of sin_doy. Together, sin and cos encode the full 360° position on the annual cycle."],
    ]
    story.append(build_table(headers, rows, s,
                             col_widths=[2.5*cm, 4.2*cm, 1.8*cm, 7.0*cm]))
    story.append(Paragraph("Table 4 — Temporal / cyclical features.", s["caption"]))

    story.append(Paragraph(
        "<b>Why sin/cos instead of raw day-of-year?</b>  A linear day-of-year value "
        "assigns a numeric distance of 364 between day 1 (January 1) and day 365 "
        "(December 31), even though they are consecutive days. Sine/cosine encoding "
        "maps the year onto a circle where this distance is approximately 0, "
        "correctly representing seasonal continuity across the year boundary.",
        s["note"]))

    # ── Group 2: Lag Features ──
    story.append(Paragraph("3.2  Lag / Autoregressive Features  (3 features)", s["subsection_h"]))
    story.append(Paragraph(
        "Lag features provide the model with direct access to past temperature "
        "values, capturing autocorrelation in the time series. "
        "<font name='Courier'>shift(n)</font> shifts the series forward by n steps "
        "so that the value at row t equals the original value at row t − n.",
        s["body"]))

    headers = ["Feature", "Formula", "Lag", "Rationale"]
    rows = [
        ["temp_lag_1",  "meantemp.shift(1)",  "1 day",
         "Yesterday's temperature is typically the strongest single predictor of today's temperature due to high day-to-day autocorrelation."],
        ["temp_lag_7",  "meantemp.shift(7)",  "7 days",
         "One-week lag captures weekly weather cycles and atmospheric patterns that tend to repeat on a ~7-day timescale."],
        ["temp_lag_14", "meantemp.shift(14)", "14 days",
         "Two-week lag captures slower synoptic-scale patterns and medium-term seasonal transitions."],
    ]
    story.append(build_table(headers, rows, s,
                             col_widths=[2.8*cm, 4.0*cm, 1.5*cm, 7.2*cm]))
    story.append(Paragraph("Table 5 — Lag / autoregressive features.", s["caption"]))

    # ── Group 3: Rolling Statistics ──
    story.append(Paragraph("3.3  Rolling Statistics  (4 features)", s["subsection_h"]))
    story.append(Paragraph(
        "Rolling features summarise temperature behaviour over a sliding window "
        "of past observations. All windows are computed on "
        "<font name='Courier'>meantemp.shift(1)</font> — the shifted series — "
        "so today's value is never included in its own window, "
        "preventing look-ahead bias.",
        s["body"]))

    headers = ["Feature", "Formula", "Window", "Rationale"]
    rows = [
        ["temp_rollmean_7",
         "shift(1).rolling(7).mean()",
         "7 days",
         "Short-term smoothed temperature trend. Reduces noise and highlights the direction of change over the past week."],
        ["temp_rollstd_7",
         "shift(1).rolling(7).std()",
         "7 days",
         "Week-scale temperature volatility. A high value indicates rapidly changing weather conditions."],
        ["temp_rollmean_30",
         "shift(1).rolling(30).mean()",
         "30 days",
         "Monthly smoothed trend. Captures seasonal momentum and gradual climate transitions (e.g., winter → spring warming)."],
        ["temp_rollstd_30",
         "shift(1).rolling(30).std()",
         "30 days",
         "Month-scale volatility. Elevated values indicate a period of seasonal transition with high day-to-day variability."],
    ]
    story.append(build_table(headers, rows, s,
                             col_widths=[3.2*cm, 4.5*cm, 1.5*cm, 6.3*cm]))
    story.append(Paragraph("Table 6 — Rolling statistic features.", s["caption"]))

    story.append(Paragraph(
        "<b>Note on shift-before-roll:</b>  Without <font name='Courier'>.shift(1)</font>, "
        "the rolling window at time t would include the temperature value at t itself. "
        "This would create a circular dependency where the target leaks into its own "
        "feature. Applying shift first guarantees that the window at time t covers "
        "only observations t−1, t−2, …, t−w.",
        s["note"]))

    # ── Feature Summary Table ──
    story.append(Paragraph("3.4  Feature Engineering Summary", s["subsection_h"]))
    headers = ["Group", "Count", "Feature Names"]
    rows = [
        ["Original (raw)",         "4",
         "meantemp, humidity, wind_speed, meanpressure"],
        ["Temporal / Cyclical",    "4",
         "month, day_of_year, sin_doy, cos_doy"],
        ["Lag / Autoregressive",   "3",
         "temp_lag_1, temp_lag_7, temp_lag_14"],
        ["Rolling Statistics",     "4",
         "temp_rollmean_7, temp_rollstd_7, temp_rollmean_30, temp_rollstd_30"],
        ["TOTAL",                  "15", "—"],
    ]
    story.append(build_table(headers, rows, s,
                             col_widths=[4.5*cm, 1.8*cm, 9.2*cm]))
    story.append(Paragraph(
        "Table 7 — Complete feature inventory after engineering.", s["caption"]))

    return story


def section_split(s):
    story = []
    story.append(section_header("4.  Train / Test Split Strategy", s))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph(
        "Time-series data must never be split randomly because doing so "
        "would allow the model to train on future information. "
        "A strict <b>chronological 80 / 20 split</b> is used.",
        s["body"]))

    headers = ["Split", "Ratio", "Approx. Rows", "Approximate Date Range"]
    rows = [
        ["Training set", "80 %", "~1,170 days", "January 1, 2013 — December 2015"],
        ["Test set",     "20 %", "~292 days",   "January 2016 — April 24, 2017"],
    ]
    story.append(build_table(headers, rows, s,
                             col_widths=[3.5*cm, 2*cm, 3.5*cm, 6.5*cm]))
    story.append(Paragraph("Table 8 — Train / test split sizes.", s["caption"]))

    story.append(Paragraph("4.1  Walk-Forward Rolling Forecast", s["subsection_h"]))
    story.append(Paragraph(
        "Rather than making a single static 292-day prediction, the models use "
        "a <b>walk-forward (rolling) forecasting strategy</b> to simulate real-world "
        "deployment conditions:",
        s["body"]))

    wf_rows = [
        ["Step", "Action"],
        ["1", "Use the full training context (up to the model's context cap) to predict the next 14 days."],
        ["2", "Append the 14 actual (ground-truth) observations to the context."],
        ["3", "Repeat until all 292 test days are covered (~21 iterations of 14 days)."],
        ["4", "Concatenate all 14-day chunks into the final 292-day forecast array."],
    ]
    t = Table(wf_rows, colWidths=[1.5*cm, 14.0*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR",  (0, 0), (-1, 0), WHITE),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [ROW_A, ROW_B]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9fa8da")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
    ]))
    story.append(t)
    story.append(Paragraph(
        "Table 9 — Walk-forward rolling forecast procedure.", s["caption"]))

    story.append(Paragraph(
        "<b>Why walk-forward?</b>  Updating the context with actual observed values "
        "after each prediction window prevents compounding errors that occur when "
        "a model's own (potentially imperfect) predictions are fed back as future "
        "context. It closely mirrors how a deployed forecasting system would operate.",
        s["note"]))

    story.append(Paragraph("4.2  Model-Specific Context Caps", s["subsection_h"]))
    headers = ["Model", "Context Cap", "Implication"]
    rows = [
        ["Chronos-T5-Tiny",    "512 days",   "Uses most recent 512 days of history at each step."],
        ["Chronos-T5-Small",   "2,048 days", "Can see the entire training set at once (1,170 days < 2,048)."],
        ["Chronos-Bolt-Small", "512 days",   "Uses most recent 512 days of history at each step."],
    ]
    story.append(build_table(headers, rows, s,
                             col_widths=[4.5*cm, 3.0*cm, 8.0*cm]))
    story.append(Paragraph("Table 10 — Context window caps per model.", s["caption"]))

    return story


def section_models(s):
    story = []
    story.append(section_header("5.  Models", s))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "All three models used in this project belong to the <b>Chronos</b> family, "
        "developed by Amazon Research. They are large neural networks "
        "pre-trained on millions of diverse real-world time series. "
        "They operate in a <b>zero-shot</b> manner — no fine-tuning on the Delhi "
        "dataset is performed. The models receive a normalised context window "
        "and return a probabilistic (or deterministic) forecast.",
        s["body"]))

    story.append(Paragraph("5.1  Input Normalisation (All Models)", s["subsection_h"]))
    story.append(Paragraph(
        "All Chronos models were trained on z-score normalised data. "
        "Normalisation is applied before every prediction call and "
        "reversed afterwards:",
        s["body"]))
    norm_data = [
        ["Step",    "Formula",                        "Purpose"],
        ["Forward", "(x − μ) / (σ + ε)",              "Scale context to zero-mean, unit-variance. ε = 1×10⁻⁸ prevents division by zero."],
        ["Inverse", "prediction × (σ + ε) + μ",       "Convert model output back to original °C scale."],
    ]
    t = Table(norm_data, colWidths=[2.2*cm, 5.0*cm, 8.3*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR",  (0, 0), (-1, 0), WHITE),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [ROW_A, ROW_B]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9fa8da")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
    ]))
    story.append(t)
    story.append(Paragraph("Table 11 — Normalisation applied before/after each prediction.", s["caption"]))

    # Per-model cards
    model_info = [
        (
            "5.2  Chronos-T5-Tiny",
            "amazon/chronos-t5-tiny",
            [
                ["Attribute",           "Value"],
                ["Parameters",          "~8 million"],
                ["Architecture",        "T5 encoder-decoder transformer (adapted for time series)"],
                ["Context window cap",  "512 days"],
                ["Prediction type",     "Probabilistic — generates 20 independent sample paths"],
                ["Final prediction",    "Median of 20 samples (denormalised)"],
                ["API class",           "ChronosPipeline"],
                ["Inference speed",     "Fastest (lowest memory footprint)"],
            ],
            "Best suited for rapid iteration and environments with limited GPU memory. "
            "With only ~8 M parameters it sacrifices some capacity for speed."
        ),
        (
            "5.3  Chronos-T5-Small",
            "amazon/chronos-t5-small",
            [
                ["Attribute",           "Value"],
                ["Parameters",          "~46 million"],
                ["Architecture",        "T5 encoder-decoder transformer"],
                ["Context window cap",  "2,048 days — can ingest the entire training set"],
                ["Prediction type",     "Probabilistic — 20 sample paths"],
                ["Final prediction",    "Median of 20 samples (denormalised)"],
                ["API class",           "ChronosPipeline"],
                ["Inference speed",     "Slowest (highest GPU memory usage)"],
            ],
            "Highest modelling capacity among the three. Its 2,048-day context window "
            "allows it to see the complete ~1,170-day training history in a single pass, "
            "giving it the longest temporal memory."
        ),
        (
            "5.4  Chronos-Bolt-Small",
            "amazon/chronos-bolt-small",
            [
                ["Attribute",           "Value"],
                ["Parameters",          "~46 million"],
                ["Architecture",        "Optimised / distilled Chronos variant"],
                ["Context window cap",  "512 days"],
                ["Prediction type",     "Deterministic — single forecast path (no num_samples)"],
                ["Final prediction",    "Direct output (no aggregation required)"],
                ["API class",           "BaseChronosPipeline"],
                ["Inference speed",     "Fast (optimised inference engine)"],
            ],
            "Production-oriented variant. Bolt uses a distilled inference engine that "
            "trades probabilistic uncertainty estimates for faster, deterministic "
            "predictions. Suitable for latency-sensitive applications."
        ),
    ]

    for title, hf_id, attr_rows, summary in model_info:
        story.append(Paragraph(title, s["subsection_h"]))
        story.append(Paragraph(
            f"HuggingFace identifier: <font name='Courier'>{hf_id}</font>", s["body"]))

        t = Table(attr_rows, colWidths=[4.2*cm, 11.3*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
            ("TEXTCOLOR",  (0, 0), (-1, 0), WHITE),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [ROW_A, ROW_B]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9fa8da")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
        ]))
        story.append(t)
        story.append(Paragraph(summary, s["body"]))
        story.append(Spacer(1, 0.15*cm))

    # Comparison table
    story.append(Paragraph("5.5  Model Comparison at a Glance", s["subsection_h"]))
    headers = ["Model", "Params", "Context Cap", "Output Type", "Speed"]
    rows = [
        ["Chronos-T5-Tiny",    "~8 M",  "512 days",   "Probabilistic (median of 20)",  "Fastest"],
        ["Chronos-T5-Small",   "~46 M", "2,048 days", "Probabilistic (median of 20)",  "Slowest"],
        ["Chronos-Bolt-Small", "~46 M", "512 days",   "Deterministic (single path)",   "Fast"],
    ]
    story.append(build_table(headers, rows, s,
                             col_widths=[4.0*cm, 1.8*cm, 2.8*cm, 5.2*cm, 1.7*cm]))
    story.append(Paragraph(
        "Table 15 — Side-by-side comparison of the three Chronos models.", s["caption"]))

    return story


def section_metrics(s):
    story = []
    story.append(section_header("6.  Evaluation Metrics", s))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "Four complementary metrics are computed for each model on the "
        "292-day test set. Together they cover magnitude of error, "
        "scale-free percentage error, and proportion of variance explained.",
        s["body"]))

    metrics = [
        (
            "6.1  MSE — Mean Squared Error",
            "MSE = mean[ (y − ŷ)² ]",
            "°C²",
            "Primary loss for neural network training. Squaring errors means large "
            "deviations are penalised disproportionately. Less interpretable than RMSE "
            "due to squared units.",
            [("Range", "[0, ∞)  — 0 is perfect"), ("Sensitivity", "High — dominated by large errors")]
        ),
        (
            "6.2  RMSE — Root Mean Squared Error  (Primary Ranking Metric)",
            "RMSE = √MSE = √( mean[ (y − ŷ)² ] )",
            "°C",
            "Square root of MSE restores the original unit (°C), making the error directly "
            "interpretable: 'on average, predictions deviate by X °C from actual temperature.' "
            "Models are ranked by RMSE ascending (lower = better).",
            [("Range", "[0, ∞)  — 0 is perfect"), ("Sensitivity", "High — penalises large errors")]
        ),
        (
            "6.3  sMAPE — Symmetric Mean Absolute Percentage Error  (Secondary Metric)",
            "sMAPE = mean( |y − ŷ| / ( (|y| + |ŷ|) / 2 ) ) × 100",
            "%",
            "Scale-free metric expressed as a percentage. The 'symmetric' formulation "
            "averages actual and predicted in the denominator, so over-prediction and "
            "under-prediction are penalised equally. Critical for this dataset because "
            "Delhi winter temperatures approach 0 °C — standard MAPE would produce "
            "extreme values and asymmetric penalties in those conditions.",
            [("Range", "[0, 200 %]  — 0 % is perfect"), ("Near-zero safe", "Yes — denominator never collapses")]
        ),
        (
            "6.4  R² — Coefficient of Determination",
            "R² = 1 − ( SS_res / SS_tot ) where SS_res = Σ(y − ŷ)²,  SS_tot = Σ(y − ȳ)²",
            "— (dimensionless)",
            "Measures the proportion of variance in the actual temperatures that is "
            "explained by the model's predictions. "
            "R² = 1.0 → perfect forecast. "
            "R² = 0.0 → no better than always predicting the mean. "
            "R² < 0.0 → worse than predicting the mean.",
            [("Range", "(−∞, 1]  — 1 is perfect"), ("Baseline", "R² = 0 corresponds to the naive mean predictor")]
        ),
    ]

    for title, formula, unit, explanation, props in metrics:
        story.append(KeepTogether([
            Paragraph(title, s["subsection_h"]),
            Paragraph(f"<b>Formula:</b>  <font name='Courier'>{formula}</font>   |   "
                      f"<b>Unit:</b>  {unit}", s["body"]),
            Paragraph(explanation, s["body"]),
        ]))
        prop_rows = [[k, v] for k, v in props]
        t = Table(prop_rows, colWidths=[3.5*cm, 12.0*cm])
        t.setStyle(TableStyle([
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [ROW_A, ROW_B]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9fa8da")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.25*cm))

    # Summary table
    story.append(Paragraph("6.5  Metrics Summary", s["subsection_h"]))
    headers = ["Metric", "Formula (compact)", "Unit", "Best Value", "Role"]
    rows = [
        ["MSE",   "mean((y−ŷ)²)",                      "°C²", "0",    "Loss function; penalises large errors"],
        ["RMSE",  "√MSE",                              "°C",   "0",   "Primary ranking; interpretable scale"],
        ["sMAPE", "|y−ŷ| / avg(|y|,|ŷ|) × 100",       "%",   "0 %",  "Scale-free; robust at near-zero values"],
        ["R²",    "1 − SS_res / SS_tot",                "—",   "1.0",  "Variance explained"],
    ]
    story.append(build_table(headers, rows, s,
                             col_widths=[2.0*cm, 4.8*cm, 1.4*cm, 2.2*cm, 5.1*cm]))
    story.append(Paragraph(
        "Table 16 — Evaluation metrics reference.", s["caption"]))

    return story


# ─── Assemble & Build ────────────────────────────────────────────────────────
def build_pdf(output_path):
    s = make_styles()

    # Two page templates: one for title (no header/footer), one for body
    doc = BaseDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=1.8*cm,
        rightMargin=1.8*cm,
        topMargin=1.8*cm,
        bottomMargin=1.6*cm,
    )

    body_frame = Frame(
        doc.leftMargin,
        doc.bottomMargin + 1.0*cm,    # leave room for footer
        doc.width,
        doc.height - 1.2*cm - 0.2*cm, # leave room for header
        id="body_frame"
    )
    title_frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width,
        doc.height,
        id="title_frame"
    )

    doc.addPageTemplates([
        PageTemplate(id="title_page", frames=[title_frame],
                     onPage=title_page_bg),
        PageTemplate(id="body_page",  frames=[body_frame],
                     onPage=add_header_footer),
    ])

    story = []

    # Title page (template 1)
    story += title_page(s)

    # Body pages (template 2)
    story.append(NextPageTemplate("body_page"))

    story += section_dataset(s)
    story.append(Spacer(1, 0.3*cm))
    story += section_preprocessing(s)
    story.append(Spacer(1, 0.3*cm))
    story += section_feature_engineering(s)
    story.append(Spacer(1, 0.3*cm))
    story += section_split(s)
    story.append(Spacer(1, 0.3*cm))
    story += section_models(s)
    story.append(Spacer(1, 0.3*cm))
    story += section_metrics(s)

    doc.build(story)
    print(f"\n  PDF saved: {output_path}\n")


if __name__ == "__main__":
    import os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "Delhi_Climate_Forecasting_Report.pdf")
    build_pdf(out)
