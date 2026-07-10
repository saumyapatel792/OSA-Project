# MESA Dataset Setup Guide
## OSA-CDSS · Alliance University

---

## What is MESA?

The **Multi-Ethnic Study of Atherosclerosis (MESA) Sleep Study** is a large-scale,
PSG-verified clinical dataset of 6,800+ participants maintained by the NHLBI (US National
Heart, Lung, and Blood Institute). It contains every feature your model uses:

| Your Feature     | MESA Column        | Description                         |
|------------------|--------------------|-------------------------------------|
| Age              | `age5c`            | Age at exam 5                       |
| BMI              | `bmi5c`            | Body mass index                     |
| Neck             | `neckgirth5c`      | Neck circumference (cm)             |
| Snoring          | `slpapnea5`        | Sleep-disordered breathing (0–4)    |
| Gender           | `gender1`          | 1=Male, 2=Female                    |
| SpO2             | `avgO2nrem5`       | Avg overnight oxygen saturation (%) |
| Diabetes         | `diabetes5c`       | Diabetes diagnosis (0/1)            |
| AHI (target)     | `ahi_c0h3a`        | Apnea-hypopnea index (events/hr)    |

---

## Step-by-Step Download Instructions

### Step 1 — Register at BioLINCC
Go to: **https://biolincc.nhlbi.nih.gov/studies/mesa/**

Click **"Request Dataset"** and create a free account. You will need:
- Your institutional email (Alliance University email preferred)
- A brief statement of intended use (e.g., "Academic research on OSA prediction
  using machine learning as part of coursework at Alliance University")

### Step 2 — Complete the Data Use Agreement
BioLINCC will send you a Data Use Agreement (DUA) by email. This is a standard
academic data sharing agreement. Sign and return it electronically.
Approval typically takes **2–5 business days**.

### Step 3 — Download the CSV
After approval, log in and download the **MESA Sleep Study ancillary dataset**.
The file will be named something like:
```
mesa-sleep-dataset-0.5.0.csv
```

### Step 4 — Place the file in the project folder
Copy the downloaded CSV into the same folder as `app.py`:

```
DPD_Project/
├── app.py
├── mesa_loader.py          ← handles the loading
├── database.py
├── pdf_report.py
├── requirements.txt
├── MESA_SETUP_GUIDE.md
└── mesa-sleep-dataset-0.5.0.csv   ← PUT THE FILE HERE
```

Any filename containing the word `mesa` will be auto-detected.
You can also place it in a `data/` subfolder.

### Step 5 — Restart the app
```bash
streamlit run app.py
```

The green banner **"✓ MESA Sleep Study Loaded"** will appear on every page,
confirming the model is now trained on real clinical data.

---

## What Changes When MESA Is Connected

| Before (Synthetic)            | After (MESA)                            |
|-------------------------------|------------------------------------------|
| 2,000 randomly generated rows | 6,800 real PSG-verified patient records  |
| Artificially balanced classes | Real-world AHI distribution              |
| Weights learned from formula  | Weights learned from actual patient data |
| AUC ~0.82 (synthetic)         | Expected AUC 0.85–0.91 (literature)     |
| No clinical validity          | Validated against gold-standard PSG      |

---

## AHI → Risk Mapping Used (AASM 2023 Guidelines)

| AHI Range    | Risk Level | Classification |
|--------------|------------|----------------|
| < 5 /hr      | Low        | Normal         |
| 5 – 14 /hr   | Moderate   | Mild OSA       |
| 15 – 29 /hr  | High       | Moderate OSA   |
| ≥ 30 /hr     | Severe     | Severe OSA     |

This is implemented in `mesa_loader.py → ahi_to_risk()`.

---

## Troubleshooting

**"MESA CSV not found"** — Check that the file is in the project folder and
contains the word "mesa" in its filename.

**"Could not find columns for: ['Neck']"** — Your CSV version uses a different
column name. Open `mesa_loader.py`, find `COLUMN_ALIASES["Neck"]`, and add
the correct column name from your CSV header row.

**"FileNotFoundError"** — The file path is wrong. Run `ls *.csv` in the project
folder to see what CSV files are present.

---

## Alternative: Kaggle OSA Dataset (Instant, No Registration)

If you need data immediately while waiting for MESA approval:

1. Go to: https://www.kaggle.com/datasets/search?q=sleep+apnea+ahi+clinical
2. Download any dataset with AHI, BMI, Neck, SpO2 columns
3. Rename to `mesa.csv` and place in the project folder
4. Open `mesa_loader.py` and update `COLUMN_ALIASES` to match
   the Kaggle dataset's column names

---

*OSA-CDSS · Alliance University · Dept. of Data Science & Sleep Medicine*
