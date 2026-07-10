# Obstructive Sleep Apnea - Clinical Decision Support System (OSA-CDSS)
### Clinical Decision Support System

---

##  Overview

**OSA-CDSS** is a machine learning-powered Obstructive Sleep Apnea (OSA) Clinical Decision Support System. It enables clinicians to assess patient risk profiles, visualize cohort statistics, manage patient records, and generate clinical PDF reports.

The system uses a soft-voting ensemble model combining **Random Forest** and **Gradient Boosting** classifiers. It is designed to run either on a realistic synthetic baseline or train directly on gold-standard polysomnography (PSG) data from the **Multi-Ethnic Study of Atherosclerosis (MESA) Sleep Study** (6,800+ participants).

---

##  Features

-  **Clinical Dashboard**: Overview of cohort statistics, model performance metrics (Accuracy, Sensitivity, Specificity, AUC), active clinical alerts, and risk distribution.
-  **Risk Assessment Tool**: Input patient anthropometrics and vitals (Age, BMI, Neck Circumference, Snoring Intensity, Resting SpO2, and comorbidities) to calculate the probability of OSA with a visual gauge and clinical recommendations based on AASM guidelines.
- **Patient Registry**: Interactive database explorer with sorting, filtering, and a detailed vitals/indicators view for individual cohorts. Includes automated PDF report generation.
- **ML Model Technical View**: Feature importance rankings, confusion matrix, model hyperparameter specifications, and training/validation loss curves.
- **Data Management**: Full CRUD interface for the underlying SQLite database, along with logs from the data preprocessing pipeline (missing value imputation, IQR outlier removal, z-score normalization).

---

##  Tech Stack

- **Frontend**: Streamlit
- **Backend / Database**: SQLite 3 (handled via Python `sqlite3`)
- **Machine Learning**: Scikit-learn (Random Forest, Gradient Boosting, Voting Classifier)
- **Data Visualization**: Plotly Express & Plotly Graph Objects
- **Report Generation**: ReportLab (automated PDF compilation)
- **Data Processing**: Pandas, NumPy

---

## Getting Started

### Prerequisites

- **Python**: version 3.10 to 3.14 (tested on Python 3.14.5)
- **Pip**: standard python package installer

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/OSA-Project.git
   cd OSA-Project
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - **macOS / Linux**:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App

Start the Streamlit application using the local environment:
```bash
streamlit run app.py
```
The application will launch and be available in your browser at `http://localhost:8501`.

---

##  Database & Schema

The application uses an SQLite database named `osa_patients.db` located in the root directory. It contains three main tables:
1. `patients`: Stores patient demographics, vitals, AHI score, risk category, and ML prediction scores.
2. `preprocessing_log`: Tracks data cleaning steps, imputation methods, and outlier count.
3. `audit_trail`: Audits all insert, update, and delete actions performed on patient records.

On the first run, the database is automatically created and seeded with baseline sample records if no data exists.

---

##  Training with MESA Dataset (Recommended)

To transition the system from synthetic training data to real PSG-verified clinical data:
1. Register and request access at [BioLINCC MESA Sleep Study](https://biolincc.nhlbi.nih.gov/studies/mesa/).
2. Once approved, download the ancillary Sleep dataset.
3. Place the CSV file (e.g., `mesa-sleep-dataset-0.5.0.csv`) directly in the project folder.
4. Restart the application. The system will automatically detect the file, process it, and train the ensemble models on real clinical records.

Refer to [MESA_SETUP_GUIDE.md](MESA_SETUP_GUIDE.md) for full instructions and column mappings.

---

## 📄 License & Attributions

- Developed as part of coursework at Alliance University.
- MESA Sleep Study data provided courtesy of the National Heart, Lung, and Blood Institute (NHLBI).
- For clinical decision support only. Not a substitute for a full laboratory PSG or home sleep apnea test (HSAT).
