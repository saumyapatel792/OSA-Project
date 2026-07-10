"""
OSA-CDSS  ·  Database & Preprocessing Layer
============================================
Handles:
  • SQLite patient database (create / seed / CRUD)
  • Data retrieval with filters & sorting
  • Full preprocessing pipeline (cleaning → encoding → scaling)
  • Preprocessing audit log stored in DB
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import datetime, json, os

DB_PATH = "osa_patients.db"

# ── SCHEMA ────────────────────────────────────────────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id              TEXT PRIMARY KEY,
    age             INTEGER,
    bmi             REAL,
    snoring         INTEGER,
    neck            REAL,
    gender          TEXT,
    ahi             REAL,
    risk            TEXT,
    pred_score      REAL,
    spo2            INTEGER,
    bp_sys          INTEGER,
    bp_dia          INTEGER,
    smoker          INTEGER,
    diabetes        INTEGER,
    created_at      TEXT,
    updated_at      TEXT
);

CREATE TABLE IF NOT EXISTS preprocessing_log (
    log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT,
    step            TEXT,
    details         TEXT,
    records_affected INTEGER
);

CREATE TABLE IF NOT EXISTS audit_trail (
    audit_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT,
    action          TEXT,
    patient_id      TEXT,
    user            TEXT,
    changes         TEXT
);
"""

SEED_DATA = [
    ("P-0001", 52, 31.2, 4, 42.0, "Male",   28.4, "High",     0.87, 88,  138, 88,  1, 0),
    ("P-0002", 34, 24.1, 1, 35.0, "Female", 4.2,  "Low",      0.12, 97,  112, 72,  0, 0),
    ("P-0003", 61, 35.8, 5, 46.0, "Male",   42.1, "Severe",   0.96, 83,  152, 96,  1, 1),
    ("P-0004", 45, 28.4, 3, 39.0, "Female", 14.7, "Moderate", 0.61, 93,  124, 80,  0, 0),
    ("P-0005", 38, 26.7, 2, 37.0, "Male",   7.3,  "Low",      0.28, 96,  118, 76,  0, 0),
    ("P-0006", 57, 33.1, 4, 44.0, "Male",   33.6, "High",     0.91, 86,  144, 92,  1, 1),
    ("P-0007", 29, 22.3, 1, 33.0, "Female", 2.1,  "Low",      0.08, 98,  108, 68,  0, 0),
    ("P-0008", 66, 37.4, 5, 48.0, "Male",   51.3, "Severe",   0.98, 80,  162, 102, 1, 1),
    ("P-0009", 41, 27.1, 2, 38.0, "Female", 9.8,  "Low",      0.34, 95,  120, 78,  0, 0),
    ("P-0010", 54, 30.5, 3, 41.0, "Male",   22.4, "Moderate", 0.72, 91,  132, 84,  0, 1),
    ("P-0011", 48, 29.8, 4, 40.0, "Female", 18.6, "Moderate", 0.65, 92,  128, 82,  0, 0),
    ("P-0012", 63, 34.2, 5, 45.0, "Male",   38.9, "Severe",   0.94, 84,  148, 94,  1, 1),
]

# ── DB INIT ───────────────────────────────────────────────────────────────────
def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    conn.executescript(SCHEMA)
    now = datetime.datetime.now().isoformat()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM patients")
    if cur.fetchone()[0] == 0:
        for row in SEED_DATA:
            cur.execute("""
                INSERT INTO patients VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (*row, now, now))
        _log(conn, "SEED", "Initial patient data loaded", len(SEED_DATA))
    conn.commit()
    conn.close()

# ── CRUD ──────────────────────────────────────────────────────────────────────
def get_all_patients(sort_by="id", order="ASC", risk_filter=None, gender_filter=None):
    conn = get_connection()
    query = "SELECT * FROM patients WHERE 1=1"
    params = []
    if risk_filter and risk_filter != "All":
        query += " AND risk = ?"
        params.append(risk_filter)
    if gender_filter and gender_filter != "All":
        query += " AND gender = ?"
        params.append(gender_filter)
    safe_cols = {"id", "age", "bmi", "ahi", "spo2", "pred_score", "risk", "created_at"}
    if sort_by in safe_cols:
        query += f" ORDER BY {sort_by} {order}"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def get_patient_by_id(patient_id):
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM patients WHERE id=?", conn, params=(patient_id,))
    conn.close()
    return df.iloc[0] if not df.empty else None

def insert_patient(data: dict):
    conn = get_connection()
    now = datetime.datetime.now().isoformat()
    bp_parts = data.get("bp", "120/80").split("/")
    conn.execute("""
        INSERT INTO patients VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        data["id"], data["age"], data["bmi"], data["snoring"], data["neck"],
        data["gender"], data.get("ahi", 0), data.get("risk", "Unknown"),
        data.get("pred", 0), data["spo2"], int(bp_parts[0]), int(bp_parts[1]),
        int(data.get("smoker", 0)), int(data.get("diabetes", 0)), now, now
    ))
    _audit(conn, "INSERT", data["id"], "system", str(data))
    _log(conn, "INSERT", f"New patient {data['id']} added", 1)
    conn.commit()
    conn.close()

def update_patient_risk(patient_id, new_risk, new_pred):
    conn = get_connection()
    now = datetime.datetime.now().isoformat()
    conn.execute(
        "UPDATE patients SET risk=?, pred_score=?, updated_at=? WHERE id=?",
        (new_risk, new_pred, now, patient_id)
    )
    _audit(conn, "UPDATE", patient_id, "system", f"risk={new_risk}, pred={new_pred}")
    conn.commit()
    conn.close()

def delete_patient(patient_id):
    conn = get_connection()
    conn.execute("DELETE FROM patients WHERE id=?", (patient_id,))
    _audit(conn, "DELETE", patient_id, "system", "Record removed")
    conn.commit()
    conn.close()

# ── RETRIEVAL QUERIES ─────────────────────────────────────────────────────────
def get_stats():
    conn = get_connection()
    stats = {}
    stats["total"] = pd.read_sql_query("SELECT COUNT(*) as c FROM patients", conn).iloc[0]["c"]
    stats["severe"] = pd.read_sql_query("SELECT COUNT(*) as c FROM patients WHERE risk='Severe'", conn).iloc[0]["c"]
    stats["avg_ahi"] = pd.read_sql_query("SELECT AVG(ahi) as v FROM patients", conn).iloc[0]["v"]
    stats["avg_bmi"] = pd.read_sql_query("SELECT AVG(bmi) as v FROM patients", conn).iloc[0]["v"]
    stats["risk_dist"] = pd.read_sql_query(
        "SELECT risk, COUNT(*) as count FROM patients GROUP BY risk ORDER BY count DESC", conn
    )
    stats["gender_risk"] = pd.read_sql_query(
        "SELECT gender, risk, COUNT(*) as count FROM patients GROUP BY gender, risk", conn
    )
    conn.close()
    return stats

def get_preprocessing_log():
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM preprocessing_log ORDER BY timestamp DESC LIMIT 50", conn
    )
    conn.close()
    return df

def get_audit_trail():
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM audit_trail ORDER BY timestamp DESC LIMIT 30", conn
    )
    conn.close()
    return df

# ── PREPROCESSING PIPELINE ────────────────────────────────────────────────────
def preprocess_patients():
    """
    Full preprocessing pipeline:
    1. Load raw data from DB
    2. Handle missing values
    3. Remove outliers (IQR)
    4. Encode categoricals (Gender, Risk)
    5. Feature engineering (BMI category, age group, MAP)
    6. Normalise numerics with StandardScaler
    7. Return processed DataFrame + scaler + report
    """
    conn = get_connection()
    raw = pd.read_sql_query("SELECT * FROM patients", conn)
    conn.close()

    report = []
    step_count = [0]

    def log_step(name, detail, n):
        step_count[0] += 1
        report.append({
            "step": step_count[0],
            "name": name,
            "detail": detail,
            "records": n
        })
        _log_step_to_db(name, detail, n)

    n0 = len(raw)
    log_step("Load", f"Loaded {n0} raw records from SQLite DB", n0)

    # ── Step 1: Drop irrelevant / audit columns
    df = raw.drop(columns=["created_at", "updated_at"], errors="ignore")
    log_step("Drop Columns", "Removed audit timestamp columns (created_at, updated_at)", len(df))

    # ── Step 2: Missing value check & imputation
    missing = df.isnull().sum()
    total_missing = missing.sum()
    if total_missing > 0:
        df["age"].fillna(df["age"].median(), inplace=True)
        df["bmi"].fillna(df["bmi"].median(), inplace=True)
        df["spo2"].fillna(df["spo2"].median(), inplace=True)
        df["snoring"].fillna(df["snoring"].mode()[0], inplace=True)
        df["gender"].fillna("Unknown", inplace=True)
        log_step("Imputation", f"Filled {total_missing} missing values (median/mode strategy)", len(df))
    else:
        log_step("Missing Values", "No missing values found — dataset is complete", len(df))

    # ── Step 3: Outlier removal using IQR on AHI and BMI
    before = len(df)
    for col in ["ahi", "bmi", "spo2"]:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
    removed = before - len(df)
    log_step("Outlier Removal", f"IQR method on ahi/bmi/spo2 — {removed} outliers removed", len(df))

    # ── Step 4: Label encode Gender
    le_gender = LabelEncoder()
    df["gender_enc"] = le_gender.fit_transform(df["gender"])
    log_step("Encode Gender", "LabelEncoder: Female=0, Male=1", len(df))

    # ── Step 5: Encode Risk as ordinal
    risk_map = {"Low": 0, "Moderate": 1, "High": 2, "Severe": 3}
    df["risk_enc"] = df["risk"].map(risk_map)
    log_step("Encode Risk", "Ordinal encoding: Low=0, Moderate=1, High=2, Severe=3", len(df))

    # ── Step 6: Feature Engineering
    df["bmi_category"] = pd.cut(df["bmi"],
        bins=[0, 18.5, 25, 30, 40, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese", "Morbidly Obese"]
    ).astype(str)
    df["age_group"] = pd.cut(df["age"],
        bins=[0, 30, 45, 60, 100],
        labels=["Young Adult", "Middle Age", "Senior", "Elderly"]
    ).astype(str)
    df["map"] = ((df["bp_sys"] + 2 * df["bp_dia"]) / 3).round(1)  # Mean Arterial Pressure
    df["ahi_log"] = np.log1p(df["ahi"])  # Log-transform skewed AHI
    df["spo2_deficit"] = 100 - df["spo2"]
    log_step("Feature Engineering",
             "Added: bmi_category, age_group, MAP (mean arterial pressure), ahi_log, spo2_deficit", len(df))

    # ── Step 7: Normalise numeric features
    num_cols = ["age", "bmi", "neck", "snoring", "ahi", "spo2", "bp_sys", "bp_dia", "map", "ahi_log", "spo2_deficit"]
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[num_cols] = scaler.fit_transform(df[num_cols])
    log_step("Normalisation", f"StandardScaler (z-score) applied to {len(num_cols)} numeric features", len(df))

    return df, df_scaled, scaler, report

# ── INTERNAL HELPERS ──────────────────────────────────────────────────────────
def _log(conn, step, details, n):
    conn.execute(
        "INSERT INTO preprocessing_log (timestamp, step, details, records_affected) VALUES (?,?,?,?)",
        (datetime.datetime.now().isoformat(), step, details, n)
    )

def _log_step_to_db(step, detail, n):
    try:
        conn = get_connection()
        _log(conn, step, detail, n)
        conn.commit()
        conn.close()
    except Exception:
        pass

def _audit(conn, action, patient_id, user, changes):
    conn.execute(
        "INSERT INTO audit_trail (timestamp, action, patient_id, user, changes) VALUES (?,?,?,?,?)",
        (datetime.datetime.now().isoformat(), action, patient_id, user, changes)
    )
