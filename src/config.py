"""Central configuration: file paths, feature lists and label meanings.

Keeping everything in one place means the training script, the prediction
code and the Streamlit app always agree about which columns are used.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "student-mat.csv"
MODEL_PATH = ROOT / "models" / "student_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

TARGET = "G3"            # final grade, scale 0-20
RANDOM_STATE = 42
TEST_SIZE = 0.20

# ---------------------------------------------------------------------------
# PREDICTION SCENARIO: "early-performance prediction"
# We predict the FINAL grade (G3) using information available BEFORE the final
# exam. G1 (first-period grade) is allowed. G2 (second-period grade) is
# deliberately EXCLUDED: it is almost a copy of G3 (correlation ~0.90) and
# would make the model look better than it is at giving EARLY warnings.
# ---------------------------------------------------------------------------
USE_G1 = True

# 38 of the 395 students have a final grade of exactly 0, and ALL of them have
# 0 recorded absences (they most likely left the course, so nothing was logged).
# Left in, the model wrongly learns "0 absences -> may fail". We therefore train
# on students who completed the course (G3 > 0). The full-data result is still
# reported in metrics.json ("sensitivity_with_zero_grades") for transparency.
EXCLUDE_ZERO_GRADES = True

NUMERIC_FEATURES = [
    "age", "studytime", "failures", "absences", "G1",
    "Medu", "Fedu", "famrel", "freetime", "goout", "health",
    "Dalc", "Walc", "traveltime",
]
CATEGORICAL_FEATURES = [
    "school", "sex", "address", "famsup", "schoolsup",
    "paid", "internet", "higher", "activities", "romantic",
]
FEATURES_NO_G1 = [c for c in NUMERIC_FEATURES if c != "G1"] + CATEGORICAL_FEATURES
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns that must NEVER be used as inputs (leakage / not known in advance)
EXCLUDED_COLUMNS = ["G2", "G3"]

STUDYTIME_LABELS = {1: "Less than 2 hours/week", 2: "2 to 5 hours/week",
                    3: "5 to 10 hours/week", 4: "More than 10 hours/week"}
EDU_LABELS = {0: "None", 1: "Primary (up to 4th grade)", 2: "5th to 9th grade",
              3: "Secondary education", 4: "Higher education"}
SCHOOL_LABELS = {"GP": "Gabriel Pereira", "MS": "Mousinho da Silveira"}

# Performance categories on the 0-20 scale (10 is the pass mark in Portugal)
CATEGORIES = [
    (16, "Excellent", "#15803d"),
    (13, "Good", "#2563eb"),
    (10, "Satisfactory (Pass)", "#d97706"),
    (0,  "At Risk (Below Pass)", "#dc2626"),
]

def categorize(score: float):
    """Return (label, colour) for a predicted 0-20 score."""
    for threshold, label, colour in CATEGORIES:
        if score >= threshold:
            return label, colour
    return CATEGORIES[-1][1], CATEGORIES[-1][2]
