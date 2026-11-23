"""Prediction logic used by the web app (and testable on its own)."""
import json
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
from .config import FEATURES, METRICS_PATH, MODEL_PATH, categorize

# Valid ranges taken from the dataset's documentation
LIMITS = {"age": (15, 22), "studytime": (1, 4), "failures": (0, 3),
          "absences": (0, 75), "G1": (0, 20), "Medu": (0, 4), "Fedu": (0, 4),
          "famrel": (1, 5), "freetime": (1, 5), "goout": (1, 5),
          "health": (1, 5), "Dalc": (1, 5), "Walc": (1, 5), "traveltime": (1, 4)}
CHOICES = {"school": ["GP", "MS"], "sex": ["F", "M"], "address": ["U", "R"],
           "famsup": ["yes", "no"], "schoolsup": ["yes", "no"], "paid": ["yes", "no"],
           "internet": ["yes", "no"], "higher": ["yes", "no"],
           "activities": ["yes", "no"], "romantic": ["yes", "no"]}


@lru_cache(maxsize=1)
def load_model():
    """Load the saved model. If it is missing or was saved by an incompatible
    scikit-learn version, retrain automatically (takes a few seconds)."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        from .train_model import run   # imported here to keep start-up light
        run()
        return joblib.load(MODEL_PATH)


def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text())


def validate(inputs: dict) -> None:
    """Raise ValueError with a clear message if any input is invalid."""
    missing = [f for f in FEATURES if f not in inputs]
    if missing:
        raise ValueError(f"Missing inputs: {missing}")
    for k, (lo, hi) in LIMITS.items():
        v = inputs[k]
        if not isinstance(v, (int, float, np.integer, np.floating)) or not lo <= v <= hi:
            raise ValueError(f"'{k}' must be a number between {lo} and {hi} (got {v!r}).")
    for k, opts in CHOICES.items():
        if inputs[k] not in opts:
            raise ValueError(f"'{k}' must be one of {opts} (got {inputs[k]!r}).")


def predict_score(inputs: dict) -> dict:
    """Return predicted G3 (0-20), category, colour and an uncertainty band."""
    validate(inputs)
    bundle = load_model()
    row = pd.DataFrame([{f: inputs[f] for f in FEATURES}])
    score = float(np.clip(bundle["pipeline"].predict(row)[0], 0, 20))
    label, colour = categorize(score)
    mae = load_metrics()["results"][bundle["model_name"]]["MAE"]
    return {"score": round(score, 2), "category": label, "colour": colour,
            "low": round(max(0, score - mae), 1), "high": round(min(20, score + mae), 1)}


def suggestions(inputs: dict, score: float) -> list:
    """Simple rule-based advice. These are study tips, not model outputs."""
    tips = []
    if inputs["studytime"] <= 2:
        tips.append("Increase weekly self-study time gradually (aim for 5+ hours/week) "
                    "and use a fixed daily timetable.")
    if inputs["absences"] >= 10:
        tips.append("Attendance is high-risk: try to reduce absences, and catch up on "
                    "missed topics the same week.")
    if inputs["failures"] >= 1:
        tips.append("Revise the earlier-failed subject topics first and ask teachers for "
                    "remedial or doubt-clearing sessions.")
    if inputs["G1"] < 10:
        tips.append("Your first-period grade is below pass level. Focus on basics now; "
                    "early correction has the largest effect.")
    if inputs["schoolsup"] == "no" and score < 12:
        tips.append("Consider extra educational support from the school or teachers.")
    if inputs["paid"] == "no" and score < 12:
        tips.append("Extra paid or group coaching classes in Mathematics may help.")
    if inputs["internet"] == "yes":
        tips.append("Use your internet access for quality learning (video lectures, "
                    "practice platforms) rather than distractions.")
    if inputs["goout"] >= 4 or inputs["Walc"] >= 4:
        tips.append("Balance social time and weekend routines so they do not eat into "
                    "study and rest.")
    if inputs["higher"] == "no":
        tips.append("Talk to a mentor about career goals; a clear goal improves motivation.")
    if not tips:
        tips.append("Good habits detected. Keep your current routine and practise "
                    "previous question papers regularly.")
    return tips[:6]
