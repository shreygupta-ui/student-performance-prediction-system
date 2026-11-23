"""Train, compare and save the models.

Run from the project root:   python -m src.train_model
"""
import json
import time
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (CATEGORICAL_FEATURES, FEATURES, FEATURES_NO_G1, METRICS_PATH,
                     MODEL_PATH, NUMERIC_FEATURES, RANDOM_STATE, TARGET, TEST_SIZE)
from .data_loader import load_clean, modeling_data


def build_pipeline(model, numeric, categorical):
    """Preprocessing + model in ONE pipeline (prevents train/test leakage)."""
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical),
    ])
    return Pipeline([("prep", pre), ("model", model)])


def candidate_models():
    return {
        "Baseline (predict mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, max_depth=2, learning_rate=0.05, random_state=RANDOM_STATE),
    }


def evaluate(pipe, X_tr, y_tr, X_te, y_te):
    pipe.fit(X_tr, y_tr)
    pred = np.clip(pipe.predict(X_te), 0, 20)
    return {
        "MAE": float(mean_absolute_error(y_te, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
        "R2": float(r2_score(y_te, pred)),
    }


def run():
    full = load_clean()
    df = modeling_data(full)
    print(f"Loaded {len(full)} rows; using {len(df)} for modelling "
          f"({len(full) - len(df)} zero-grade non-completers excluded).")

    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"Train: {len(X_tr)}  Test: {len(X_te)}")

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, model in candidate_models().items():
        pipe = build_pipeline(model, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        # Cross-validation on the TRAINING part only (used to pick the winner)
        cv_rmse = -cross_val_score(pipe, X_tr, y_tr, cv=cv,
                                   scoring="neg_root_mean_squared_error")
        m = evaluate(pipe, X_tr, y_tr, X_te, y_te)
        m["CV_RMSE_mean"] = float(cv_rmse.mean())
        m["CV_RMSE_std"] = float(cv_rmse.std())
        results[name] = m
        print(f"{name:26s} MAE={m['MAE']:.3f} RMSE={m['RMSE']:.3f} "
              f"R2={m['R2']:.3f}  CV-RMSE={m['CV_RMSE_mean']:.3f}")

    # Choose the best real model (exclude the baseline) by CV RMSE
    real = {k: v for k, v in results.items() if not k.startswith("Baseline")}
    best_name = min(real, key=lambda k: real[k]["CV_RMSE_mean"])
    print("Selected model:", best_name)

    # ---- Ablation: what if the early grade G1 is NOT available? ----------
    num_no_g1 = [c for c in NUMERIC_FEATURES if c != "G1"]
    ablation = {}
    X2_tr, X2_te = X_tr[FEATURES_NO_G1], X_te[FEATURES_NO_G1]
    for name in ["Linear Regression", "Random Forest", "Gradient Boosting"]:
        pipe = build_pipeline(candidate_models()[name], num_no_g1, CATEGORICAL_FEATURES)
        ablation[name] = evaluate(pipe, X2_tr, y_tr, X2_te, y_te)

    # ---- Sensitivity: same Random Forest trained on ALL rows (incl. G3 = 0) ----
    Xf_tr, Xf_te, yf_tr, yf_te = train_test_split(
        full[FEATURES], full[TARGET], test_size=TEST_SIZE, random_state=RANDOM_STATE)
    sens = evaluate(build_pipeline(candidate_models()["Random Forest"],
                    NUMERIC_FEATURES, CATEGORICAL_FEATURES), Xf_tr, yf_tr, Xf_te, yf_te)
    sens["rows"] = int(len(full))

    # ---- Final fit of the selected pipeline on the training data ---------
    final = build_pipeline(candidate_models()[best_name],
                           NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    final.fit(X_tr, y_tr)
    test_pred = np.clip(final.predict(X_te), 0, 20)

    # Feature importance from a Random Forest fitted on the training data
    # (works whichever model was selected) + standardised Linear Regression
    # coefficients (shows direction: positive helps the grade, negative hurts).
    rf = build_pipeline(candidate_models()["Random Forest"], NUMERIC_FEATURES,
                        CATEGORICAL_FEATURES).fit(X_tr, y_tr)
    names = rf.named_steps["prep"].get_feature_names_out()
    importance = {}
    for n, v in zip(names, rf.named_steps["model"].feature_importances_):
        base = n.split("__", 1)[1]
        for c in CATEGORICAL_FEATURES:
            if base.startswith(c + "_"):
                base = c
        importance[base] = importance.get(base, 0.0) + float(v)
    importance = dict(sorted(importance.items(), key=lambda kv: -kv[1]))

    lr = build_pipeline(LinearRegression(), NUMERIC_FEATURES, CATEGORICAL_FEATURES).fit(X_tr, y_tr)
    coefs = {n.split("__", 1)[1]: float(c) for n, c in
             zip(lr.named_steps["prep"].get_feature_names_out(), lr.named_steps["model"].coef_)}
    coefs = dict(sorted(coefs.items(), key=lambda kv: -abs(kv[1])))

    # Save model + metadata (Joblib)
    joblib.dump({"pipeline": final, "features": FEATURES,
                 "model_name": best_name}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({
        "dataset_rows": int(len(full)), "modelling_rows": int(len(df)),
        "zero_grade_rows_excluded": int(len(full) - len(df)),
        "sensitivity_with_zero_grades": sens, "train_rows": int(len(X_tr)),
        "test_rows": int(len(X_te)), "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE, "best_model": best_name,
        "selection_rule": "lowest 5-fold CV RMSE on training data",
        "results": results, "ablation_without_G1": ablation,
        "feature_importance_rf": importance, "lr_coefficients": coefs,
        "test_actual": [float(v) for v in y_te],
        "test_predicted": [float(v) for v in test_pred],
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2))
    print("Saved:", MODEL_PATH.name, "and", METRICS_PATH.name)
    print("Ablation (no G1):", {k: round(v['MAE'], 3) for k, v in ablation.items()})


if __name__ == "__main__":
    run()
