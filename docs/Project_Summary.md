# Project Summary (abstract, problem, objectives, methods, results)

**Abstract.** A Machine Learning web application estimates a student's final grade (0–20) early in the term using the public UCI Student Performance dataset (Mathematics, 395 students). G2 and G3 are excluded as inputs to avoid leakage; G1 is allowed. Thirty-eight zero-grade records (all with 0 absences) are excluded from training. Linear Regression, Random Forest and Gradient Boosting are compared with a baseline. The selected model (Linear Regression, chosen by cross-validation) reaches test MAE 1.247, RMSE 1.542 and R² 0.753 on 72 unseen students (baseline MAE 2.572). Without G1 the R² falls to about 0.26.

**Problem statement.** Weak students are usually identified only after final exams. Build a system that estimates the final score from information available beforehand, categorises it and suggests improvements.

**Objectives.** Clean data and avoid leakage; compare algorithms; evaluate with MAE, RMSE, R² and cross-validation; save the model; build a Streamlit dashboard; give categories and study tips.

**Existing system.** Manual review of marks after exams: late, subjective, ignores background factors, gives no guidance.

**Proposed system.** ML model predicting before the exam from academic, family and school factors; outputs score, category, likely range and suggestions; shows its own metrics and limits.

**Methodology.** Load → clean → EDA → scale/encode (Pipeline) → 80/20 split → 5-fold CV → train 3 models + baseline → test metrics → select → save (Joblib) → serve with Streamlit.

**Future scope.** More data, SHAP explanations, login/history, batch prediction, dropout model, retraining on college data.

**References.** Cortez & Silva (2008), FUBUTEC; UCI Student Performance dataset (https://archive.ics.uci.edu/dataset/320/student+performance); Breiman (2001) Random Forests; Friedman (2001) Gradient boosting; Pedregosa et al. (2011) Scikit-learn.

The full report is in `docs/Project_Report.docx`.
