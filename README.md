# Student Performance Prediction System Using Machine Learning

## Live Demo

🌐 [Open Student Performance Prediction System](https://student-performance-shrey.streamlit.app)

---
B.Tech CSE (AI & ML) minor project. A Streamlit web app that estimates a student's final grade (0–20) from academic and study-related information, using a model trained on the real UCI Student Performance dataset.

## What is in this folder

| Path | Purpose |
|---|---|
| `app/streamlit_app.py` | The web app (5 pages) |
| `src/config.py` | Paths, features, categories |
| `src/data_loader.py` | Load and clean the data |
| `src/train_model.py` | Train, compare and save models |
| `src/predict.py` | Validation, prediction, suggestions |
| `data/student-mat.csv` | UCI dataset (395 rows, 33 columns) |
| `models/` | Saved model (`student_model.joblib`) and results (`metrics.json`) |
| `tests/test_app.py` | Automatic check that every page works |
| `docs/` | Report, slides, viva guide, diagrams, screenshots |

## Dataset

- **Source:** P. Cortez, *Student Performance*, UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/320/student+performance
- **File used:** `student-mat.csv` (Mathematics course), separator `;`, 395 students, 33 attributes, no missing values. Grades use a 0–20 scale; 10 is a pass.
- **Target:** `G3` (final grade). **Input grade:** `G1` (first period). `G2` and `G3` are never inputs (data leakage).
- **How to get it:** a copy is already in `data/`. To re-download: open the UCI link, click *Download*, unzip, and copy `student-mat.csv` into the `data` folder.
- Citation: Cortez, P. & Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance.* FUBUTEC 2008, pp. 5–12. (Check the dataset DOI with the "Cite" button on the UCI page.)
- **Dataset license:** CC BY 4.0 — © original dataset contributors. Source: https://archive.ics.uci.edu/dataset/320/student+performance. The dataset is licensed separately from this project's MIT-licensed code.
- 38 students with `G3 = 0` all have 0 absences (probably course leavers). They are excluded from training (see `src/config.py`, `EXCLUDE_ZERO_GRADES`).

## Run on your computer (Windows steps; macOS/Linux in brackets)

1. **Install Python 3.10 or newer** from https://www.python.org/downloads/ . On Windows tick **"Add python.exe to PATH"** during installation. Check: `python --version`.
2. **Install VS Code** (optional but recommended) from https://code.visualstudio.com/ .
3. **Open the project folder:** VS Code → File → Open Folder → select `student_performance_system`. Open the terminal: Terminal → New Terminal.
4. **Create a virtual environment**
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - [macOS/Linux: `python3 -m venv venv` then `source venv/bin/activate`]
   - If PowerShell blocks activation, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
5. **Install requirements:** `pip install -r requirements.txt`
6. **Train the model** (a few seconds, prints the metrics): `python -m src.train_model`
7. **Start the app:** `streamlit run app/streamlit_app.py`
8. **Open the browser** at http://localhost:8501 (it usually opens by itself). Stop the app with `Ctrl + C`.
9. **Optional self-test:** `python tests/test_app.py` should print `OK` for all 5 pages and the prediction click.

Always run commands from inside the `student_performance_system` folder.

## Sample inputs and expected output

Typical student: G1 = 12, study time 2–5 h, 4 absences, 0 failures, school GP, internet yes, family support yes (the form's default values) → about **12.9 / 20, Satisfactory (Pass)**.
High achiever (G1 = 17, study >10 h, 0 absences, parents' education 4) → about **17.1, Excellent**.
At risk (G1 = 6, study <2 h, 25 absences, 2 failures, no higher education) → about **7.7, At Risk**.
Values come from the model saved with the results in `metrics.json`; if you retrain, small differences are possible.

## Results (test set, 72 students)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (mean) | 2.572 | 3.106 | -0.001 |
| **Linear Regression (selected)** | 1.247 | 1.542 | 0.753 |
| Random Forest | 1.116 | 1.471 | 0.776 |
| Gradient Boosting | 1.098 | 1.410 | 0.794 |

The model was selected by 5-fold cross-validation on the training data (not by the test set). Without the first-period grade G1, R² falls to about 0.26. Always rerun training to reproduce your own numbers.

## Troubleshooting

| Problem | Fix |
|---|---|
| `'python' is not recognized` | Reinstall Python and tick "Add to PATH", or use `py` instead of `python` |
| `No module named ...` | Activate the virtual environment, then `pip install -r requirements.txt` |
| `No module named 'src'` | Run commands from the project's root folder, and use `python -m src.train_model` |
| `streamlit is not recognized` | Use `python -m streamlit run app/streamlit_app.py` |
| Dataset not found | Put `student-mat.csv` in the `data` folder |
| Model not found / version warning | The app retrains automatically; or run `python -m src.train_model` |
| Old numbers still showing after retraining | Stop the app (Ctrl+C) and start it again (results are cached) |
| Port already in use | `streamlit run app/streamlit_app.py --server.port 8502` |
| Charts blank | Refresh the page; use Chrome or Edge |

## Deploy online (GitHub + Streamlit Community Cloud)

Not performed or verified by the author of this package; follow and test it yourself.

1. Create a free account at https://github.com and a new **public** repository (e.g. `student-performance-prediction`).
2. Upload the whole project folder contents (drag and drop in the browser, or `git init`, `git add .`, `git commit -m "first commit"`, `git remote add origin <repo-url>`, `git push -u origin main`). Make sure these are included: `app/`, `src/`, `data/student-mat.csv`, `models/`, `.streamlit/config.toml`, `requirements.txt`. The repository root must contain `requirements.txt`.
3. Go to https://share.streamlit.io and sign in with GitHub.
4. Click **Create app** → *Deploy a public app from GitHub*. Choose your repository, branch `main`, **Main file path: `app/streamlit_app.py`**. (Under *Advanced settings* choose Python 3.11 or 3.12.)
5. Click **Deploy**. The first build takes a few minutes. Open the link it gives you and test the Predict button yourself before telling anyone it works.
6. If the saved model was made with a different scikit-learn version, the app retrains automatically on first start (a few extra seconds).
7. Free apps sleep after inactivity; the first visit then takes some time to wake up.

## Limitations

Small dataset (357 students used) from two Portuguese schools; relies strongly on G1; does not predict dropout; shows association not cause. Predictions are estimates, not guaranteed results, and must not be the only basis for decisions about a student.
