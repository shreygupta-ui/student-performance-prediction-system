"""Generates the architecture, data-flow and workflow diagrams (PNG).
Run:  python docs/make_diagrams.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path(__file__).parent / "diagrams"
OUT.mkdir(exist_ok=True)
C = {"blue": "#4f46e5", "sky": "#0ea5e9", "green": "#15803d", "amber": "#d97706", "grey": "#475569"}


def box(ax, x, y, w, h, text, color, fs=10, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc,
            fontsize=fs, fontweight="bold", wrap=True)


def arrow(ax, p1, p2, label=None):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16,
                                 lw=1.8, color="#334155"))
    if label:
        if p1[0] == p2[0]:   # vertical arrow: put the label beside it
            ax.text(p1[0] + 0.15, (p1[1] + p2[1]) / 2, label, ha="left", va="center",
                    fontsize=8, color="#334155")
        else:
            ax.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + 0.17, label, ha="center",
                    fontsize=8, color="#334155")


def canvas(w, h, title):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    ax.text(w / 2, h - 0.35, title, ha="center", fontsize=14, fontweight="bold", color="#0f172a")
    return fig, ax


# 1. ARCHITECTURE (layers) ---------------------------------------------------
fig, ax = canvas(11, 6.6, "System Architecture")
layers = [
    (5.0, "PRESENTATION LAYER", C["blue"], ["Home", "Student Prediction", "Data Analysis", "Model Evaluation", "About"],
     "Streamlit web app (app/streamlit_app.py)"),
    (3.45, "APPLICATION LAYER", C["sky"], ["Input validation", "Prediction logic", "Category + tips"],
     "src/predict.py"),
    (1.9, "MACHINE LEARNING LAYER", C["green"], ["Preprocessing pipeline", "Model training", "Evaluation"],
     "src/train_model.py"),
    (0.35, "DATA / STORAGE LAYER", C["amber"], ["student-mat.csv (UCI)", "student_model.joblib", "metrics.json"],
     "data/ and models/"),
]
for y, name, col, items, sub in layers:
    ax.add_patch(FancyBboxPatch((0.3, y), 10.4, 1.25, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="#f1f5f9", ec=col, lw=2))
    ax.text(0.5, y + 1.02, name, fontsize=9, fontweight="bold", color=col)
    ax.text(10.5, y + 1.02, sub, fontsize=8, color=C["grey"], ha="right")
    n = len(items); w = 10.0 / n
    for i, it in enumerate(items):
        box(ax, 0.45 + i * w, y + 0.12, w - 0.2, 0.62, it, col, 9)
for y in (4.95, 3.4, 1.85):
    arrow(ax, (5.5, y), (5.5, y - 0.2))
fig.savefig(OUT / "architecture.png", dpi=170, bbox_inches="tight"); plt.close(fig)

# 2. DATA FLOW ---------------------------------------------------------------
fig, ax = canvas(12, 5.2, "Data Flow Diagram")
box(ax, 0.2, 2.1, 1.9, 1.0, "UCI Student\nDataset (CSV)", C["amber"])
box(ax, 2.6, 2.1, 1.9, 1.0, "Cleaning &\nPreprocessing", C["sky"])
box(ax, 5.0, 2.1, 1.9, 1.0, "Model Training\n& Evaluation", C["green"])
box(ax, 7.4, 2.1, 1.9, 1.0, "Saved Model\n(.joblib)", C["amber"])
box(ax, 9.8, 2.1, 1.9, 1.0, "Prediction\nEngine", C["blue"])
box(ax, 9.8, 0.4, 1.9, 1.0, "Student\n(User)", C["grey"])
box(ax, 9.8, 3.9, 1.9, 0.9, "Dashboard\n(Streamlit)", C["blue"])
box(ax, 5.0, 0.4, 1.9, 1.0, "metrics.json", C["amber"])
for a, b in [((2.1, 2.6), (2.6, 2.6)), ((4.5, 2.6), (5.0, 2.6)), ((6.9, 2.6), (7.4, 2.6)),
             ((9.3, 2.6), (9.8, 2.6))]:
    arrow(ax, a, b)
arrow(ax, (5.95, 2.1), (5.95, 1.4)); arrow(ax, (6.9, 0.9), (9.8, 0.9), "results shown")
arrow(ax, (10.75, 1.4), (10.75, 2.1), "student details"); arrow(ax, (10.75, 3.1), (10.75, 3.9), "score + tips")
fig.savefig(OUT / "data_flow.png", dpi=170, bbox_inches="tight"); plt.close(fig)

# 3. WORKFLOW ----------------------------------------------------------------
fig, ax = canvas(12, 5.8, "System Workflow")
steps = ["Load\ndataset", "Clean &\nvalidate", "Explore\n(EDA)", "Encode &\nscale", "Train/test\nsplit",
         "Train 3\nmodels", "Evaluate\nMAE/RMSE/R²", "Select &\nsave model"]
for i, s in enumerate(steps):
    r, c = divmod(i, 4)
    col = c if r == 0 else 3 - c          # second row runs right-to-left (snake)
    x = 0.4 + col * 2.9; y = 3.6 if r == 0 else 1.9
    box(ax, x, y, 2.2, 1.0, s, C["green"] if r == 0 else C["sky"], 10)
    if c < 3:
        if r == 0: arrow(ax, (x + 2.2, y + 0.5), (x + 2.9, y + 0.5))
        else:      arrow(ax, (x, y + 0.5), (x - 0.7, y + 0.5))
arrow(ax, (0.4 + 3 * 2.9 + 1.1, 3.6), (0.4 + 3 * 2.9 + 1.1, 2.9))
ax.text(6, 1.35, "RUNTIME (when a user opens the app)", ha="center", fontsize=9, color=C["grey"], fontweight="bold")
rt = ["Enter student\ndetails", "Validate\ninputs", "Model\npredicts", "Category +\nsuggestions", "Show result\n& charts"]
for i, s in enumerate(rt):
    x = 0.2 + i * 2.35
    box(ax, x, 0.15, 1.95, 0.85, s, C["blue"], 9)
    if i < 4: arrow(ax, (x + 1.95, 0.57), (x + 2.35, 0.57))
fig.savefig(OUT / "workflow.png", dpi=170, bbox_inches="tight"); plt.close(fig)
print("diagrams saved")
