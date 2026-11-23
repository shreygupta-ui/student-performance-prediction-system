"""Student Performance Prediction System - Streamlit dashboard.

Run from the project root:   streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make 'src' importable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import EDU_LABELS, SCHOOL_LABELS, STUDYTIME_LABELS, TARGET
from src.data_loader import dataset_summary, load_clean
from src.predict import LIMITS, load_metrics, load_model, predict_score, suggestions

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓",
                   layout="wide", initial_sidebar_state="expanded")

PRIMARY, ACCENT = "#4f46e5", "#0ea5e9"

st.markdown(f"""
<style>
.block-container {{padding-top: 1.6rem; max-width: 1200px;}}
.hero {{background: linear-gradient(120deg,{PRIMARY},{ACCENT}); color:white;
        padding: 28px 32px; border-radius: 16px; margin-bottom: 20px;}}
.hero h1 {{color:white; margin:0; font-size:2rem;}}
.hero p {{margin:6px 0 0 0; opacity:.92; font-size:1.02rem;}}
.card {{background:white; border:1px solid #e2e8f0; border-radius:14px;
        padding:18px 20px; box-shadow:0 1px 3px rgba(15,23,42,.06); min-height:132px;}}
.card h4 {{margin:0 0 6px 0; color:#475569; font-size:.85rem; font-weight:600;
          text-transform:uppercase; letter-spacing:.04em;}}
.card .big {{font-size:1.55rem; white-space:nowrap; font-weight:700; color:#0f172a;}}
.card .small {{color:#64748b; font-size:.9rem;}}
.result {{border-radius:16px; padding:22px 26px; color:white; text-align:center;}}
.result .score {{font-size:3rem; font-weight:800; line-height:1.1;}}
.note {{background:#fff7ed; border-left:4px solid #f59e0b; padding:12px 16px;
        border-radius:8px; color:#7c2d12; font-size:.92rem;}}
section[data-testid="stSidebar"] {{background:#eef2ff;}}
</style>""", unsafe_allow_html=True)


@st.cache_data
def get_data():
    return load_clean()


@st.cache_data
def get_metrics():
    return load_metrics()


def card(title, value, sub=""):
    st.markdown(f'<div class="card"><h4>{title}</h4><div class="big">{value}</div>'
                f'<div class="small">{sub}</div></div>', unsafe_allow_html=True)


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


def style(fig, h=380):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=50, b=10),
                      paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(family="Segoe UI, sans-serif"))
    return fig


# ---------------------------------------------------------------- sidebar --
st.sidebar.markdown("## 🎓 Student Performance")
st.sidebar.caption("Machine Learning Minor Project")
page = st.sidebar.radio("Navigation", ["🏠 Home", "🔮 Student Prediction",
                                       "📊 Data Analysis", "🧪 Model Evaluation",
                                       "ℹ️ About Project"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption("Dataset: UCI Student Performance (Mathematics course, 395 students). "
                   "Grades use a 0–20 scale.")

try:
    load_model()                      # trains automatically if the model is missing
    df, metrics = get_data(), get_metrics()
except FileNotFoundError as err:
    st.error(f"{err}")
    st.stop()

best = metrics["best_model"]
bm = metrics["results"][best]

# ------------------------------------------------------------------- HOME --
if page == "🏠 Home":
    hero("Student Performance Prediction System",
         "Early estimation of a student's final grade using Machine Learning")
    s = dataset_summary(df)
    c1, c2, c3, c4 = st.columns(4)
    with c1: card("Students in dataset", s["rows"], "UCI Mathematics course")
    with c2: card("Average final grade", f"{s['mean_g3']:.1f} / 20", "G3 target variable")
    with c3: card("Selected model", best, "chosen by cross-validation")
    with c4: card("Test MAE", f"{bm['MAE']:.2f} marks", f"R² = {bm['R2']:.2f} on unseen data")
    st.write("")
    a, b = st.columns(2)
    with a:
        st.subheader("Project objective")
        st.write("Build a working ML application that analyses a student's study habits, "
                 "family background and earlier performance to **estimate the final "
                 "grade (0–20)**, place it in a performance category and suggest "
                 "improvement actions — early enough for a teacher or student to act.")
        st.subheader("Key features")
        st.markdown("- Interactive prediction form with validated inputs\n"
                    "- Performance category and personalised study tips\n"
                    "- Interactive data-analysis charts and correlation heatmap\n"
                    "- Honest model comparison (Linear Regression vs Random Forest vs "
                    "Gradient Boosting)\n"
                    "- Leakage-safe design: the second-period and final grades are not inputs")
    with b:
        st.subheader("Technology stack")
        st.markdown("| Purpose | Tool |\n|---|---|\n| Language | Python 3 |\n"
                    "| Data handling | Pandas, NumPy |\n| Machine Learning | Scikit-learn |\n"
                    "| Charts | Plotly |\n| Web app | Streamlit |\n"
                    "| Model saving | Joblib |\n| Data | CSV (UCI repository) |")
        st.markdown('<div class="note"><b>Go to “Student Prediction”</b> in the sidebar '
                    'to try the model.</div>', unsafe_allow_html=True)

# ------------------------------------------------------------- PREDICTION --
elif page == "🔮 Student Prediction":
    hero("Student Prediction", "Enter student details to estimate the final grade (0–20)")
    with st.form("predict_form"):
        st.markdown("##### 📚 Academic information")
        c1, c2, c3, c4 = st.columns(4)
        g1 = c1.slider("First-period grade (G1, 0–20)", 0, 20, 11,
                       help="Previous academic performance available before the final exam.")
        studytime = c2.selectbox("Weekly study time", list(STUDYTIME_LABELS), index=1,
                                 format_func=STUDYTIME_LABELS.get)
        absences = c3.number_input("Absences (days/classes)", 0, 75, 4)
        failures = c4.number_input("Past class failures", 0, 3, 0)

        st.markdown("##### 👨‍👩‍👧 Student & family")
        c1, c2, c3, c4 = st.columns(4)
        age = c1.number_input("Age", 15, 22, 17)
        sex = c2.selectbox("Sex", ["F", "M"], format_func={"F": "Female", "M": "Male"}.get)
        address = c3.selectbox("Home area", ["U", "R"],
                               format_func={"U": "Urban", "R": "Rural"}.get)
        traveltime = c4.selectbox("Travel time to school", [1, 2, 3, 4],
                                  format_func={1: "< 15 min", 2: "15–30 min",
                                               3: "30–60 min", 4: "> 1 hour"}.get)
        c1, c2, c3, c4 = st.columns(4)
        medu = c1.selectbox("Mother's education", list(EDU_LABELS), index=3,
                            format_func=EDU_LABELS.get)
        fedu = c2.selectbox("Father's education", list(EDU_LABELS), index=2,
                            format_func=EDU_LABELS.get)
        famsup = c3.selectbox("Family educational support", ["yes", "no"])
        internet = c4.selectbox("Internet access at home", ["yes", "no"])

        st.markdown("##### 🏫 School & lifestyle")
        c1, c2, c3, c4 = st.columns(4)
        school = c1.selectbox("School", list(SCHOOL_LABELS), format_func=SCHOOL_LABELS.get)
        schoolsup = c2.selectbox("Extra school support", ["no", "yes"])
        paid = c3.selectbox("Paid extra classes", ["no", "yes"])
        higher = c4.selectbox("Wants higher education", ["yes", "no"])
        c1, c2, c3, c4 = st.columns(4)
        activities = c1.selectbox("Extra-curricular activities", ["yes", "no"])
        romantic = c2.selectbox("In a relationship", ["no", "yes"])
        famrel = c3.slider("Family relationship quality (1–5)", 1, 5, 4)
        health = c4.slider("Health status (1–5)", 1, 5, 3)
        c1, c2, c3, c4 = st.columns(4)
        freetime = c1.slider("Free time after school (1–5)", 1, 5, 3)
        goout = c2.slider("Going out with friends (1–5)", 1, 5, 3)
        dalc = c3.slider("Weekday alcohol use (1–5)", 1, 5, 1)
        walc = c4.slider("Weekend alcohol use (1–5)", 1, 5, 2)
        submitted = st.form_submit_button("🔮 Predict Final Score", type="primary",
                                   width="stretch")

    if submitted:
        inputs = dict(age=age, studytime=studytime, failures=failures, absences=absences,
                      G1=g1, Medu=medu, Fedu=fedu, famrel=famrel, freetime=freetime,
                      goout=goout, health=health, Dalc=dalc, Walc=walc,
                      traveltime=traveltime, school=school, sex=sex, address=address,
                      famsup=famsup, schoolsup=schoolsup, paid=paid, internet=internet,
                      higher=higher, activities=activities, romantic=romantic)
        try:
            res = predict_score(inputs)
        except ValueError as err:
            st.error(f"Invalid input: {err}")
            st.stop()

        left, right = st.columns([1, 1.3])
        with left:
            st.markdown(f'<div class="result" style="background:{res["colour"]}">'
                        f'<div>Predicted final score</div>'
                        f'<div class="score">{res["score"]:.1f} / 20</div>'
                        f'<div style="font-size:1.2rem;font-weight:600">{res["category"]}'
                        f'</div></div>', unsafe_allow_html=True)
            st.write("")
            st.markdown(f'<div class="note"><b>This is an estimate, not a guaranteed '
                        f'result.</b> On unseen test data the model is typically off by '
                        f'about ±{bm["MAE"]:.1f} marks, so the likely range is '
                        f'<b>{res["low"]}–{res["high"]}</b>.</div>', unsafe_allow_html=True)
        with right:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=res["score"], number={"suffix": " / 20"},
                gauge={"axis": {"range": [0, 20]}, "bar": {"color": res["colour"]},
                       "steps": [{"range": [0, 10], "color": "#fee2e2"},
                                 {"range": [10, 13], "color": "#fef3c7"},
                                 {"range": [13, 16], "color": "#dbeafe"},
                                 {"range": [16, 20], "color": "#dcfce7"}]}))
            st.plotly_chart(style(fig, 280), width="stretch")

        st.subheader("💡 Personalised study suggestions")
        for t in suggestions(inputs, res["score"]):
            st.markdown(f"- {t}")
        st.caption("Suggestions are simple rule-based study tips; the numeric estimate "
                   "comes from the trained machine learning model.")

# ---------------------------------------------------------- DATA ANALYSIS --
elif page == "📊 Data Analysis":
    hero("Data Analysis", "Exploring the UCI Student Performance dataset")
    s = dataset_summary(df)
    c1, c2, c3, c4 = st.columns(4)
    with c1: card("Records", s["rows"])
    with c2: card("Columns", s["columns"])
    with c3: card("Missing values", s["missing"])
    with c4: card("Pass rate (G3 ≥ 10)", f"{s['pass_rate']:.1f}%")
    st.write("")
    with st.expander("View dataset sample"):
        st.dataframe(df.head(25), width="stretch")
    t1, t2, t3, t4, t5 = st.tabs(["Grade distribution", "Study time", "Absences",
                                  "Correlation", "Other factors"])
    with t1:
        fig = px.histogram(df, x=TARGET, nbins=21, color_discrete_sequence=[PRIMARY],
                           title="Distribution of final grade (G3)")
        st.plotly_chart(style(fig), width="stretch")
        st.caption(f"{s['zero_g3']} students have G3 = 0. They are shown here because they are "
                   "real records, but they are excluded from model training (see Model Evaluation).")
        melt = df[["G1", "G2", "G3"]].melt(var_name="Period", value_name="Grade")
        st.plotly_chart(style(px.box(melt, x="Period", y="Grade", color="Period",
                        title="Grades across the three periods")), width="stretch")
    with t2:
        d = df.assign(Study=df["studytime"].map(STUDYTIME_LABELS))
        avg = d.groupby("Study")[TARGET].mean().reindex(STUDYTIME_LABELS.values()).reset_index()
        st.plotly_chart(style(px.bar(avg, x="Study", y=TARGET, text_auto=".2f",
                        color_discrete_sequence=[ACCENT],
                        title="Average final grade by weekly study time")),
                        width="stretch")
        st.plotly_chart(style(px.box(d, x="Study", y=TARGET, category_orders={
                        "Study": list(STUDYTIME_LABELS.values())},
                        title="Grade spread by study time")), width="stretch")
    with t3:
        st.plotly_chart(style(px.scatter(df, x="absences", y=TARGET, opacity=.6,
                        trendline="ols", color_discrete_sequence=[PRIMARY],
                        title="Absences vs final grade")), width="stretch")
        bins = pd.cut(df["absences"], [-1, 0, 4, 9, 20, 100],
                      labels=["0", "1–4", "5–9", "10–20", "20+"])
        g = df.groupby(bins, observed=True)[TARGET].mean().reset_index()
        st.plotly_chart(style(px.bar(g, x="absences", y=TARGET, text_auto=".2f",
                        title="Average grade by absence group")), width="stretch")
    with t4:
        num = df.select_dtypes("number")
        corr = num.corr().round(2)
        st.plotly_chart(style(px.imshow(corr, text_auto=True, aspect="auto",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Correlation heatmap (numeric columns)"), 640),
                        width="stretch")
        st.caption("G1 and G2 are strongly correlated with G3. This project uses G1 "
                   "but excludes G2 so that predictions remain useful early in the term.")
    with t5:
        c1, c2 = st.columns(2)
        f = df.groupby("failures")[TARGET].mean().reset_index()
        c1.plotly_chart(style(px.bar(f, x="failures", y=TARGET, text_auto=".2f",
                        title="Average grade by past failures")), width="stretch")
        i = df.groupby("internet")[TARGET].mean().reset_index()
        c2.plotly_chart(style(px.bar(i, x="internet", y=TARGET, text_auto=".2f",
                        title="Average grade by internet access")), width="stretch")
        c1, c2 = st.columns(2)
        h = df.groupby("higher")[TARGET].mean().reset_index()
        c1.plotly_chart(style(px.bar(h, x="higher", y=TARGET, text_auto=".2f",
                        title="Wants higher education")), width="stretch")
        m = df.groupby("Medu")[TARGET].mean().reset_index()
        c2.plotly_chart(style(px.bar(m, x="Medu", y=TARGET, text_auto=".2f",
                        title="Average grade by mother's education")), width="stretch")

# ------------------------------------------------------ MODEL EVALUATION --
elif page == "🧪 Model Evaluation":
    hero("Model Evaluation", "Actual results measured on unseen test data")
    st.write(f"Data was split **{int((1-metrics['test_size'])*100)}% training "
             f"({metrics['train_rows']} rows) / {int(metrics['test_size']*100)}% testing "
             f"({metrics['test_rows']} rows)** with random_state={metrics['random_state']}. "
             f"Selection rule: *{metrics['selection_rule']}*.")
    c1, c2, c3, c4 = st.columns(4)
    with c1: card("Selected model", best, "lowest CV RMSE")
    with c2: card("MAE", f"{bm['MAE']:.3f}", "average error in marks")
    with c3: card("RMSE", f"{bm['RMSE']:.3f}", "penalises big errors")
    with c4: card("R²", f"{bm['R2']:.3f}", "variance explained")
    st.write("")
    rows = [{"Model": k + (" ★ selected" if k == best else ""), "MAE": v["MAE"], "RMSE": v["RMSE"], "R²": v["R2"],
             "CV RMSE (train)": v["CV_RMSE_mean"]} for k, v in metrics["results"].items()]
    tbl = pd.DataFrame(rows)
    st.subheader("Algorithm comparison")
    st.dataframe(tbl.style.format({c: "{:.3f}" for c in tbl.columns[1:]})
                 .highlight_min(subset=["MAE", "RMSE"], color="#dcfce7")
                 .highlight_max(subset=["R²"], color="#dcfce7"),
                 width="stretch", hide_index=True)
    st.caption("Green = lowest error on the test set. ★ = the model actually selected. It was "
               "chosen using cross-validation on the TRAINING data (last column), not the test "
               "set, so the test set stays a fair, untouched final exam. The three real models "
               "are within about 0.15 marks of each other on only "
               f"{metrics['test_rows']} test students, so the gap is small and not conclusive.")
    st.plotly_chart(style(px.bar(tbl.melt(id_vars="Model", value_vars=["MAE", "RMSE"]),
                    x="Model", y="value", color="variable", barmode="group",
                    title="Error comparison (lower is better)")), width="stretch")
    c1, c2 = st.columns(2)
    act, pred = metrics["test_actual"], metrics["test_predicted"]
    sc = px.scatter(x=act, y=pred, labels={"x": "Actual G3", "y": "Predicted G3"},
                    title="Actual vs predicted (test set)", opacity=.7)
    sc.add_shape(type="line", x0=0, y0=0, x1=20, y1=20, line=dict(dash="dash", color="gray"))
    c1.plotly_chart(style(sc), width="stretch")
    imp = metrics["feature_importance_rf"]
    top = pd.DataFrame(list(imp.items())[:10], columns=["Feature", "Importance"])
    c2.plotly_chart(style(px.bar(top[::-1], x="Importance", y="Feature", orientation="h",
                    title="Top features (Random Forest importance)")), width="stretch")
    co = pd.DataFrame(list(metrics["lr_coefficients"].items())[:10], columns=["Feature", "Coefficient"])
    st.plotly_chart(style(px.bar(co[::-1], x="Coefficient", y="Feature", orientation="h",
                    title="Linear Regression: standardised coefficients (+ raises, − lowers the grade)")),
                    width="stretch")
    st.subheader("How much does the first-period grade (G1) matter?")
    ab = pd.DataFrame([{"Model": k, "MAE without G1": v["MAE"], "R² without G1": v["R2"]}
                       for k, v in metrics["ablation_without_G1"].items()])
    st.dataframe(ab.style.format({"MAE without G1": "{:.3f}", "R² without G1": "{:.3f}"}),
                 width="stretch", hide_index=True)
    st.caption("Without any earlier grade the model is clearly less accurate. This is "
               "reported honestly so the limits of the system are clear.")
    st.subheader("Why this model?")
    why = {"Linear Regression": "It had the lowest cross-validated RMSE. The grade depends "
           "mostly on the earlier grade in a nearly linear way, so a simple, interpretable "
           "model is enough, and it avoids overfitting on a small dataset.",
           "Random Forest": "It had the lowest cross-validated RMSE. Averaging many decision "
           "trees captures non-linear patterns.",
           "Gradient Boosting": "It had the lowest cross-validated RMSE. It builds trees one "
           "after another, each correcting the previous errors."}[best]
    st.write(f"**{best}** was selected. {why} Differences between the three models are "
             "small on only "
             f"{metrics['test_rows']} test students, so they should not be over-interpreted. "
             "All figures come from `models/metrics.json`, produced by `src/train_model.py`.")
    z = metrics["sensitivity_with_zero_grades"]
    st.info(f"**Data decision:** {metrics['zero_grade_rows_excluded']} students with a final "
            "grade of 0 were excluded from training because every one of them has 0 recorded "
            "absences (they most likely left the course). Keeping them made the model "
            "wrongly predict failure for students with perfect attendance. For transparency, "
            f"a Random Forest trained on all {z['rows']} rows scored MAE {z['MAE']:.2f}, "
            f"RMSE {z['RMSE']:.2f}, R² {z['R2']:.2f} on its own (different) test split, so "
            "the two sets of numbers are not directly comparable.")

# ------------------------------------------------------------------ ABOUT --
else:
    hero("About the Project", "Problem, solution, method, limitations and references")
    st.subheader("Problem statement")
    st.write("Teachers and students usually learn about poor academic performance only "
             "after final results. There is a need for a simple tool that gives an "
             "early, data-driven estimate so that timely support can be provided.")
    st.subheader("Proposed solution")
    st.write("A Machine Learning regression model trained on the UCI Student "
             "Performance dataset predicts the final grade (G3) from academic, family "
             "and lifestyle attributes, delivered through a Streamlit dashboard.")
    st.subheader("Methodology")
    st.markdown("1. Load and clean data  2. Exploratory data analysis  3. Encode "
                "categorical features, scale numeric features  4. Train/test split  "
                "5. Train Linear Regression, Random Forest, Gradient Boosting  "
                "6. Evaluate with MAE, RMSE, R²  7. Save best model with Joblib  "
                "8. Serve through Streamlit")
    st.subheader("Limitations")
    st.markdown("- Small dataset (395 students) from two Portuguese schools; results may "
                "not transfer to other countries or institutions.\n"
                "- Predictions are estimates and depend strongly on the first-period grade.\n"
                "- 38 students with a final grade of 0 (who all have 0 recorded absences, "
                "probably course leavers) are excluded from training; the model does not "
                "predict dropout.\n"
                "- The model finds correlations, not causes; it must not be used to make "
                "decisions about a student on its own.\n"
                "- Self-reported survey attributes can be inaccurate.")
    st.subheader("Future scope")
    st.markdown("Add the Portuguese-language course data, use SHAP explanations, add "
                "teacher login and student history, and retrain on local college data.")
    st.subheader("References")
    st.markdown("- P. Cortez and A. Silva, *Using Data Mining To Predict Secondary School "
                "Student Performance*, in A. Brito and J. Teixeira (Eds.), Proceedings of "
                "5th FUture BUsiness TEChnology Conference (FUBUTEC 2008), pp. 5–12, "
                "Porto, Portugal, April 2008.\n"
                "- UCI Machine Learning Repository: Student Performance dataset, "
                "https://archive.ics.uci.edu/dataset/320/student+performance "
                "(DOI 10.24432/C5TG7T).\n"
                "- Scikit-learn documentation: https://scikit-learn.org\n"
                "- Streamlit documentation: https://docs.streamlit.io")
