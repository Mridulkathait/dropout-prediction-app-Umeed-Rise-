# app.py
"""
Streamlit app for AI-Based Dropout Prediction & Counselling System
Team: Umeed Rise (SIH 2025)
Author: (You can add your name)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import joblib
import os
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_fscore_support,
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Umeed Rise — Dropout Predictor", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------
# Helper functions
# ---------------------------
@st.cache_data
def load_sample_data(n=500, seed=42):
    """Generate a synthetic dataset for quick testing/demo."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "student_id": [f"SID{1000+i}" for i in range(n)],
        "attendance_pct": np.clip(rng.normal(80, 12, size=n), 20, 100),
        "avg_grade": np.clip(rng.normal(65, 15, size=n), 0, 100),
        "assignments_submitted_pct": np.clip(rng.normal(85, 10, size=n), 10, 100),
        "lms_activity_count": rng.integers(0, 50, size=n),
        "fees_pending": rng.integers(0, 3, size=n),  # 0,1,2
        "num_warns": rng.integers(0, 5, size=n),
        "wellbeing_score": np.clip(rng.normal(70, 15, size=n), 0, 100),
        "extracurricular": rng.integers(0, 2, size=n),  # 0/1
    })
    # create a synthetic target: higher risk when attendance low, grades low, warnings high
    risk_score = (
        (100 - df["attendance_pct"]) * 0.35 +
        (100 - df["avg_grade"]) * 0.35 +
        df["num_warns"] * 5 +
        df["fees_pending"] * 5 +
        (50 - (df["wellbeing_score"] - 50)) * 0.1
    )
    prob = 1 / (1 + np.exp(-(risk_score - 40) / 10))
    df["dropout"] = (rng.random(n) < prob).astype(int)
    return df

def preprocess(df, target_col="dropout"):
    """Basic preprocessing: drop ids, fill NA, scale numeric features."""
    df = df.copy()
    # keep student_id for display if present
    sid = None
    if "student_id" in df.columns:
        sid = df["student_id"]
        df = df.drop(columns=["student_id"])
    # fill numeric NAs
    for c in df.columns:
        if df[c].dtype.kind in "biufc":
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].fillna("unknown")
    if target_col in df.columns:
        y = df[target_col].astype(int)
        X = df.drop(columns=[target_col])
    else:
        y = None
        X = df
    # select numeric features only (simple). Extend with one-hot for categorical if needed.
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_num = X[numeric_cols].copy()
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_num), columns=numeric_cols)
    return X_scaled, y, scaler, numeric_cols, sid

@st.cache_data
def train_xgb(X_train, y_train, params=None, num_round=200):
    if params is None:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "verbosity": 0,
        }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(params, dtrain, num_boost_round=num_round)
    return model

def predict_xgb(model, X):
    dmat = xgb.DMatrix(X)
    preds_proba = model.predict(dmat)
    preds = (preds_proba >= 0.5).astype(int)
    return preds, preds_proba

def compute_metrics(y_true, y_pred, y_proba):
    report = classification_report(y_true, y_pred, output_dict=True)
    auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true))>1 else None
    return report, auc

def plot_risk_distribution(df):
    fig = px.histogram(df, x="risk_prob", nbins=20, title="Predicted Risk Probability Distribution")
    return fig

def st_shap_plot(explainer, X_row):
    """Render SHAP force/summary plot in Streamlit using matplotlib fallback."""
    shap_values = explainer(X_row)
    try:
        st.pyplot(shap.plots.waterfall(shap_values[0], show=False))
    except Exception:
        # fallback to bar chart of shap values absolute
        sv = np.abs(shap_values.values[0])
        features = X_row.columns.tolist()
        df = pd.DataFrame({"feature": features, "importance": sv})
        df = df.sort_values("importance", ascending=True).tail(10)
        fig, ax = plt.subplots()
        ax.barh(df["feature"], df["importance"])
        ax.set_title("Top SHAP feature importances (abs)")
        st.pyplot(fig)

# ---------------------------
# Sidebar: app controls
# ---------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/AI_%26_Cloud_icon.svg/240px-AI_%26_Cloud_icon.svg.png",
                 width=120)
st.sidebar.title("Umeed Rise — Dropout Predictor")
st.sidebar.markdown("Smart India Hackathon 2025 — Prototype")

mode = st.sidebar.radio("Mode", ["Demo (sample data)", "Upload CSV", "My Project Flow"])
st.sidebar.markdown("---")
st.sidebar.markdown("Model settings")
num_round = st.sidebar.slider("Boosting rounds (XGBoost)", 50, 500, 150, 10)
train_btn = st.sidebar.button("Train / Retrain Model")

# ---------------------------
# Main layout
# ---------------------------
st.title("Umeed Rise — AI Dropout Prediction & Counselling System")
st.markdown("**Early detection • Explainability (SHAP) • Counselling suggestions**")

# load dataset
if mode == "Demo (sample data)":
    df = load_sample_data(700)
    st.success("Loaded demo dataset (synthetic) — good for testing.")
    with st.expander("Show sample data (first 10 rows)"):
        st.dataframe(df.head(10))
elif mode == "Upload CSV":
    uploaded = st.file_uploader("Upload student dataset (CSV). Columns: student_id, attendance_pct, avg_grade, assignments_submitted_pct, lms_activity_count, fees_pending, num_warns, wellbeing_score, extracurricular, dropout(optional)", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
        with st.expander("Show sample data"):
            st.dataframe(df.head(10))
    else:
        st.info("Upload a CSV or switch to Demo mode.")
        st.stop()
else:
    # quick project flow info
    st.header("Project Flow & What to Build")
    st.markdown("""
    **This app prototype includes:**
    - Data ingestion, preprocessing, and feature engineering
    - XGBoost model training and evaluation
    - SHAP explainability for per-student insights
    - Dashboard with risk distribution & student reports
    - Placeholder hooks for alerts, chatbot, mentoring modules
    """)
    st.markdown("Use the Demo or Upload mode to test the model.")
    st.stop()

# ---------------------------
# Preprocess & prepare
# ---------------------------
target_col = "dropout"
X, y, scaler, numeric_cols, sid = preprocess(df, target_col=target_col)

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# model storage
MODEL_PATH = "xgb_model.joblib"
SCALER_PATH = "scaler.joblib"

model = None
explainer = None

if os.path.exists(MODEL_PATH) and not train_btn:
    try:
        model = joblib.load(MODEL_PATH)
        st.info("Loaded saved model from disk.")
    except Exception:
        model = None

# Train on button
if train_btn or (model is None):
    st.info("Training XGBoost model... (this may take a few seconds)")
    # train using sklearn-compatible interface for simplicity: use xgb.XGBClassifier
    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=num_round, verbosity=0)
    clf.fit(X_train, y_train)
    model = clf
    joblib.dump(model, MODEL_PATH)
    st.success("Model trained and saved.")
    # build a SHAP explainer (tree explainer for tree-based model)
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = None
else:
    # build explainer for loaded model
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = None

# ---------------------------
# Evaluate model
# ---------------------------
st.markdown("## Model Evaluation")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
report, auc = compute_metrics(y_test, y_pred, y_proba)

col1, col2, col3 = st.columns([1.5, 1, 1])
with col1:
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))
with col2:
    st.subheader("ROC-AUC")
    if auc is not None:
        st.metric(label="AUC", value=f"{auc:.3f}")
    else:
        st.write("AUC not available (single class).")
with col3:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig_cm)

# ---------------------------
# Dashboard: risk distribution & top features
# ---------------------------
st.markdown("---")
st.markdown("## Dashboard & Insights")

# compute risk probs for whole dataset
df_display = df.copy()
X_all = X
probs_all = model.predict_proba(X_all)[:, 1]
df_display["risk_prob"] = probs_all
df_display["risk_label"] = pd.cut(df_display["risk_prob"], bins=[-0.01, 0.33, 0.66, 1.0], labels=["Low", "Medium", "High"])

col1, col2 = st.columns([2, 1])
with col1:
    st.plotly_chart(plot_risk_distribution(df_display), use_container_width=True)
with col2:
    st.subheader("Risk Breakdown")
    breakdown = df_display["risk_label"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
    fig = px.pie(values=breakdown.values, names=breakdown.index, title="Risk Categories")
    st.plotly_chart(fig, use_container_width=True)

# Top features from trained model (feature importance)
st.markdown("### Top Model Features (by gain)")
try:
    importance = model.get_booster().get_score(importance_type="gain")
    importance_df = pd.DataFrame({
        "feature": list(importance.keys()),
        "gain": list(importance.values())
    }).sort_values("gain", ascending=False).head(10)
    st.bar_chart(data=importance_df.set_index("feature")["gain"])
except Exception:
    st.write("Feature importance not available for this model type.")

# ---------------------------
# Per-student report / prediction
# ---------------------------
st.markdown("---")
st.markdown("## Predict & Explain for a Student")
student_selector = None
if sid is not None:
    student_selector = st.selectbox("Select student ID", options=sid.tolist())
else:
    student_selector = st.selectbox("Select row index", options=list(df.index[:200]))

st.write("Selected:", student_selector)

# find row
if sid is not None:
    row_idx = df[df["student_id"] == student_selector].index[0]
else:
    row_idx = int(student_selector)

X_row = X.iloc[[row_idx]]
student_row_raw = df.iloc[[row_idx]]
proba = model.predict_proba(X_row)[:, 1][0]
label = "High" if proba >= 0.66 else ("Medium" if proba >= 0.33 else "Low")

c1, c2, c3 = st.columns(3)
c1.metric("Predicted Risk", f"{label}")
c2.metric("Risk Probability", f"{proba:.2f}")
c3.metric("Current Dropout (true)", f"{int(df.iloc[row_idx].get('dropout', -1))}")

st.markdown("### Why this student is at risk (explainability)")
if explainer is not None:
    with st.expander("SHAP Explanation"):
        try:
            # shap waterfall or bar
            shap_values = explainer(X_row)
            fig_shap = shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig_shap)
        except Exception:
            st_shap_plot(explainer, X_row)
else:
    st.info("SHAP explainer not available (model may not support it).")

# suggested counselling actions
st.markdown("### Suggested Counselling Actions (auto-suggest)")
actions = []
if df_display.loc[row_idx, "risk_prob"] >= 0.66:
    actions = [
        "Immediate counselor meeting (within 48h)",
        "Parent/guardian notification",
        "Assign peer mentor",
        "Create 2-week study catch-up plan",
        "Monitor attendance daily for 2 weeks"
    ]
elif df_display.loc[row_idx, "risk_prob"] >= 0.33:
    actions = [
        "Schedule counseling session",
        "Provide study resources",
        "Encourage LMS engagement",
        "Weekly check-ins"
    ]
else:
    actions = ["Continue normal monitoring", "Encourage participation"]

for a in actions:
    st.write("- " + a)

# ---------------------------
# Alerts / Notifications (placeholders)
# ---------------------------
st.markdown("---")
st.markdown("## Alerts & Notifications (Placeholders)")
st.info("In production, integrate with email/SMS services (SendGrid/Twilio) to notify teachers/parents.")
st.write("Example: If 'High' risk detected, notify counselor and parent with recommended actions.")

# sample export
if st.button("Export high-risk list to CSV"):
    hr = df_display[df_display["risk_label"] == "High"].copy()
    hr = hr[["student_id", "risk_prob", "risk_label"]]
    csv = hr.to_csv(index=False)
    st.download_button("Download CSV", data=csv, file_name="high_risk_students.csv", mime="text/csv")

# ---------------------------
# Chatbot (very simple placeholder)
# ---------------------------
st.markdown("---")
st.markdown("## Chatbot (Prototype)")
st.write("A production chatbot can integrate Dialogflow / Rasa / OpenAI. Below is a simple placeholder for demo.")

user_text = st.text_input("Student says (demo):", value="I'm stressed about exams.")
if user_text:
    # simple rule-based replies (demo)
    if "stress" in user_text.lower() or "stressed" in user_text.lower():
        st.success("Chatbot: I'm sorry you're stressed. Would you like breathing exercises, study tips, or to schedule a counseling session?")
    elif "help study" in user_text.lower() or "study" in user_text.lower():
        st.success("Chatbot: Try this study plan: 25-min focused session + 5-min break. Want resources on the topic?")
    else:
        st.success("Chatbot: I can connect you to a counselor or peer mentor. Would you like that?")

# ---------------------------
# Footer / Next steps
# ---------------------------
st.markdown("---")
st.markdown("### Next steps to production")
st.write("""
- Hook to real data sources (LMS API / college ERP / Google Classroom)
- Add authentication & RBAC (teachers, counselors, parents)
- Replace placeholder chatbot with Dialogflow/Rasa/OpenAI (with privacy compliance)
- Integrate email/SMS alerts (SendGrid/Twilio), ensure privacy & opt-in
- Deploy model behind an API (FastAPI) with autoscaling; host UI on Streamlit Cloud
- Add consent & data privacy flows (student/parent consent)
""")
