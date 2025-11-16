# app.py
"""
Umeed Rise — AI Dropout Prediction & Counselling System (Robust Prototype)
Single-file Streamlit app: auto-detects target, accepts any CSV, trains/evaluates model,
supports prediction-only mode, SHAP explainability, dashboard, export, and simple chatbot.
"""

import streamlit as st
st.set_page_config(page_title="Umeed Rise — Dropout Predictor", layout="wide", initial_sidebar_state="expanded")

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import shap
import warnings
warnings.filterwarnings("ignore")

from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb

# ---------------------------
# Helpers
# ---------------------------
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
PREPROCESS_PATH = os.path.join(MODEL_DIR, "preprocess.joblib")

@st.cache_data
def load_sample_data(n=800, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "student_id": [f"SID{1000+i}" for i in range(n)],
        "attendance_pct": np.clip(rng.normal(80, 12, size=n), 20, 100),
        "avg_grade": np.clip(rng.normal(65, 15, size=n), 0, 100),
        "assignments_submitted_pct": np.clip(rng.normal(85, 10, size=n), 10, 100),
        "lms_activity_count": rng.integers(0, 50, size=n),
        "fees_pending": rng.integers(0, 3, size=n),
        "num_warns": rng.integers(0, 5, size=n),
        "wellbeing_score": np.clip(rng.normal(70, 15, size=n), 0, 100),
        "extracurricular": rng.integers(0, 2, size=n),
    })
    # synthetic risk generation
    risk_score = (
        (100 - df["attendance_pct"]) * 0.36 +
        (100 - df["avg_grade"]) * 0.36 +
        df["num_warns"] * 5 +
        df["fees_pending"] * 4 +
        (50 - (df["wellbeing_score"] - 50)) * 0.1
    )
    prob = 1 / (1 + np.exp(-(risk_score - 40) / 10))
    rng2 = np.random.default_rng(seed+1)
    df["dropout"] = (rng2.random(n) < prob).astype(int)
    return df

def infer_target_candidates(columns):
    # Common target names to check
    candidates = ["dropout", "is_dropout", "label", "target", "risk", "y", "outcome"]
    found = [c for c in candidates if c in columns]
    return found

def build_preprocessor(df, ignore_cols):
    # Identify numeric and categorical features automatically
    features = [c for c in df.columns if c not in ignore_cols]
    numeric_feats = df[features].select_dtypes(include=[np.number]).columns.tolist()
    cat_feats = [c for c in features if c not in numeric_feats]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_feats),
        ("cat", cat_pipeline, cat_feats)
    ], remainder="drop")

    return preprocessor, numeric_feats, cat_feats

def train_model(X_train, y_train, use_xgb=True, n_estimators=150):
    # Train XGBoost where possible, fallback to RandomForest
    if use_xgb:
        try:
            clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=n_estimators, verbosity=0)
            clf.fit(X_train, y_train)
            return clf, "xgboost"
        except Exception as e:
            st.warning(f"XGBoost failed, falling back to RandomForest: {e}")
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    return clf, "rf"

def safe_classify(model, X):
    try:
        proba = model.predict_proba(X)[:, 1]
        preds = (proba >= 0.5).astype(int)
        return preds, proba
    except Exception:
        preds = model.predict(X)
        # if no proba available, create pseudo-proba
        proba = preds.copy().astype(float)
        return preds, proba

def show_classification_metrics(y_true, y_pred, y_proba):
    st.subheader("Model performance")
    st.text(classification_report(y_true, y_pred))
    if len(np.unique(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, y_proba)
            st.metric("ROC-AUC", f"{auc:.3f}")
        except Exception:
            st.write("AUC could not be computed.")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# ---------------------------
# UI: Sidebar
# ---------------------------
st.sidebar.title("Umeed Rise — Dropout Predictor")
st.sidebar.markdown("**Smart India Hackathon 2025** — Prototype")
st.sidebar.markdown("---")
mode = st.sidebar.radio("Mode", ["Demo (sample data)", "Upload CSV", "Project Info"])
st.sidebar.markdown("---")
n_estimators = st.sidebar.slider("Model Trees / Estimators", 50, 500, 150, 10)
train_btn = st.sidebar.button("Train / Retrain Model (use current dataset)")

# ---------------------------
# Main: Header
# ---------------------------
st.title("Umeed Rise — AI Dropout Prediction & Counselling System")
st.markdown("**Early detection • Explainability • Counselling Suggestions**")
st.markdown("---")

# ---------------------------
# Load data
# ---------------------------
if mode == "Demo (sample data)":
    df = load_sample_data(700)
    st.success("Loaded demo synthetic dataset.")
    with st.expander("Show demo data (first 10 rows)"):
        st.dataframe(df.head(10))
elif mode == "Upload CSV":
    uploaded = st.file_uploader("Upload a CSV file (any schema). If you have a 'dropout' column, the app will train; otherwise it will predict.", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV or switch to Demo mode.")
        st.stop()
    try:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        with st.expander("Show sample rows"):
            st.dataframe(df.head(10))
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
else:
    st.header("Project Flow & How to use this app")
    st.markdown("""
    - Use **Demo** to train quickly on synthetic data.
    - Use **Upload CSV** to either train (if dataset has labels) or predict (if labels absent).
    - The app auto-detects common target names; you can also map columns manually.
    """)
    st.stop()

# ---------------------------
# Target detection & mapping
# ---------------------------
common_targets = infer_target_candidates(df.columns.tolist())
target_col = None
st.markdown("### 1) Target Detection / Selection")
if len(common_targets) > 0:
    st.success(f"Detected possible target columns: {common_targets}")
    target_col = st.selectbox("If one of these is the actual label column, select it (otherwise choose 'None')", options=["None"] + common_targets)
else:
    st.info("No common target column automatically detected.")

# allow user to set manually
st.markdown("Or manually select a column as target (if present)")
manual_select = st.selectbox("Manual target column (or choose 'None')", options=["None"] + list(df.columns))
if manual_select != "None":
    target_col = manual_select

if target_col == "None":
    target_col = None

# ---------------------------
# Prepare training vs prediction-only
# ---------------------------
label_present = target_col is not None and target_col in df.columns
# optional student id column
id_col_candidates = [c for c in df.columns if "id" in c.lower() or "sid" in c.lower() or "student" in c.lower()]
id_col = st.selectbox("Student ID column (optional, used for per-student view)", options=["None"] + id_col_candidates)
if id_col == "None":
    id_col = None

# ---------------------------
# Preprocessing
# ---------------------------
ignore_cols = []
if id_col:
    ignore_cols.append(id_col)
if label_present:
    ignore_cols.append(target_col)

try:
    preprocessor, numeric_feats, cat_feats = build_preprocessor(df, ignore_cols)
except Exception as e:
    st.error(f"Preprocessing failed to build: {e}")
    st.stop()

st.markdown(f"Numeric features detected: `{numeric_feats}`")
st.markdown(f"Categorical features detected: `{cat_feats}`")

# ---------------------------
# Prepare data for modeling
# ---------------------------
X_all = df.drop(columns=[c for c in [id_col, target_col] if c is not None], errors="ignore")
y_all = None
if label_present:
    y_all = df[target_col].copy()
    # try convert to binary
    if y_all.dtype == object:
        # attempt mapping yes/no/true/1 etc
        y_all = y_all.str.strip().str.lower().map({
            "yes":1,"no":0,"true":1,"false":0,"1":1,"0":0,"dropout":1,"notdropout":0
        }).fillna(y_all)
    # final attempt to cast to numeric
    try:
        y_all = pd.to_numeric(y_all)
    except Exception:
        pass

# show a quick preview of X_all
with st.expander("Preview features prepared for model"):
    st.dataframe(X_all.head(8))

# ---------------------------
# Train / Load model
# ---------------------------
model = None
preprocessor_fitted = None
explainer = None
is_trained = False

# If user clicks Train OR if label_present and no saved model exists -> train
if (train_btn and label_present) or (label_present and not os.path.exists(MODEL_PATH)):
    st.info("Training model on uploaded dataset...")
    # build pipeline: preprocessor + model
    try:
        # fit preprocessor
        preprocessor_fitted = preprocessor.fit(X_all)
        X_transformed = preprocessor_fitted.transform(X_all)
        # split
        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_all, test_size=0.25, random_state=42, stratify=y_all if len(np.unique(y_all))>1 else None)
        # train model
        model, model_type = train_model(X_train, y_train, use_xgb=True, n_estimators=n_estimators)
        # save artifacts
        joblib.dump(model, MODEL_PATH)
        joblib.dump(preprocessor_fitted, PREPROCESS_PATH)
        st.success(f"Model trained and saved ({model_type}).")
        is_trained = True
        # build SHAP explainer if possible
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            explainer = None
    except Exception as e:
        st.error(f"Training failed: {e}")
        is_trained = False

# If model exists on disk, load it for prediction-only mode
if os.path.exists(MODEL_PATH) and not is_trained:
    try:
        model = joblib.load(MODEL_PATH)
        if os.path.exists(PREPROCESS_PATH):
            preprocessor_fitted = joblib.load(PREPROCESS_PATH)
        st.info("Loaded existing saved model & preprocessor.")
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            explainer = None
        is_trained = True
    except Exception as e:
        st.warning(f"Could not load saved model: {e}")

# If label not present and no model -> recommend training on demo
if not label_present and model is None:
    st.warning("No label column detected and no trained model available. Use Demo mode to train a model or upload a labelled CSV.")
    st.stop()

# ---------------------------
# Predict for entire dataset
# ---------------------------
st.markdown("---")
st.header("Dashboard & Predictions")

try:
    X_trans = preprocessor_fitted.transform(X_all)
    preds, probas = safe_classify(model, X_trans)
    df_display = X_all.copy()
    if id_col:
        df_display[id_col] = df[id_col]
    df_display["risk_prob"] = probas
    df_display["risk_label"] = pd.cut(df_display["risk_prob"], bins=[-0.01, 0.33, 0.66, 1.0], labels=["Low","Medium","High"])
except Exception as e:
    st.error(f"Prediction error: {e}")
    st.stop()

col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Risk Probability Distribution")
    fig = px.histogram(df_display, x="risk_prob", nbins=20, title="Predicted risk probability")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.subheader("Risk Breakdown")
    breakdown = df_display["risk_label"].value_counts().reindex(["Low","Medium","High"]).fillna(0)
    fig2 = px.pie(values=breakdown.values, names=breakdown.index, title="Risk categories")
    st.plotly_chart(fig2, use_container_width=True)

# If labels are present, show evaluation
if label_present:
    st.markdown("### Model Evaluation on provided labels")
    # compute metrics on original rows
    try:
        # ensure y_all is numeric binary
        y_true = pd.to_numeric(y_all).astype(int)
        st.text(classification_report(y_true, preds))
        if len(np.unique(y_true))>1:
            auc = roc_auc_score(y_true, probas)
            st.metric("ROC-AUC", f"{auc:.3f}")
        cm = confusion_matrix(y_true, preds)
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig_cm)
    except Exception as e:
        st.write("Could not compute evaluation: ", e)

# ---------------------------
# Per-student view & explainability
# ---------------------------
st.markdown("---")
st.header("Per-Student Report")

if id_col:
    student_selection = st.selectbox("Select student", options=list(df[id_col].unique()))
    idx = df[df[id_col] == student_selection].index[0]
else:
    student_selection = st.selectbox("Select row index", options=list(df.index[:200]))
    idx = int(student_selection)

st.write("Selected:", student_selection)
student_row = df.iloc[[idx]]
student_feat = X_all.iloc[[idx]]
student_trans = preprocessor_fitted.transform(student_feat)

p = float(probas[idx])
label = "High" if p >= 0.66 else ("Medium" if p >= 0.33 else "Low")
st.metric("Predicted Risk", label, f"{p:.2f}")

st.markdown("#### Reasons (SHAP explanation)")
if explainer is not None:
    try:
        shap_values = explainer(preprocessor_fitted.transform(student_feat))
        # waterfall plot
        fig_shap = shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig_shap)
    except Exception:
        # fallback bar chart of absolute shap
        try:
            sv = explainer(preprocessor_fitted.transform(student_feat)).values[0]
            cols = (numeric_feats + cat_feats)[:len(sv)]
            df_shap = pd.DataFrame({"feature": cols, "importance": np.abs(sv)})
            df_shap = df_shap.sort_values("importance", ascending=True).tail(8)
            fig, ax = plt.subplots()
            ax.barh(df_shap["feature"], df_shap["importance"])
            st.pyplot(fig)
        except Exception:
            st.info("SHAP explanation not available for this model.")
else:
    st.info("No SHAP explainer available for this model. Tree-based models usually support SHAP.")

# Suggested actions (simple rules)
st.markdown("#### Suggested Counselling Actions")
actions = []
if p >= 0.66:
    actions = [
        "Immediate counselor meeting within 48 hours",
        "Parent/guardian notification",
        "Assign peer mentor",
        "2-week catch-up study plan",
        "Daily attendance monitoring"
    ]
elif p >= 0.33:
    actions = [
        "Schedule counseling session",
        "Provide study resources/mentoring",
        "Weekly check-ins"
    ]
else:
    actions = ["Continue normal monitoring", "Encourage participation"]

for a in actions:
    st.write("- " + a)

# ---------------------------
# Alerts & Export
# ---------------------------
st.markdown("---")
st.header("Alerts & Export")
st.info("In production integrate with SendGrid/Twilio/email for notifications. The button below exports high-risk students.")

if st.button("Export High-Risk Students to CSV"):
    hr = df_display[df_display["risk_label"] == "High"].copy()
    if id_col:
        hr_export = hr[[id_col, "risk_prob", "risk_label"]]
    else:
        hr_export = hr[["risk_prob", "risk_label"]]
    csv = hr_export.to_csv(index=False)
    st.download_button("Download CSV", data=csv, file_name="high_risk_students.csv", mime="text/csv")

# ---------------------------
# Chatbot simple demo (placeholder)
# ---------------------------
st.markdown("---")
st.header("Chatbot (Demo)")
txt = st.text_input("Student says (demo):", value="I am stressed about assignments.")
if txt:
    if "stres" in txt.lower() or "sad" in txt.lower():
        st.success("Chatbot: I'm sorry. Would you like breathing exercises, study tips, or to schedule a counselor meeting?")
    elif "help" in txt.lower() and "study" in txt.lower():
        st.success("Chatbot: Try Pomodoro: 25 min focus + 5 min break. Want a study template?")
    else:
        st.success("Chatbot: I can connect you with a counselor or a peer mentor. Would you like that?")

# ---------------------------
# Next steps & deployment tips
# ---------------------------
st.markdown("---")
st.header("Next Steps & Tips")
st.write("""
1. Hook to real data sources (LMS/ERP): use APIs or periodic CSV imports.  
2. Add auth & RBAC (Streamlit-Auth, OAuth, or host behind secure portal).  
3. Replace placeholder chatbot with Dialogflow/Rasa/OpenAI (ensure privacy).  
4. Integrate email/SMS for alerts.  
5. Monitor model performance and retrain regularly.  
6. Add fairness checks (gender/SES bias) and consent workflows.
""")

st.markdown("### Done — you can now use this app to train on labelled data or predict on any new dataset.")
