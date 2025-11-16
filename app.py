# app.py
# Umeed Rise – AI-Based Student Dropout Prediction & Counseling
# Streamlit UI, robust preprocessing, XGBoost/RandomForest training, SHAP explainability, and counseling suggestions.

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.multiclass import type_of_target

from xgboost import XGBClassifier
import shap
from typing import List, Tuple, Optional, Dict, Any

# Page config
st.set_page_config(page_title="Umeed Rise – Dropout Prediction & Counseling", page_icon="🎓", layout="wide")

# Modern UI styles
APP_CSS = """
<style>
:root {
  --bg-grad-start: #f3f5ff; --bg-grad-end: #e8ecff;
  --card-bg: #ffffff; --primary: #5a67d8; --secondary: #7f9cf5; --accent: #a78bfa;
}
.stApp { background: linear-gradient(135deg, var(--bg-grad-start), var(--bg-grad-end)); }
.block-container { padding-top: 1rem; }
.umeed-card { background: var(--card-bg); border-radius: 16px; padding: 1rem 1.2rem; box-shadow: 0 10px 30px rgba(90,103,216,0.12); border: 1px solid rgba(167,139,250,0.2); transition: transform .2s ease, box-shadow .2s ease; }
.umeed-card:hover { transform: translateY(-2px); box-shadow: 0 14px 40px rgba(90,103,216,0.18); }
.umeed-header { font-size: 1.25rem; font-weight: 700; color: #2d3748; margin-bottom: .75rem; display: flex; align-items: center; gap: .5rem; }
.umeed-pill { display:inline-block; padding:.25rem .6rem; border-radius:24px; font-size:.85rem; font-weight:600; background: rgba(90,103,216,0.12); color: var(--primary); }
.umeed-small { font-size:.85rem; color:#4a5568; }
.umeed-risk-low { background: rgba(16,185,129,0.08); color:#065f46; }
.umeed-risk-med { background: rgba(245,158,11,0.10); color:#7c5014; }
.umeed-risk-high { background: rgba(239,68,68,0.10); color:#7f1d1d; }
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

# Session bootstrap
for k, v in {
    "model": None, "pipeline": None, "feature_cols": None,
    "num_cols": None, "cat_cols": None, "target_col": None,
    "id_col": None, "metrics": None, "X_test": None,
    "y_test": None, "proba_test": None, "pred_test": None,
    "shap_sample": 300
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Data loading
@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    try:
        return pd.read_csv(file, low_memory=False)
    except Exception:
        return pd.read_csv(file, low_memory=False, encoding_errors="ignore")

# Heuristics for target and ID detection
def detect_target_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["dropout","is_dropout","at_risk","risk","label","target","outcome","status","y"]
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in cols_lower:
            return cols_lower[c]
    # fallback: any binary-like column
    for c in df.columns:
        u = df[c].dropna().unique()
        if len(u) == 2:
            return c
    return None

def detect_id_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["student_id","id","studentid","roll","roll_no","roll_number","admission_no","enrollment","enrollment_id","sid"]
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in cols_lower:
            return cols_lower[c]
    likely = [c for c in df.columns if any(tok in c.lower() for tok in ["id","roll","enroll"])]
    return likely[0] if likely else None

# Positive class choice
def choose_positive_class(y: pd.Series) -> Any:
    pos_tokens = {"dropout","yes","true","1","high","risk","at_risk"}
    uniques = pd.Series(y).dropna().unique().tolist()
    for u in uniques:
        if str(u).strip().lower() in pos_tokens:
            return u
    # minority class as positive if no semantic match
    vc = pd.Series(y).value_counts()
    return vc.idxmin()

# Feature aligner to ensure new data matches training columns
def build_aligner(feature_cols: List[str]):
    from sklearn.base import BaseEstimator, TransformerMixin
    class FeatureAligner(BaseEstimator, TransformerMixin):
        def __init__(self, cols: List[str]): self.cols = cols
        def fit(self, X, y=None): return self
        def transform(self, X):
            Xc = X.copy()
            for c in self.cols:
                if c not in Xc.columns: Xc[c] = np.nan
            return Xc[self.cols]
    return FeatureAligner(feature_cols)

# Preprocessing
def preprocess(df: pd.DataFrame, target_col: str, id_col: Optional[str]):
    work = df.copy()
    if id_col and id_col in work.columns:
        work = work.drop(columns=[id_col])
    y_raw = work[target_col]
    X = work.drop(columns=[target_col])

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    for b in X.select_dtypes(include=["bool"]).columns:
        if b not in cat_cols: cat_cols.append(b)

    # label encoding to 0/1 with chosen positive class
    pos_class = choose_positive_class(y_raw)
    classes = pd.Series(y_raw).dropna().unique().tolist()
    # map pos_class -> 1, others -> 0 (binary); for multiclass, keep ordinal but positify pos_class
    if len(classes) == 2:
        le_map = {pos_class: 1}
        for c in classes:
            if c not in le_map: le_map[c] = 0
        y = y_raw.map(le_map).fillna(0).astype(int)
    else:
        ordered = [pos_class] + [c for c in classes if c != pos_class]
        le_map = {c: i for i, c in enumerate(ordered)}
        y = y_raw.map(le_map).fillna(0).astype(int)

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    ct = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)], remainder="drop", sparse_threshold=0.0)

    feature_cols = num_cols + cat_cols
    aligner = build_aligner(feature_cols)
    pipe = Pipeline([("align", aligner), ("preprocess", ct)])
    return X, y, pipe, feature_cols, num_cols, cat_cols, le_map

# Model factory
def make_model(algo: str = "XGBoost"):
    if algo == "XGBoost":
        return XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=42, n_jobs=-1, objective="binary:logistic"
        )
    return RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)

# Training with fallback
def train_model(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, algo_primary="XGBoost"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if len(pd.Series(y).unique()) > 1 else None
    )
    try:
        clf = make_model(algo_primary)
        full = Pipeline(steps=pipe.steps + [("clf", clf)])
        full.fit(X_train, y_train)
    except Exception:
        clf = make_model("RandomForest")
        full = Pipeline(steps=pipe.steps + [("clf", clf)])
        full.fit(X_train, y_train)

    proba = full.predict_proba(X_test)
    pred = full.predict(X_test)

    acc = accuracy_score(y_test, pred)
    avg = "binary" if len(np.unique(y_test))==2 else "macro"
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, average=avg, zero_division=0)
    try:
        roc = roc_auc_score(y_test, proba[:,1]) if proba.shape[1] >= 2 else np.nan
    except Exception:
        roc = np.nan

    metrics = {"Accuracy": acc, "Precision": precision, "Recall": recall, "F1": f1, "ROC-AUC": float(roc) if not np.isnan(roc) else np.nan}
    return full, clf, metrics, proba, pred, X_test, y_test

# Risk buckets + counseling text
def risk_bucket(p: float) -> str:
    if p < 0.33: return "Low"
    elif p < 0.66: return "Medium"
    return "High"

def counseling_for_risk(r: str) -> str:
    if r == "Low": return "Continue good performance; maintain consistency."
    if r == "Medium": return "Meet mentor; improve attendance and assignment submission."
    return "Schedule counseling; personalized academic support plan and mentor support."

# Metrics UI
def evaluate_model_ui(metrics: Dict[str, float], y_true: pd.Series, y_pred: np.ndarray):
    cmat = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cmat)
    fig = px.imshow(df_cm, text_auto=True, color_continuous_scale="Blues")
    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown('<div class="umeed-card"><div class="umeed-header">📊 Metrics</div>', unsafe_allow_html=True)
        for k, v in metrics.items():
            st.metric(k, f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else str(v))
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="umeed-card"><div class="umeed-header">🧮 Confusion Matrix</div>', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Feature names from ColumnTransformer
def get_feature_names_from_ct(ct: ColumnTransformer) -> List[str]:
    names = []
    for name, transformer, cols in ct.transformers_:
        if name == "num":
            names.extend(cols)
        elif name == "cat":
            ohe = transformer.named_steps["encoder"]
            ohe_names = ohe.get_feature_names_out(cols)
            names.extend(list(ohe_names))
    return names

# Cached SHAP explainer
@st.cache_resource
def get_shap_explainer(pipeline: Pipeline):
    if pipeline is None: return None, [], None
    pre = pipeline.named_steps["preprocess"]
    clf = pipeline.named_steps["clf"]
    feat_names = get_feature_names_from_ct(pre)
    try:
        # Prefer TreeExplainer for tree models for speed and stability
        explainer = shap.TreeExplainer(clf)
    except Exception:
        explainer = shap.Explainer(clf)
    return explainer, feat_names, pre

# Compute SHAP values with sample size
def compute_shap_values(pipeline: Pipeline, X_sample: pd.DataFrame, max_samples: int = 300):
    explainer, feat_names, pre = get_shap_explainer(pipeline)
    if explainer is None: return None, None, [], np.array([])
    Xt = pipeline.named_steps["align"].transform(X_sample.copy())
    Xt = pre.transform(Xt)
    n = min(len(Xt), max_samples)
    try:
        shap_values = explainer(Xt[:n])
    except Exception:
        shap_values = explainer.shap_values(Xt[:n])
    return shap_values, Xt, feat_names, Xt[:n]

# Predictions table for display/download
def predictions_table(df_src: pd.DataFrame, id_col: Optional[str], proba: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    p1 = proba[:,1] if proba.shape[1] >= 2 else proba[:,0]
    risks = [risk_bucket(x) for x in p1]
    suggest = [counseling_for_risk(r) for r in risks]
    out = pd.DataFrame({"Risk Probability": p1, "Risk Level": risks, "Suggestion": suggest})
    if id_col and id_col in df_src.columns:
        out[id_col] = df_src[id_col].values
        out = out[[id_col, "Risk Probability", "Risk Level", "Suggestion"]]
    return out.sort_values("Risk Probability", ascending=False)

# Risk coloring
def color_risk(val: str) -> str:
    if val == "High": return 'background-color: rgba(239,68,68,0.12); color: #7f1d1d'
    if val == "Medium": return 'background-color: rgba(245,158,11,0.12); color: #7c5014'
    return 'background-color: rgba(16,185,129,0.12); color: #065f46'

# Sidebar
st.sidebar.title("🎓 Umeed Rise")
page = st.sidebar.radio("Navigate", [
    "🏠 Home",
    "📤 Upload + Train",
    "📈 Dashboard",
    "🧮 Predict on New Data",
    "🧠 SHAP Explainability",
    "🤝 Counseling Suggestions"
])

# Home
if page.startswith("🏠"):
    st.markdown('<div class="umeed-card"><div class="umeed-header">🎓 Umeed Rise – AI-Based Student Dropout Prediction & Counseling</div>', unsafe_allow_html=True)
    st.write("Upload any student dataset, train robust models, understand risks via explainability, and generate actionable counseling suggestions. Stable, modern, and production-ready.")
    st.markdown('<div class="umeed-pill">Modern UI • Robust ML • Explainability • Counseling</div></div>', unsafe_allow_html=True)
    st.info("Use the sidebar to upload data, train, explore dashboards, explain predictions, and generate risk-based suggestions.")

# Upload + Train
elif page.startswith("📤"):
    st.markdown('<div class="umeed-card"><div class="umeed-header">📤 Upload Dataset & Train Model</div>', unsafe_allow_html=True)
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            df = load_data(up)
            st.dataframe(df.head(20), use_container_width=True)
            st.caption(f"Rows: {len(df)} • Columns: {len(df.columns)}")
            tgt_auto = detect_target_column(df)
            id_auto = detect_id_column(df)
            target_col = st.selectbox("Select target column (dropout/label/target)", options=["<Select>"] + list(df.columns), index=(list(df.columns).index(tgt_auto)+1) if tgt_auto in df.columns else 0)
            id_col = st.selectbox("Select student ID column (optional)", options=["<None>"] + list(df.columns), index=(list(df.columns).index(id_auto)+1) if id_auto in df.columns else 0)
            st.markdown('</div>', unsafe_allow_html=True)

            if target_col != "<Select>":
                id_use = None if id_col == "<None>" else id_col
                try:
                    X, y, pipe, feat_cols, num_cols, cat_cols, le_map = preprocess(df, target_col, id_use)
                    algo = st.radio("Model", ["XGBoost (preferred)","RandomForest (fallback)"])
                    algo_choice = "XGBoost" if "XGBoost" in algo else "RandomForest"
                    st.session_state["shap_sample"] = st.slider("SHAP sample size", min_value=100, max_value=1000, value=st.session_state.get("shap_sample", 300), step=50, help="Reduce for faster explainability on large datasets")
                    if st.button("Train Model", type="primary"):
                        with st.spinner("Training model..."):
                            full, clf, metrics, proba, pred, X_test, y_test = train_model(pipe, X, y, algo_choice)
                        st.success("Training complete")
                        st.session_state["model"] = clf
                        st.session_state["pipeline"] = full
                        st.session_state["feature_cols"] = feat_cols
                        st.session_state["num_cols"] = num_cols
                        st.session_state["cat_cols"] = cat_cols
                        st.session_state["target_col"] = target_col
                        st.session_state["id_col"] = id_use
                        st.session_state["metrics"] = metrics
                        st.session_state["X_test"] = X_test
                        st.session_state["y_test"] = y_test
                        st.session_state["proba_test"] = proba
                        st.session_state["pred_test"] = pred
                        evaluate_model_ui(metrics, y_test, pred)
                except Exception as e:
                    st.warning(f"Training failed – please check target and data types. Details: {e}")
            else:
                st.warning("No target selected. Please choose the target column.")
        except Exception as e:
            st.error(f"Failed to read dataset: {e}")

# Dashboard
elif page.startswith("📈"):
    if st.session_state["pipeline"] is None or st.session_state["proba_test"] is None:
        st.warning("Train a model first in 'Upload + Train'.")
    else:
        st.markdown('<div class="umeed-card"><div class="umeed-header">📈 Risk Dashboard</div>', unsafe_allow_html=True)
        proba = st.session_state["proba_test"]
        y_test = st.session_state["y_test"]
        X_test = st.session_state["X_test"]
        id_col = st.session_state["id_col"]
        p1 = proba[:,1] if proba.shape[1] >= 2 else proba[:,0]
        risks = pd.Series(p1).apply(risk_bucket)
        df_dash = predictions_table(X_test.assign(**({id_col: X_test.index} if id_col is None else {})), id_col, proba, st.session_state["pred_test"])

        fig_hist = px.histogram(p1, nbins=20, title="Risk Probability Distribution", labels={"value":"Probability"}, color_discrete_sequence=["#7f9cf5"])
        pie = px.pie(pd.DataFrame({"Risk": risks.value_counts().index, "Count": risks.value_counts().values}), names="Risk", values="Count", title="Risk Levels", color="Risk", color_discrete_map={"Low":"#10b981","Medium":"#f59e0b","High":"#ef4444"})
        top_risk = df_dash.head(20)

        col1, col2 = st.columns([1,1])
        with col1: st.plotly_chart(fig_hist, use_container_width=True)
        with col2: st.plotly_chart(pie, use_container_width=True)

        st.markdown('<div class="umeed-header">🏅 Top-Risk Students</div>', unsafe_allow_html=True)
        st.dataframe(top_risk.style.apply(lambda s: [color_risk(v) for v in s["Risk Level"]], axis=1), use_container_width=True)

        try:
            shap_vals, Xt_full, feat_names, Xt = compute_shap_values(st.session_state["pipeline"], X_test, st.session_state.get("shap_sample", 300))
            if shap_vals is not None and len(feat_names) > 0:
                st.markdown('<div class="umeed-header">🔥 Feature Importance (SHAP)</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(8,6))
                shap.summary_plot(shap_vals.values if hasattr(shap_vals, "values") else shap_vals, features=Xt, feature_names=feat_names, plot_type="bar", show=False)
                st.pyplot(fig, clear_figure=True, use_container_width=True)
        except Exception:
            st.info("SHAP importance not available for this model or dataset.")

# Predict on New Data
elif page.startswith("🧮"):
    if st.session_state["pipeline"] is None:
        st.warning("Train a model first in 'Upload + Train'.")
    else:
        st.markdown('<div class="umeed-card"><div class="umeed-header">🧮 Predict on New Data</div>', unsafe_allow_html=True)
        up2 = st.file_uploader("Upload new CSV (without target)", type=["csv"], key="u2")
        if up2 is not None:
            try:
                newdf = load_data(up2)
                idc = st.session_state["id_col"]
                id_series = newdf[idc] if (idc and idc in newdf.columns) else pd.Series(range(len(newdf)), name=idc if idc else "Index")
                feats = st.session_state["feature_cols"]
                for c in feats:
                    if c not in newdf.columns: newdf[c] = np.nan
                newdf = newdf[feats]
                pipe = st.session_state["pipeline"]
                proba = pipe.predict_proba(newdf)
                pred = pipe.predict(newdf)
                src = pd.DataFrame({idc if idc else "Index": id_series})
                table = predictions_table(src, idc, proba, pred)
                st.dataframe(table.style.apply(lambda s: [color_risk(v) for v in s["Risk Level"]], axis=1), use_container_width=True)
                csv = table.to_csv(index=False).encode("utf-8")
                st.download_button("Download Predictions CSV", data=csv, file_name="umeed_rise_predictions.csv", mime="text/csv")
            except Exception as e:
                st.warning(f"Prediction failed – dataset may not align with trained features. Details: {e}")

# SHAP Explainability
elif page.startswith("🧠"):
    if st.session_state["pipeline"] is None or st.session_state["X_test"] is None:
        st.warning("Train a model first in 'Upload + Train'.")
    else:
        X_test = st.session_state["X_test"]
        pipe = st.session_state["pipeline"]
        st.markdown('<div class="umeed-card"><div class="umeed-header">🧠 SHAP Explainability</div>', unsafe_allow_html=True)
        st.session_state["shap_sample"] = st.slider("SHAP sample size", min_value=100, max_value=1000, value=st.session_state.get("shap_sample", 300), step=50)
        try:
            shap_vals, Xt_full, feat_names, Xt = compute_shap_values(pipe, X_test, st.session_state["shap_sample"])
            if shap_vals is not None and len(feat_names) > 0:
                st.markdown('<div class="umeed-pill">Global Importance</div>', unsafe_allow_html=True)
                fig1, ax1 = plt.subplots(figsize=(8,6))
                shap.summary_plot(shap_vals.values if hasattr(shap_vals, "values") else shap_vals, features=Xt, feature_names=feat_names, show=False)
                st.pyplot(fig1, clear_figure=True, use_container_width=True)

                st.markdown('<div class="umeed-pill">Per-Student Explanation</div>', unsafe_allow_html=True)
                idx = st.number_input("Row index", min_value=0, max_value=max(0, len(X_test)-1), value=0, step=1)
                sv_row = shap_vals[idx]
                fig2, ax2 = plt.subplots(figsize=(8,6))
                try:
                    shap.plots.waterfall(sv_row, max_display=15, show=False)
                except Exception:
                    shap.plots.bar(sv_row, max_display=15, show=False)
                st.pyplot(fig2, clear_figure=True, use_container_width=True)
            else:
                st.info("SHAP explanations unavailable.")
        except Exception as e:
            st.warning(f"SHAP computation failed: {e}")

# Counseling Suggestions
elif page.startswith("🤝"):
    if st.session_state["pipeline"] is None or st.session_state["proba_test"] is None:
        st.warning("Train a model first in 'Upload + Train'.")
    else:
        st.markdown('<div class="umeed-card"><div class="umeed-header">🤝 Counseling Suggestions</div>', unsafe_allow_html=True)
        proba = st.session_state["proba_test"]
        X_test = st.session_state["X_test"]
        id_col = st.session_state["id_col"]
        table = predictions_table(X_test.assign(**({id_col: X_test.index} if id_col is None else {})), id_col, proba, st.session_state["pred_test"])
        st.dataframe(table.style.apply(lambda s: [color_risk(v) for v in s["Risk Level"]], axis=1), use_container_width=True)

        st.markdown('<div class="umeed-header">📋 Suggestions by Risk</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="umeed-card umeed-risk-low"><div class="umeed-header">✅ Low</div><div class="umeed-small">Continue good performance; maintain consistency.</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="umeed-card umeed-risk-med"><div class="umeed-header">⚠️ Medium</div><div class="umeed-small">Meet mentor; improve attendance and assignment submission.</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="umeed-card umeed-risk-high"><div class="umeed-header">🚨 High</div><div class="umeed-small">Schedule counseling; personalized academic support plan and mentor support.</div></div>', unsafe_allow_html=True)
