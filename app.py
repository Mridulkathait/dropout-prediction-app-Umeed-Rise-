# UmeedRise – AI-Driven Student Dropout Prediction & Counseling System
# Streamlit app: production-ready, robust ML pipeline, SHAP explanations, Plotly dashboard, premium UI
# Author: GitHub Copilot
# Compatible with Streamlit Cloud

import os
import io
import sys
import json
import time
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.multiclass import type_of_target

# Try to import xgboost
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# Try to import shap
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

warnings.filterwarnings("ignore")


# ============================
# Global UI Styles and Helpers
# ============================

APP_NAME = "UmeedRise – AI-Driven Student Dropout Prediction & Counseling System"

PRIMARY_GRADIENT = """
background: linear-gradient(135deg, #5f6fff 0%, #9a89ff 50%, #c7a8ff 100%);
"""

CARD_STYLE = """
background-color: #ffffff;
border-radius: 16px;
box-shadow: 0 8px 24px rgba(0,0,0,0.08);
padding: 18px 20px;
border: 1px solid rgba(136, 136, 136, 0.15);
"""

LABEL_CHIP_STYLE = """
display: inline-block;
padding: 4px 10px;
border-radius: 999px;
background: #f1f0ff;
color: #4c4cff;
font-weight: 600;
border: 1px solid rgba(120,120,255,0.3);
"""

RISK_COLORS = {
    "Low": "#34c759",      # green
    "Medium": "#ffcc00",   # yellow
    "High": "#ff3b30",     # red
}

def inject_global_css():
    st.markdown(
        f"""
        <style>
        /* Page styling */
        .stApp {{
            {PRIMARY_GRADIENT}
            min-height: 100vh;
            background-attachment: fixed;
        }}
        /* Sidebar styling */
        section[data-testid="stSidebar"] > div {{
            {PRIMARY_GRADIENT}
            color: white !important;
        }}
        .sidebar-title {{
            font-weight: 700;
            font-size: 20px;
            margin-bottom: 8px;
            color: white;
        }}
        .sidebar-sub {{
            opacity: 0.9;
        }}
        /* Header */
        .app-header {{
            color: white;
            margin-bottom: 10px;
        }}
        .app-title {{
            font-size: 32px;
            font-weight: 800;
            letter-spacing: 0.2px;
        }}
        .app-subtitle {{
            font-size: 16px;
            opacity: 0.92;
            font-weight: 500;
        }}
        /* Card */
        .card {{
            {CARD_STYLE}
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 12px 28px rgba(0,0,0,0.12);
        }}
        .chip {{
            {LABEL_CHIP_STYLE}
            margin-right: 8px;
        }}
        .metric-good {{
            color: #34c759;
            font-weight: 700;
        }}
        .metric-mid {{
            color: #ffcc00;
            font-weight: 700;
        }}
        .metric-bad {{
            color: #ff3b30;
            font-weight: 700;
        }}
        .risk-low {{ color: #34c759; font-weight: 700; }}
        .risk-medium {{ color: #ffcc00; font-weight: 700; }}
        .risk-high {{ color: #ff3b30; font-weight: 700; }}
        .caption {{
            font-size: 13px; 
            opacity: 0.8;
        }}
        .small-note {{
            font-size: 12px; 
            opacity: 0.7;
        }}
        .icon {{
            margin-right: 6px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# ============================
# Core Modular Functions
# ============================

def load_data(file: io.BytesIO) -> Optional[pd.DataFrame]:
    """
    Load CSV file into pandas DataFrame with robust parsing.
    Returns None if loading fails.
    """
    if file is None:
        return None
    try:
        # Try UTF-8, then fallback to latin-1
        for enc in ["utf-8", "utf-8-sig", "latin-1"]:
            try:
                df = pd.read_csv(file, encoding=enc)
                return df
            except Exception:
                file.seek(0)
        # Final fallback: let pandas guess
        df = pd.read_csv(file, engine="python", error_bad_lines=False)
        return df
    except Exception as e:
        st.warning(f"Could not read CSV. Please ensure the file is a valid CSV. Details: {e}")
        return None


def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Auto-detect target, student ID, numeric and categorical features.
    Returns a dictionary of detected column names and lists.
    """
    if df is None or df.empty:
        return {
            "target": None,
            "student_id": None,
            "numeric": [],
            "categorical": []
        }

    # Normalize column names for heuristics
    cols_lower = [str(c).lower() for c in df.columns]

    # Target detection heuristics
    target_candidates = [
        "dropout", "label", "target", "risk", "outcome", "y", "class", "status"
    ]
    target = None
    for tc in target_candidates:
        for i, c in enumerate(cols_lower):
            if tc == c or tc in c:
                target = df.columns[i]
                break
        if target:
            break

    # Student ID detection heuristics
    id_candidates = [
        "student_id", "id", "student", "roll", "rollno", "roll_no", "admission", "enrollment"
    ]
    student_id = None
    for ic in id_candidates:
        for i, c in enumerate(cols_lower):
            if ic == c or ic in c:
                student_id = df.columns[i]
                break
        if student_id:
            break

    # Identify numeric & categorical feature columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Exclude detected target and student_id from feature lists
    for special in [target, student_id]:
        if special in numeric_cols:
            numeric_cols.remove(special)
        if special in categorical_cols:
            categorical_cols.remove(special)

    # If no target was found, attempt to infer binary column
    if target is None:
        # Look for a column with two unique values plausibly dropout-related
        for col in df.columns:
            unique_vals = pd.Series(df[col]).dropna().unique()
            if len(unique_vals) == 2 and str(col).lower() not in [str(student_id).lower() if student_id else ""]:
                # Heuristic: choose the first binary column
                target = col
                break

    return {
        "target": target,
        "student_id": student_id,
        "numeric": numeric_cols,
        "categorical": categorical_cols
    }


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """
    Build preprocessing pipeline:
    - Numeric: median impute + StandardScaler
    - Categorical: constant impute + OneHotEncoder (sparse_output=False, handle_unknown='ignore')
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder="drop"
    )
    return preprocessor


def train_model(
    df: pd.DataFrame,
    target: str,
    student_id: Optional[str],
    numeric_cols: List[str],
    categorical_cols: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Pipeline, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Train the ML model using XGBoostClassifier primarily with RandomForest fallback.
    Returns:
        - Full pipeline (preprocessor + classifier)
        - X_train_processed, X_test_processed (after preprocessor)
        - y_train, y_test
        - feature_names (after transformation)
    Ensures stratified split if possible.
    """
    # Safety checks
    if target is None or target not in df.columns:
        raise ValueError("Target column missing or not found.")

    # Prepare X, y
    feature_cols = numeric_cols + categorical_cols
    # If neither numeric nor categorical exist, fallback to all except target and id
    if len(feature_cols) == 0:
        feature_cols = [c for c in df.columns if c != target and c != student_id]

    X = df[feature_cols].copy()
    y = df[target].copy()

    # Standardize labels to integer encoding if necessary
    y_raw = y.copy()
    y_unique = pd.Series(y_raw).dropna().unique()

    # Determine target type and possibly encode
    y_type = type_of_target(y_raw)
    if y_type in ["binary", "multiclass"]:
        # If y is not numeric, factorize
        if not np.issubdtype(y_raw.dtype, np.number):
            y_encoded, y_categories = pd.factorize(y_raw)
            y = y_encoded
        else:
            y = y_raw.astype(int)
    elif y_type == "continuous":
        # Convert to binary by threshold median (best-effort fallback)
        st.warning("Detected continuous target; converting to binary with median threshold for classification.")
        thresh = pd.to_numeric(y_raw, errors="coerce").median()
        y = (pd.to_numeric(y_raw, errors="coerce") >= thresh).astype(int)
    else:
        # Unknown type fallback
        y = pd.factorize(y_raw)[0]

    # Split with stratification if possible
    stratify_arg = None
    try:
        if len(np.unique(y)) > 1:
            stratify_arg = y
    except Exception:
        stratify_arg = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg
    )

    # Build preprocessor
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # Choose model
    if XGB_AVAILABLE:
        classifier = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic" if len(np.unique(y)) == 2 else "multi:softprob",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1
        )
    else:
        classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1
        )

    # Full pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clf", classifier)
    ])

    # Fit
    model.fit(X_train, y_train)

    # Extract transformed feature names
    feature_names = []
    try:
        # Fit a clone of preprocessor to get names
        preprocessor.fit(X_train)
        num_features = numeric_cols
        cat_features = []
        # Get OHE feature names
        try:
            ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
            cat_features = ohe.get_feature_names_out(categorical_cols).tolist()
        except Exception:
            cat_features = categorical_cols  # Fallback
        feature_names = num_features + cat_features
    except Exception:
        feature_names = [f"f_{i}" for i in range(len(model.named_steps["preprocessor"].transform(X_train).shape[1]))]

    # Transform X for SHAP & importance use
    X_train_processed = model.named_steps["preprocessor"].transform(X_train)
    X_test_processed = model.named_steps["preprocessor"].transform(X_test)

    return model, X_train_processed, X_test_processed, y_train, y_test, feature_names


def evaluate(
    model: Pipeline,
    X_test_processed: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Compute evaluation metrics safely for binary or multiclass.
    Returns a dict of metrics.
    """
    # Predictions
    y_pred = model.named_steps["clf"].predict(X_test_processed)
    # Probabilities for ROC-AUC
    y_proba = None
    try:
        y_proba = model.named_steps["clf"].predict_proba(X_test_processed)
    except Exception:
        y_proba = None

    # Metrics
    metrics = {}
    metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
    average_strategy = "weighted" if len(np.unique(y_test)) > 2 else "binary"
    metrics["precision"] = float(precision_score(y_test, y_pred, average=average_strategy, zero_division=0))
    metrics["recall"] = float(recall_score(y_test, y_pred, average=average_strategy, zero_division=0))
    metrics["f1"] = float(f1_score(y_test, y_pred, average=average_strategy, zero_division=0))

    # ROC-AUC
    try:
        if y_proba is not None:
            if len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
            metrics["roc_auc"] = float(auc)
        else:
            metrics["roc_auc"] = np.nan
    except Exception:
        metrics["roc_auc"] = np.nan

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def risk_from_proba(p: float, low_thr: float = 0.33, high_thr: float = 0.66) -> str:
    """
    Map probability to risk level: Low, Medium, High
    """
    if p < low_thr:
        return "Low"
    elif p < high_thr:
        return "Medium"
    else:
        return "High"


def predict(
    model: Pipeline,
    df_input: pd.DataFrame,
    feature_cols: List[str],
    low_thr: float = 0.33,
    high_thr: float = 0.66
) -> pd.DataFrame:
    """
    Predict dropout risk for new data.
    Returns a DataFrame with probabilities, predicted class, and risk band.
    """
    # Align columns; missing feature columns filled with NaN
    for col in feature_cols:
        if col not in df_input.columns:
            df_input[col] = np.nan

    X_new = df_input[feature_cols].copy()

    # Predict probabilities with safe fallback
    try:
        proba = model.predict_proba(X_new)
        # Binary: use proba of class 1
        if proba.shape[1] == 2:
            risk_p = proba[:, 1]
        else:
            # For multiclass, use max probability as risk proxy
            risk_p = proba.max(axis=1)
    except Exception:
        # If predict_proba is not available
        preds = model.predict(X_new)
        # Convert to 0/1 floats
        risk_p = (preds == np.max(preds)).astype(float)

    # Compute risk bands
    risks = [risk_from_proba(p, low_thr, high_thr) for p in risk_p]

    # Predicted class
    try:
        y_pred = model.predict(X_new)
    except Exception:
        y_pred = (np.array(risk_p) >= high_thr).astype(int)

    result = df_input.copy()
    result["predicted_class"] = y_pred
    result["dropout_risk_score"] = np.round(risk_p, 4)
    result["dropout_risk_band"] = risks

    return result


def explain_with_shap(
    model: Pipeline,
    X_train_processed: np.ndarray,
    feature_names: List[str]
):
    """
    Build SHAP explainer and return explainer, shap_values/explanations for summary and per-instance.
    Works best for tree models. Falls back gracefully.
    """
    if not SHAP_AVAILABLE:
        st.warning("SHAP is not installed or not available. Please include shap in requirements.")
        return None, None

    clf = model.named_steps["clf"]
    explainer = None
    shap_values = None

    try:
        # Use TreeExplainer for tree-based models
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_train_processed)
    except Exception:
        try:
            explainer = shap.Explainer(clf.predict, model.named_steps["preprocessor"].transform)
            shap_values = explainer(X_train_processed)
        except Exception as e:
            st.warning(f"Could not initialize SHAP explainer. Details: {e}")
            return None, None

    return explainer, shap_values


# ============================
# Counseling Recommendation
# ============================

def counseling_recommendations(risk_band: str) -> List[str]:
    """
    Generate recommendations based on risk band.
    """
    if risk_band == "Low":
        return [
            "Maintain consistent study habits and weekly review sessions.",
            "Strengthen time-management through simple routines and calendar planning.",
            "Take part in peer learning groups to stay engaged."
        ]
    elif risk_band == "Medium":
        return [
            "Connect with a mentor for guidance on academic progress.",
            "Improve attendance and set reminders for classes and deadlines.",
            "Track assignments with a checklist and weekly progress reviews."
        ]
    elif risk_band == "High":
        return [
            "Schedule an immediate meeting with the counselor to discuss challenges.",
            "Create a personalized academic plan focusing on high-impact courses/topics.",
            "Implement targeted intervention: tutoring, resource access, and regular follow-up."
        ]
    else:
        return ["No recommendations available."]


# ============================
# Streamlit App Structure
# ============================

def sidebar_navigation() -> str:
    st.sidebar.markdown('<div class="sidebar-title">UmeedRise</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-sub">AI-Driven Dropout Prediction</div>', unsafe_allow_html=True)
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigate",
        [
            "🏠 Home",
            "📤 Upload + Train Model",
            "📊 Dashboard",
            "🧮 Predict New Students",
            "🔍 SHAP Explainability",
            "🧭 Counseling Plan",
        ],
        index=0
    )

    # Risk threshold controls
    st.sidebar.markdown("### ⚙️ Risk thresholds")
    low_thr = st.sidebar.slider("Low threshold", 0.0, 0.5, 0.33, 0.01)
    high_thr = st.sidebar.slider("High threshold", 0.5, 1.0, 0.66, 0.01)

    st.session_state["low_thr"] = low_thr
    st.session_state["high_thr"] = high_thr

    st.sidebar.markdown("---")
    st.sidebar.caption("Tip: Ensure your target column is detected or select it manually after upload.")
    return page


def home():
    st.markdown('<div class="app-header">', unsafe_allow_html=True)
    st.markdown(f'<div class="app-title">{APP_NAME}</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Predict student dropout risk, explain drivers, and generate actionable counseling plans — all in one modern dashboard.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Overview:** UmeedRise uses robust preprocessing, XGBoost/RandomForest models, and SHAP explainability to assess dropout risk. It supports any CSV dataset with automatic detection of columns and safe fallbacks.")
        st.markdown('<span class="chip">Auto-detection</span><span class="chip">XAI</span><span class="chip">Plotly Dashboard</span><span class="chip">Counseling</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Workflow:**")
        st.markdown("- **Upload** dataset and confirm detected target.")
        st.markdown("- **Train** model and review metrics.")
        st.markdown("- **Dashboard** with risk distribution and feature importance.")
        st.markdown("- **Predict** for new students and download results.")
        st.markdown("- **Explain** with SHAP summary and per-student insights.")
        st.markdown("- **Counsel** tailored recommendations.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Quality & Safety:**")
    st.markdown("- Robust error handling and safe defaults.")
    st.markdown("- Clean UI with premium gradient theme and rounded cards.")
    st.markdown("- No deprecated scikit-learn syntax; compatible with Streamlit Cloud.")
    st.markdown('</div>', unsafe_allow_html=True)


def upload_and_train():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload dataset")
    file = st.file_uploader("Upload CSV", type=["csv"])
    st.markdown('</div>', unsafe_allow_html=True)

    if file is not None:
        df = load_data(file)
        if df is None or df.empty:
            st.error("Failed to load data or dataset is empty. Please upload a valid CSV.")
            return

        st.session_state["df"] = df

        # Preview
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Dataset preview**")
        st.dataframe(df.head(50), use_container_width=True)
        st.markdown("**Shape:** " + f"{df.shape[0]} rows × {df.shape[1]} columns")
        st.markdown('</div>', unsafe_allow_html=True)

        # Detect columns
        detected = detect_columns(df)
        target = detected["target"]
        student_id = detected["student_id"]
        numeric_cols = detected["numeric"]
        categorical_cols = detected["categorical"]

        # Manual override UI
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🔎 Column detection")
        col1, col2 = st.columns(2)
        with col1:
            target = st.selectbox("Target column", options=[None] + list(df.columns), index=(list(df.columns).index(target) + 1) if target in df.columns else 0)
        with col2:
            student_id = st.selectbox("Student ID column (optional)", options=[None] + list(df.columns), index=(list(df.columns).index(student_id) + 1) if student_id in df.columns else 0)

        # Feature selection (auto prefill)
        st.markdown("#### Feature selection")
        all_features = [c for c in df.columns if c != target and c != student_id]
        col3, col4 = st.columns(2)
        with col3:
            numeric_cols = st.multiselect("Numeric features", options=all_features, default=[c for c in numeric_cols if c in all_features])
        with col4:
            categorical_cols = st.multiselect("Categorical features", options=all_features, default=[c for c in categorical_cols if c in all_features])

        st.markdown('</div>', unsafe_allow_html=True)

        # Missing values summary
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🧹 Missing values summary")
        miss = df.isna().sum()
        miss_df = pd.DataFrame({"column": miss.index, "missing_count": miss.values})
        miss_df["missing_percent"] = (miss_df["missing_count"] / len(df) * 100).round(2)
        st.dataframe(miss_df.sort_values("missing_count", ascending=False), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Train model
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🚀 Train model")
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
        run = st.button("Train")
        st.markdown('</div>', unsafe_allow_html=True)

        if run:
            if target is None:
                st.error("Target column is missing. Please select the target column.")
                return

            try:
                model, X_train_p, X_test_p, y_train, y_test, feature_names = train_model(
                    df=df,
                    target=target,
                    student_id=student_id,
                    numeric_cols=numeric_cols,
                    categorical_cols=categorical_cols,
                    test_size=test_size
                )
            except Exception as e:
                st.error(f"Training failed: {e}")
                return

            st.session_state["model"] = model
            st.session_state["target"] = target
            st.session_state["student_id"] = student_id
            st.session_state["numeric_cols"] = numeric_cols
            st.session_state["categorical_cols"] = categorical_cols
            st.session_state["feature_cols"] = numeric_cols + categorical_cols if (numeric_cols or categorical_cols) else [c for c in df.columns if c != target and c != student_id]
            st.session_state["X_test_p"] = X_test_p
            st.session_state["y_test"] = y_test
            st.session_state["feature_names"] = feature_names

            # Evaluate
            metrics = evaluate(model, X_test_p, y_test)

            # Metrics display
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📈 Performance metrics")

            mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
            with mcol1:
                st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            with mcol2:
                st.metric("Precision", f"{metrics['precision']:.3f}")
            with mcol3:
                st.metric("Recall", f"{metrics['recall']:.3f}")
            with mcol4:
                st.metric("F1", f"{metrics['f1']:.3f}")
            with mcol5:
                roc_display = metrics['roc_auc'] if not np.isnan(metrics['roc_auc']) else 0.0
                st.metric("ROC-AUC", f"{roc_display:.3f}")

            # Confusion matrix plot
            cm = np.array(metrics["confusion_matrix"])
            labels = [str(i) for i in sorted(np.unique(st.session_state["y_test"]))]
            cm_fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=[f"Pred {l}" for l in labels],
                y=[f"True {l}" for l in labels],
                colorscale="Purples",
                hovertemplate="True %{y}<br>Pred %{x}<br>Count %{z}<extra></extra>"
            ))
            cm_fig.update_layout(title="Confusion Matrix", margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(cm_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.success("Model trained successfully! Navigate to Dashboard, Predict, SHAP, and Counseling sections.")


def dashboard():
    if "model" not in st.session_state or st.session_state["model"] is None:
        st.warning("Model is not trained yet. Please upload a dataset and train the model first.")
        return

    model = st.session_state["model"]
    df = st.session_state.get("df", None)
    target = st.session_state.get("target", None)
    student_id = st.session_state.get("student_id", None)
    feature_cols = st.session_state.get("feature_cols", [])
    X_test_p = st.session_state.get("X_test_p", None)
    y_test = st.session_state.get("y_test", None)
    feature_names = st.session_state.get("feature_names", [])

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Risk distribution & overview")

    # Predict on test set for visualization
    try:
        y_proba = model.named_steps["clf"].predict_proba(X_test_p)
        if y_proba.shape[1] == 2:
            risk_scores = y_proba[:, 1]
        else:
            risk_scores = y_proba.max(axis=1)
    except Exception:
        preds = model.named_steps["clf"].predict(X_test_p)
        risk_scores = (preds == np.max(preds)).astype(float)

    low_thr = st.session_state.get("low_thr", 0.33)
    high_thr = st.session_state.get("high_thr", 0.66)
    risk_bands = [risk_from_proba(p, low_thr, high_thr) for p in risk_scores]

    # Histogram
    hist_fig = px.histogram(
        x=risk_scores, nbins=30,
        color=risk_bands,
        color_discrete_map=RISK_COLORS,
        labels={"x": "Dropout risk score"},
        title="Risk score distribution"
    )
    hist_fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(hist_fig, use_container_width=True)

    # Pie chart of risk categories
    st.markdown("### 🥧 Risk categories")
    risk_counts = pd.Series(risk_bands).value_counts()
    pie_fig = px.pie(
        names=risk_counts.index,
        values=risk_counts.values,
        color=risk_counts.index,
        color_discrete_map=RISK_COLORS,
        title="Risk category proportions"
    )
    pie_fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(pie_fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Feature importance
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 💡 Feature importance")
    importances = None
    try:
        importances = model.named_steps["clf"].feature_importances_
    except Exception:
        st.info("Feature importances unavailable for this classifier.")
        importances = None

    if importances is not None and len(feature_names) == len(importances):
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False).head(30)
        imp_fig = px.bar(
            imp_df, x="importance", y="feature", orientation="h",
            title="Top feature importances"
        )
        imp_fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(imp_fig, use_container_width=True)
    else:
        st.caption("Feature names or importances unavailable. Try XGBoost or RandomForest for importance.")

    st.markdown('</div>', unsafe_allow_html=True)

    # Top high-risk students table (from original df if possible)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧑‍🎓 Top high-risk students")
    if df is not None and target is not None:
        # Predict on entire dataset to list top risky students
        try:
            preds_df = predict(model, df.copy(), st.session_state["feature_cols"], low_thr, high_thr)
            if student_id and student_id in preds_df.columns:
                cols_show = [student_id, "dropout_risk_score", "dropout_risk_band"]
            else:
                # Show index instead
                preds_df["index"] = np.arange(len(preds_df))
                cols_show = ["index", "dropout_risk_score", "dropout_risk_band"]

            top_high = preds_df.sort_values("dropout_risk_score", ascending=False).head(20)[cols_show]
            st.dataframe(top_high, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute top high-risk table. Details: {e}")
    else:
        st.info("Upload and train to view top high-risk students.")
    st.markdown('</div>', unsafe_allow_html=True)


def predict_new():
    if "model" not in st.session_state or st.session_state["model"] is None:
        st.warning("Model is not trained yet. Please upload a dataset and train the model first.")
        return

    model = st.session_state["model"]
    feature_cols = st.session_state["feature_cols"]
    student_id = st.session_state.get("student_id", None)
    low_thr = st.session_state.get("low_thr", 0.33)
    high_thr = st.session_state.get("high_thr", 0.66)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧮 Predict for new students (CSV)")
    file = st.file_uploader("Upload CSV with new student records", type=["csv"], key="predict_upload")
    st.markdown('</div>', unsafe_allow_html=True)

    if file is not None:
        df_new = load_data(file)
        if df_new is None or df_new.empty:
            st.error("Failed to load prediction data or dataset is empty.")
            return

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Preview new data**")
        st.dataframe(df_new.head(30), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        try:
            preds_df = predict(model, df_new.copy(), feature_cols, low_thr, high_thr)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        # Color risk table
        def color_risk(val):
            color = RISK_COLORS.get(val, "#999999")
            return f"color: {color}; font-weight: 700;"

        styled = preds_df.style.applymap(
            lambda v: color_risk(v) if isinstance(v, str) and v in RISK_COLORS else ""
        )
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Predictions")
        st.dataframe(styled, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Download CSV
        csv_bytes = preds_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download predictions (CSV)",
            data=csv_bytes,
            file_name="umeedrise_predictions.csv",
            mime="text/csv"
        )


def shap_explainability():
    if "model" not in st.session_state or st.session_state["model"] is None:
        st.warning("Model is not trained yet. Please upload a dataset and train the model first.")
        return

    model = st.session_state["model"]
    feature_names = st.session_state.get("feature_names", [])
    X_test_p = st.session_state.get("X_test_p", None)
    X_train_p = st.session_state.get("X_test_p", None)  # Use test if train not stored

    if not SHAP_AVAILABLE:
        st.warning("SHAP is not available. Please ensure 'shap' is included in requirements.")
        return

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔍 SHAP summary plot")
    try:
        explainer, shap_values = explain_with_shap(model, X_test_p, feature_names)
        if explainer is None or shap_values is None:
            st.info("SHAP explainer could not be initialized.")
        else:
            # For binary classification, shap_values may be list [class0, class1].
            # Use the positive class if available.
            to_plot = shap_values
            if isinstance(shap_values, list) and len(shap_values) >= 2:
                to_plot = shap_values[1]

            fig = shap.summary_plot(to_plot, features=X_test_p, feature_names=feature_names, show=False)
            st.pyplot(bbox_inches="tight", clear_figure=True)
    except Exception as e:
        st.warning(f"Failed to render SHAP summary plot. Details: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 👤 Per-student explanation")
    try:
        # Select an instance
        idx = st.number_input("Select test instance index", min_value=0, max_value=int(X_test_p.shape[0] - 1), value=0, step=1)
        explainer, shap_values = explain_with_shap(model, X_test_p, feature_names)
        if explainer is None or shap_values is None:
            st.info("SHAP explainer could not be initialized.")
        else:
            if isinstance(shap_values, list) and len(shap_values) >= 2:
                values_for_instance = shap_values[1][idx]
                base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
            else:
                values_for_instance = shap_values[idx]
                base_value = explainer.expected_value

            try:
                shap.waterfall_plot(shap.Explanation(values=values_for_instance, base_values=base_value, data=X_test_p[idx], feature_names=feature_names), show=False)
                st.pyplot(bbox_inches="tight", clear_figure=True)
            except Exception:
                # Fallback to force plot rendered as matplotlib
                fig = shap.force_plot(base_value, values_for_instance, matplotlib=True)
                st.pyplot(bbox_inches="tight", clear_figure=True)

            # Top contributing features
            contrib_df = pd.DataFrame({
                "feature": feature_names,
                "shap_value": np.array(values_for_instance).flatten()
            }).sort_values("shap_value", ascending=False)
            st.markdown("#### Top contributing features")
            st.dataframe(contrib_df.head(10), use_container_width=True)
    except Exception as e:
        st.warning(f"Failed to render per-student SHAP explanation. Details: {e}")
    st.markdown('</div>', unsafe_allow_html=True)


def counseling_plan():
    if "model" not in st.session_state or st.session_state["model"] is None:
        st.warning("Model is not trained yet. Please upload a dataset and train the model first.")
        return

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧭 Counseling recommendation engine")
    st.caption("Upload a file on the Predict page to view recommendations per student. Or simulate below:")
    risk = st.selectbox("Select risk band", options=["Low", "Medium", "High"], index=2)
    recs = counseling_recommendations(risk)

    st.markdown(f"**Selected risk band:** <span class='risk-{risk.lower()}'>{risk}</span>", unsafe_allow_html=True)
    st.markdown("#### Suggested actions")
    for r in recs:
        st.markdown(f"- **Action:** {r}")
    st.markdown('</div>', unsafe_allow_html=True)

    # If predictions exist, map recommendations
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📄 Recommendations from latest predictions")
    if "latest_predictions" in st.session_state:
        preds_df = st.session_state["latest_predictions"].copy()
        if "dropout_risk_band" in preds_df.columns:
            # Attach recommendations
            preds_df["recommendations"] = preds_df["dropout_risk_band"].apply(lambda b: "; ".join(counseling_recommendations(b)))
            st.dataframe(preds_df, use_container_width=True)
            # Download
            csv_bytes = preds_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download recommendations (CSV)",
                data=csv_bytes,
                file_name="umeedrise_recommendations.csv",
                mime="text/csv"
            )
        else:
            st.info("No risk bands found in predictions. Upload on Predict page first.")
    else:
        st.info("No predictions cached yet. Upload on Predict page to generate counseling plans.")
    st.markdown('</div>', unsafe_allow_html=True)


# ============================
# Entry Point
# ============================

def main():
    st.set_page_config(
        page_title="UmeedRise – Student Dropout Prediction",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    inject_global_css()

    # Navigation
    page = sidebar_navigation()

    # Router
    if page == "🏠 Home":
        home()
    elif page == "📤 Upload + Train Model":
        upload_and_train()
    elif page == "📊 Dashboard":
        dashboard()
    elif page == "🧮 Predict New Students":
        predict_new()
    elif page == "🔍 SHAP Explainability":
        shap_explainability()
    elif page == "🧭 Counseling Plan":
        counseling_plan()
    else:
        home()

    # Cache predictions when made (hook in Predict page)
    # This small listener caches the most recent predictions for counseling.
    # It relies on the user visiting Predict New Students page.
    if "model" in st.session_state and st.session_state["model"] is not None:
        # If user is on predict page and uploaded data
        if "predict_upload" in st.session_state and st.session_state["predict_upload"] is not None:
            try:
                df_new = load_data(st.session_state["predict_upload"])
                if df_new is not None and not df_new.empty:
                    preds_df = predict(
                        st.session_state["model"],
                        df_new.copy(),
                        st.session_state["feature_cols"],
                        st.session_state.get("low_thr", 0.33),
                        st.session_state.get("high_thr", 0.66)
                    )
                    st.session_state["latest_predictions"] = preds_df
            except Exception:
                pass


if __name__ == "__main__":
    main()
