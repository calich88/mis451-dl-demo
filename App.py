import io
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import tensorflow as tf
from tensorflow import keras

KAGGLE_DATASET_URL = (
    "https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction"
)

st.set_page_config(
    page_title="MIS 451 DL for Business Demo",
    page_icon="🧠",
    layout="wide",
)

# Top Header with Corner Course Tag
head_left, head_right = st.columns([3.2, 1.3])
with head_left:
    st.title("🛒 ML vs. DL Interactive Studio")
    st.markdown(
        "**End-to-End Business AI Demo:** Load Excel/CSV ➔ Preprocess ➔ Compare ML vs. DL ➔ Predict ➔ Understand Deep Learning."
    )
with head_right:
    st.markdown(
        """
        <div style="text-align: right; padding: 10px 14px; background-color: #eff6ff;
                    border: 1px solid #bfdbfe; border-radius: 8px; margin-top: 8px;">
            <span style="font-size: 14px; font-weight: 700; color: #1e3a8a;">
                🎓 MIS 451 DL for Business Demo
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 1. DATA LOADER & FALLBACK SAMPLE GENERATOR
# =========================================================
def load_tabular_file(uploaded_file, ui_container, key_prefix="train"):
    """Load an .xlsx, .xls, or .csv file into a clean DataFrame."""
    if uploaded_file is None:
        return None

    file_name = uploaded_file.name.lower()
    if file_name.endswith((".xlsx", ".xls")):
        excel_file = pd.ExcelFile(uploaded_file)
        sheets = excel_file.sheet_names
        default_idx = sheets.index("E Comm") if "E Comm" in sheets else 0
        selected_sheet = ui_container.selectbox(
            "Select Excel Sheet:",
            sheets,
            index=default_idx,
            key=f"{key_prefix}_sheet_select",
        )
        df = excel_file.parse(selected_sheet)
    elif file_name.endswith(".csv"):
        for enc in ("utf-8", "utf-8-sig", "latin1"):
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Unable to decode CSV file. Please save it in UTF-8 format.")
    else:
        raise ValueError("Unsupported file format. Please upload .xlsx, .xls, or .csv.")

    df.columns = df.columns.astype(str).str.strip()
    return df


def load_fallback_sample():
    """Synthetic E-Commerce dataset with non-linear patterns for both Classification and Regression."""
    np.random.seed(42)
    n = 800
    tenure = np.random.randint(0, 36, n)
    sat_score = np.random.randint(1, 6, n)
    complain = np.random.choice([0, 1], n, p=[0.7, 0.3])
    order_count = np.random.randint(1, 15, n)
    order_cat = np.random.choice(["Laptop & Accessory", "Mobile Phone", "Fashion"], n)
    marital = np.random.choice(["Single", "Married", "Divorced"], n)

    cat_base = np.where(order_cat == "Laptop & Accessory", 210, np.where(order_cat == "Mobile Phone", 165, 130))
    cashback = (
        cat_base
        + 2.2 * tenure
        - 0.04 * (tenure ** 2)
        + 4.5 * np.sqrt(order_count) * sat_score
        + np.random.normal(0, 10, n)
    )
    cashback = np.round(np.clip(cashback, 50, 350), 2)

    risk = (
        (complain * 2.2)
        - (sat_score * 0.45)
        - (tenure * 0.08)
        + np.where((order_cat == "Mobile Phone") & (tenure < 6), 1.4, 0.0)
    )
    prob = 1 / (1 + np.exp(-risk))
    churn = (np.random.rand(n) < prob).astype(int)

    return pd.DataFrame({
        "Tenure": tenure,
        "SatisfactionScore": sat_score,
        "Complain": complain,
        "OrderCount": order_count,
        "PreferedOrderCat": order_cat,
        "MaritalStatus": marital,
        "CashbackAmount": cashback,
        "Churn": churn,
    })


def get_app_source_code():
    """Read this app.py file so visitors can download and run it locally."""
    try:
        return Path(__file__).read_text(encoding="utf-8")
    except Exception:
        return "# Unable to read source file directly in this environment."


# =========================================================
# 2. PREPROCESSING PIPELINE (SHARED BY TRAIN & INFERENCE)
# =========================================================
def clean_features_dataframe(df_in, feature_cols, num_cols, cat_cols):
    """Clean data types and standardize synonymous category labels for both Train & Predict."""
    df_clean = df_in[feature_cols].copy()

    for col in num_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    for col in cat_cols:
        df_clean[col] = df_clean[col].where(
            df_clean[col].isna(),
            df_clean[col].astype(str).str.strip(),
        )

    synonyms = {
        "PreferredLoginDevice": {"Phone": "Mobile Phone"},
        "PreferredPaymentMode": {"CC": "Credit Card", "COD": "Cash on Delivery"},
        "PreferedOrderCat": {"Mobile": "Mobile Phone"},
    }
    for col, mapping in synonyms.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace(mapping)

    return df_clean


def build_preprocessor(num_cols, cat_cols):
    """Build ColumnTransformer for numeric (Median + StandardScaler) and categorical (Mode + OneHotEncoder)."""
    transformers = []

    if len(num_cols) > 0:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, num_cols))

    if len(cat_cols) > 0:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipeline, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def preprocess_for_inference(raw_input_df, feature_cols, num_cols, cat_cols, fitted_preprocessor):
    """Validate columns -> Clean types -> Transform using the fitted training preprocessor."""
    missing_cols = [c for c in feature_cols if c not in raw_input_df.columns]
    if missing_cols:
        raise ValueError(f"Input data is missing required feature columns: {missing_cols}")

    cleaned_df = clean_features_dataframe(raw_input_df, feature_cols, num_cols, cat_cols)
    return fitted_preprocessor.transform(cleaned_df)


# =========================================================
# 3. MODEL TRAINING & EVALUATION (CLASSIFICATION & REGRESSION)
# =========================================================
@st.cache_resource
def train_and_evaluate(
    task_type, X_df_clean, y_arr, num_cols, cat_cols, test_size, epochs, lr, use_class_weight, use_early_stop
):
    np.random.seed(42)
    tf.random.set_seed(42)

    is_clf = task_type == "Classification"
    stratify_arg = y_arr if (is_clf and pd.Series(y_arr).value_counts().min() >= 2) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X_df_clean, y_arr, test_size=test_size, random_state=42, stratify=stratify_arg
    )

    preprocessor = build_preprocessor(num_cols, cat_cols)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    try:
        raw_out_names = preprocessor.get_feature_names_out()
        out_names = [n.replace("num__", "").replace("cat__", "") for n in raw_out_names]
    except Exception:
        out_names = [f"Feature_{i}" for i in range(X_train_proc.shape[1])]

    callbacks = []
    if use_early_stop:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
            )
        )

    batch_size = 32
    val_split = 0.2

    if is_clf:
        ml_cw = "balanced" if use_class_weight else None
        dl_cw = None
        if use_class_weight:
            weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
            dl_cw = {0: float(weights[0]), 1: float(weights[1])}

        t0 = time.time()
        ml_model = LogisticRegression(random_state=42, max_iter=1000, class_weight=ml_cw)
        ml_model.fit(X_train_proc, y_train)
        ml_time = time.time() - t0
        ml_preds = ml_model.predict(X_test_proc)

        dl_layers = (32, 16)
        dropout_rate = 0.2
        dl_model = keras.Sequential([
            keras.layers.Input(shape=(X_train_proc.shape[1],)),
            keras.layers.Dense(dl_layers[0], activation="relu"),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(dl_layers[1], activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ])
        dl_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        t1 = time.time()
        history = dl_model.fit(
            X_train_proc, y_train, epochs=epochs, batch_size=batch_size,
            validation_split=val_split, class_weight=dl_cw, callbacks=callbacks, verbose=0
        )
        dl_time = time.time() - t1

        dl_probs = dl_model.predict(X_test_proc, verbose=0).flatten()
        dl_preds = (dl_probs >= 0.5).astype(int)

        def calc_clf_metrics(name, preds, train_t):
            return {
                "Model": name,
                "Accuracy": round(accuracy_score(y_test, preds), 4),
                "Precision": round(precision_score(y_test, preds, zero_division=0), 4),
                "Recall": round(recall_score(y_test, preds, zero_division=0), 4),
                "F1-Score": round(f1_score(y_test, preds, zero_division=0), 4),
                "Train Time (s)": round(train_t, 4),
            }

        comparison_df = pd.DataFrame([
            calc_clf_metrics("Simple ML (Logistic Regression)", ml_preds, ml_time),
            calc_clf_metrics("Simple DL (Neural Network MLP)", dl_preds, dl_time),
        ])

        extra_eval = {
            "cm_ml": confusion_matrix(y_test, ml_preds),
            "cm_dl": confusion_matrix(y_test, dl_preds),
            "ml_test_probs": ml_model.predict_proba(X_test_proc)[:, 1],
            "dl_test_probs": dl_probs,
            "dl_loss_name": "binary_crossentropy",
            "dl_out_act": "sigmoid",
        }

    else:
        t0 = time.time()
        ml_model = LinearRegression()
        ml_model.fit(X_train_proc, y_train)
        ml_time = time.time() - t0
        ml_preds = ml_model.predict(X_test_proc)

        dl_layers = (64, 32)
        dropout_rate = 0.1
        dl_model = keras.Sequential([
            keras.layers.Input(shape=(X_train_proc.shape[1],)),
            keras.layers.Dense(dl_layers[0], activation="relu"),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(dl_layers[1], activation="relu"),
            keras.layers.Dense(1, activation="linear"),
        ])
        dl_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="mse",
            metrics=["mae"],
        )

        t1 = time.time()
        history = dl_model.fit(
            X_train_proc, y_train, epochs=epochs, batch_size=batch_size,
            validation_split=val_split, callbacks=callbacks, verbose=0
        )
        dl_time = time.time() - t1
        dl_preds = dl_model.predict(X_test_proc, verbose=0).flatten()

        def calc_reg_metrics(name, preds, train_t):
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            return {
                "Model": name,
                "MAE": round(float(mae), 4),
                "RMSE": round(float(rmse), 4),
                "R² Score": round(float(r2), 4),
                "Train Time (s)": round(train_t, 4),
            }

        comparison_df = pd.DataFrame([
            calc_reg_metrics("Simple ML (Linear Regression)", ml_preds, ml_time),
            calc_reg_metrics("Simple DL (Neural Network MLP)", dl_preds, dl_time),
        ])

        extra_eval = {
            "dl_loss_name": "mse",
            "dl_out_act": "linear",
        }

    return {
        "task_type": task_type,
        "preprocessor": preprocessor,
        "ml_model": ml_model,
        "dl_model": dl_model,
        "history": history.history,
        "comparison_df": comparison_df,
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "X_train_proc": X_train_proc,
        "X_test_proc": X_test_proc,
        "y_train": y_train,
        "y_test": y_test,
        "ml_preds": ml_preds,
        "dl_preds": dl_preds,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features_after_ohe": X_train_proc.shape[1],
        "feature_names_out": out_names,
        "dl_layers": dl_layers,
        "dropout_rate": dropout_rate,
        "batch_size": batch_size,
        "val_split": val_split,
        **extra_eval,
    }


def run_predictions(task_type, proc_array, ml_model, dl_model):
    if task_type == "Classification":
        ml_preds = ml_model.predict(proc_array)
        ml_probs = ml_model.predict_proba(proc_array)[:, 1]
        dl_probs = dl_model.predict(proc_array, verbose=0).flatten()
        dl_preds = (dl_probs >= 0.5).astype(int)
        return ml_preds, ml_probs, dl_preds, dl_probs
    else:
        ml_preds = ml_model.predict(proc_array)
        dl_preds = dl_model.predict(proc_array, verbose=0).flatten()
        return ml_preds, None, dl_preds, None


# =========================================================
# 4. SIDEBAR: TASK, DATA SOURCE, HYPERPARAMETERS & SOURCE CODE
# =========================================================
st.sidebar.header("🎯 0. Select Task Mode")
task_type = st.sidebar.radio(
    "Problem Type:",
    ["Classification", "Regression"],
)
is_clf = task_type == "Classification"

st.sidebar.header("⚙️ 1. Training Data Source")
source_mode = st.sidebar.radio(
    "Select Source:",
    ["Upload Excel / CSV", "Quick E-Commerce Sample"],
)

raw_df = None
if source_mode == "Upload Excel / CSV":
    uploaded_train = st.sidebar.file_uploader(
        "Upload Dataset (.xlsx, .xls, .csv)",
        type=["xlsx", "xls", "csv"],
        key="train_uploader",
    )
    st.sidebar.markdown(
        f"📥 **Download Benchmark Dataset (E-Commerce Churn):**  \n"
        f"[Open Kaggle Dataset]({KAGGLE_DATASET_URL})  \n"
        f"*(Unzip, select `E Commerce Dataset.xlsx` and sheet **`E Comm`**)*"
    )

    if uploaded_train is not None:
        try:
            raw_df = load_tabular_file(uploaded_train, st.sidebar, key_prefix="train")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
            st.stop()
    else:
        st.info(
            f"👈 Please upload `E Commerce Dataset.xlsx` on the left sidebar "
            f"(available at [Kaggle]({KAGGLE_DATASET_URL})), "
            f"or switch to **Quick E-Commerce Sample** to test immediately."
        )
        st.stop()
else:
    raw_df = load_fallback_sample()
    st.sidebar.markdown(f"📥 Want to try real data? [Download on Kaggle]({KAGGLE_DATASET_URL})")

all_cols = raw_df.columns.tolist()
id_keywords = {"customerid", "id", "rownumber", "unnamed: 0"}

if is_clf:
    default_target_idx = all_cols.index("Churn") if "Churn" in all_cols else len(all_cols) - 1
    target_col = st.sidebar.selectbox(
        "Target Column (Binary 0/1):",
        all_cols,
        index=default_target_idx,
        key="target_clf",
    )
else:
    numeric_candidates = [
        c for c in raw_df.select_dtypes(include=[np.number]).columns.tolist()
        if c.lower() not in id_keywords
    ]
    if len(numeric_candidates) == 0:
        st.error("⚠️ No numeric columns available for Regression.")
        st.stop()
    default_reg_idx = numeric_candidates.index("CashbackAmount") if "CashbackAmount" in numeric_candidates else 0
    target_col = st.sidebar.selectbox(
        "Target Column (Numeric):",
        numeric_candidates,
        index=default_reg_idx,
        key="target_reg",
    )

candidate_features = [c for c in all_cols if c != target_col]
default_features = [c for c in candidate_features if c.lower() not in id_keywords]
feature_cols = st.sidebar.multiselect(
    "Select Feature Columns:",
    candidate_features,
    default=default_features,
    key=f"features_{task_type}",
)

if len(feature_cols) == 0:
    st.warning("⚠️ Please select at least 1 feature column.")
    st.stop()

df = raw_df.dropna(subset=[target_col]).copy()
if is_clf:
    if df[target_col].nunique() != 2:
        st.error("⚠️ Classification requires a binary target column (exactly 2 unique values). If you want to predict a continuous number like `CashbackAmount`, switch to **Regression** at the top of the sidebar.")
        st.stop()
    unique_vals = sorted(df[target_col].unique())
    if unique_vals != [0, 1]:
        df[target_col] = df[target_col].map({unique_vals[0]: 0, unique_vals[1]: 1})
    y_arr = df[target_col].astype(int).values
else:
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col]).copy()
    y_arr = df[target_col].astype(float).values

lbl_0, lbl_1 = ("Retained (0)", "Churned (1)") if target_col.lower() == "churn" else ("Class 0", "Class 1")

num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in feature_cols if c not in num_cols]
X_clean_df = clean_features_dataframe(df, feature_cols, num_cols, cat_cols)

st.sidebar.header("🎛️ 2. Training Hyperparameters")
test_size = st.sidebar.slider("Test Split Ratio", 0.1, 0.4, 0.2, 0.05)
epochs = st.sidebar.slider("Epochs (DL)", 10, 150, 60 if not is_clf else 50, 10)
lr = st.sidebar.select_slider("Learning Rate (DL)", options=[0.001, 0.005, 0.01, 0.02], value=0.005)
use_cw = st.sidebar.checkbox("Balance Classes (Class Weight)", value=True) if is_clf else False
use_es = st.sidebar.checkbox("Early Stopping (Prevent Overfitting)", value=True)

st.sidebar.divider()
st.sidebar.header("💻 3. Download Code & Run Locally")

source_code_str = get_app_source_code()
req_txt_str = "streamlit\nnumpy\npandas\nscikit-learn\ntensorflow\nmatplotlib\nopenpyxl\n"

st.sidebar.download_button(
    "⬇️ Download Source Code (app.py)",
    data=source_code_str.encode("utf-8"),
    file_name="app.py",
    mime="text/x-python",
    width="stretch",
)
st.sidebar.download_button(
    "⬇️ Download requirements.txt",
    data=req_txt_str.encode("utf-8"),
    file_name="requirements.txt",
    mime="text/plain",
    width="stretch",
)

with st.sidebar.expander("📖 How to Run on Windows PowerShell"):
    st.markdown("**Step 1:** Create a folder (e.g., `D:\\App Demo`) and place `app.py` and `requirements.txt` inside.")
    st.markdown("**Step 2:** Open PowerShell in that folder (type `powershell` in the File Explorer address bar and press Enter).")
    st.markdown("**Step 3:** Run the following commands in order:")
    st.code(
        "python -m venv venv\n"
        "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser\n"
        ".\\venv\\Scripts\\Activate.ps1\n"
        "pip install -r requirements.txt\n"
        "streamlit run app.py",
        language="powershell",
    )
    st.caption("Press Ctrl + C in PowerShell to stop the app when finished.")

with st.spinner(f"Preprocessing and training {task_type} models (ML & DL)..."):
    results = train_and_evaluate(
        task_type, X_clean_df, y_arr, num_cols, cat_cols, test_size, epochs, lr, use_cw, use_es
    )


# =========================================================
# 5. MAIN INTERFACE (5 TABS)
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data & Preprocessing",
    "⚖️ ML vs DL Comparison",
    "🔮 Prediction (Single & Batch)",
    "📚 Theory: ML vs DL",
    "🎨 How DL Works (Simple Guide)",
])

# --- TAB 1: RAW DATA, PREPROCESSED TRAIN & TEST DATA ---
with tab1:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("1️⃣ Cleaned Input Data (Before Numeric Scaling / One-Hot Encoding)")
        preview_raw_clean = X_clean_df.copy()
        preview_raw_clean[f"TARGET ({target_col})"] = y_arr
        st.dataframe(preview_raw_clean.head(10))
    with c2:
        if is_clf:
            st.subheader(f"Target Distribution (`{target_col}`)")
            counts = df[target_col].value_counts().rename({0: lbl_0, 1: lbl_1})
            st.bar_chart(counts)
            st.metric("Total Valid Rows", f"{len(df):,}")
            st.metric("Positive Class Rate (1)", f"{df[target_col].mean() * 100:.1f}%")
        else:
            st.subheader(f"Target Distribution (`{target_col}`)")
            fig_hist, ax_hist = plt.subplots(figsize=(4.5, 3))
            ax_hist.hist(y_arr, bins=25, color="#3b82f6", edgecolor="white")
            ax_hist.set_xlabel(target_col)
            ax_hist.set_ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig_hist)
            plt.close(fig_hist)
            st.metric("Total Valid Rows", f"{len(df):,}")
            st.metric("Mean (Min - Max)", f"{np.mean(y_arr):.2f} ({np.min(y_arr):.1f} - {np.max(y_arr):.1f})")

    st.divider()
    st.subheader("2️⃣ Preprocessing Pipeline Summary")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Numeric Features", len(num_cols), help="Imputed with Median + Scaled via StandardScaler")
    p2.metric("Categorical Features", len(cat_cols), help="Synonyms merged + Imputed with Mode + OneHotEncoder")
    p3.metric("Dimensions After One-Hot", results["n_features_after_ohe"])
    p4.metric("Train / Test Split", f"{results['n_train']:,} / {results['n_test']:,}")

    missing_series = X_clean_df.isna().sum()
    missing_df = missing_series[missing_series > 0].reset_index()
    if len(missing_df) > 0:
        missing_df.columns = ["Column Name", "Missing Values (NaN)"]
        st.caption("Columns with missing values (automatically imputed inside the pipeline):")
        st.dataframe(missing_df, hide_index=True)
    else:
        st.success("✅ Selected feature columns contain no missing values.")

    st.divider()
    st.subheader("3️⃣ Actual Preprocessed Data Fed Into Train & Test")
    st.caption(
        "These are the exact numeric matrices passed into both ML and DL: numeric columns are standardized around 0 via **StandardScaler**, "
        "and categorical columns are binary-encoded via **OneHotEncoder**. "
        "The Imputer, Scaler, and Encoder are **fitted strictly on the Training Set** before transforming the Test Set."
    )

    train_proc_df = pd.DataFrame(np.round(results["X_train_proc"], 4), columns=results["feature_names_out"])
    train_proc_df[f"TARGET ({target_col})"] = results["y_train"]

    test_proc_df = pd.DataFrame(np.round(results["X_test_proc"], 4), columns=results["feature_names_out"])
    test_proc_df[f"TARGET ({target_col})"] = results["y_test"]

    n_preview = st.slider("Number of preview rows:", 5, 50, 10, 5, key="preview_proc_rows")

    col_tr, col_te = st.columns(2)
    with col_tr:
        st.markdown(f"**📘 Preprocessed TRAIN Set** (`{train_proc_df.shape[0]:,} rows × {results['n_features_after_ohe']} features + 1 Target`)")
        st.dataframe(train_proc_df.head(n_preview))
        st.download_button(
            "⬇️ Download Full Preprocessed Train Set (.CSV)",
            data=train_proc_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="train_preprocessed.csv",
            mime="text/csv",
            width="stretch",
        )
    with col_te:
        st.markdown(f"**📙 Preprocessed TEST Set** (`{test_proc_df.shape[0]:,} rows × {results['n_features_after_ohe']} features + 1 Target`)")
        st.dataframe(test_proc_df.head(n_preview))
        st.download_button(
            "⬇️ Download Full Preprocessed Test Set (.CSV)",
            data=test_proc_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="test_preprocessed.csv",
            mime="text/csv",
            width="stretch",
        )

# --- TAB 2: ML VS DL PERFORMANCE COMPARISON ---
with tab2:
    st.subheader(f"Performance Comparison ({task_type} — Target: `{target_col}`)")
    st.dataframe(results["comparison_df"], hide_index=True)

    col_c1, col_c2 = st.columns(2)

    if is_clf:
        with col_c1:
            st.subheader("Metrics Comparison")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
            x_idx = np.arange(len(metrics))
            width = 0.35

            ml_vals = [results["comparison_df"].loc[0, m] for m in metrics]
            dl_vals = [results["comparison_df"].loc[1, m] for m in metrics]

            ax1.bar(x_idx - width / 2, ml_vals, width, label="ML (Logistic Reg)", color="#3b82f6")
            ax1.bar(x_idx + width / 2, dl_vals, width, label="DL (Neural Net)", color="#f97316")
            ax1.set_xticks(x_idx)
            ax1.set_xticklabels(metrics)
            ax1.set_ylim(0, 1.15)
            ax1.legend()
            ax1.grid(axis="y", linestyle="--", alpha=0.5)
            st.pyplot(fig1)
            plt.close(fig1)

        with col_c2:
            st.subheader(f"DL Training Loss Curve ({len(results['history']['loss'])} epochs)")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(results["history"]["loss"], label="Train Loss", color="#10b981")
            ax2.plot(results["history"]["val_loss"], label="Validation Loss", color="#ef4444")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Binary Crossentropy")
            ax2.legend()
            ax2.grid(linestyle="--", alpha=0.5)
            st.pyplot(fig2)
            plt.close(fig2)

        st.subheader("Confusion Matrices")
        cm_col1, cm_col2 = st.columns(2)

        def plot_cm(cm, title):
            fig, ax = plt.subplots(figsize=(4.2, 3.2))
            ax.imshow(cm, cmap="Blues")
            ax.set_title(title)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels([f"Pred {lbl_0}", f"Pred {lbl_1}"])
            ax.set_yticklabels([f"Actual {lbl_0}", f"Actual {lbl_1}"])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=12, fontweight="bold")
            plt.tight_layout()
            return fig

        with cm_col1:
            f1 = plot_cm(results["cm_ml"], "Simple ML (Logistic Regression)")
            st.pyplot(f1)
            plt.close(f1)
        with cm_col2:
            f2 = plot_cm(results["cm_dl"], "Simple DL (Neural Network)")
            st.pyplot(f2)
            plt.close(f2)

    else:
        with col_c1:
            st.subheader("R² Score Comparison (Closer to 1.0 is better)")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            models_lbl = ["ML (Linear Reg)", "DL (Neural Net)"]
            r2_vals = [results["comparison_df"].loc[0, "R² Score"], results["comparison_df"].loc[1, "R² Score"]]
            bars = ax1.bar(models_lbl, r2_vals, color=["#3b82f6", "#f97316"], width=0.5)
            ax1.set_ylim(min(0.0, min(r2_vals) - 0.1), 1.1)
            ax1.axhline(0, color="black", linewidth=0.8)
            ax1.grid(axis="y", linestyle="--", alpha=0.5)
            for bar, v in zip(bars, r2_vals):
                ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.4f}", ha="center", fontweight="bold")
            st.pyplot(fig1)
            plt.close(fig1)

        with col_c2:
            st.subheader(f"DL Training Loss Curve ({len(results['history']['loss'])} epochs)")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(results["history"]["loss"], label="Train Loss (MSE)", color="#10b981")
            ax2.plot(results["history"]["val_loss"], label="Validation Loss (MSE)", color="#ef4444")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Mean Squared Error (MSE)")
            ax2.legend()
            ax2.grid(linestyle="--", alpha=0.5)
            st.pyplot(fig2)
            plt.close(fig2)

        st.subheader("Actual vs. Predicted on Test Set")
        st.caption("Points closer to the dashed diagonal line represent more accurate predictions.")
        sc_col1, sc_col2 = st.columns(2)

        def plot_scatter_reg(y_true, y_pred, title, color):
            fig, ax = plt.subplots(figsize=(5.5, 4))
            ax.scatter(y_true, y_pred, alpha=0.45, color=color, edgecolors="white", s=30)
            min_val = min(np.min(y_true), np.min(y_pred))
            max_val = max(np.max(y_true), np.max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1.5, label="Ideal Line (y = x)")
            ax.set_title(title)
            ax.set_xlabel(f"Actual ({target_col})")
            ax.set_ylabel(f"Predicted ({target_col})")
            ax.legend()
            ax.grid(linestyle="--", alpha=0.4)
            return fig

        with sc_col1:
            f_s1 = plot_scatter_reg(results["y_test"], results["ml_preds"], "Simple ML (Linear Regression)", "#3b82f6")
            st.pyplot(f_s1)
            plt.close(f_s1)
        with sc_col2:
            f_s2 = plot_scatter_reg(results["y_test"], results["dl_preds"], "Simple DL (Neural Network)", "#f97316")
            st.pyplot(f_s2)
            plt.close(f_s2)

# --- TAB 3: SINGLE & BATCH PREDICTION ---
with tab3:
    predict_mode = st.radio(
        "Select Prediction Mode:",
        ["🧍 Single Prediction (Manual Input)", "📂 Batch Prediction (Upload Excel / CSV)"],
        horizontal=True,
    )
    st.divider()

    if predict_mode == "🧍 Single Prediction (Manual Input)":
        st.subheader(f"Enter Customer Attributes to Predict `{target_col}`")
        with st.form("single_predict_form"):
            form_cols = st.columns(3)
            user_record = {}

            for idx, col in enumerate(feature_cols):
                with form_cols[idx % 3]:
                    if col in num_cols:
                        s = X_clean_df[col].dropna()
                        is_int = pd.api.types.is_integer_dtype(s) or np.all(s % 1 == 0)
                        if is_int:
                            min_v, max_v, mean_v = int(s.min()), int(s.max()), int(round(s.mean()))
                            if min_v == max_v:
                                max_v = min_v + 1
                            user_record[col] = st.number_input(
                                col, min_value=min_v, max_value=max_v * 2, value=mean_v, step=1
                            )
                        else:
                            min_v, max_v, mean_v = float(s.min()), float(s.max()), float(s.mean())
                            if min_v == max_v:
                                max_v = min_v + 1.0
                            user_record[col] = st.number_input(
                                col, value=round(mean_v, 2), format="%.2f"
                            )
                    else:
                        opts = sorted(X_clean_df[col].dropna().astype(str).unique().tolist())
                        user_record[col] = st.selectbox(col, opts)

            submitted = st.form_submit_button("🚀 Predict Now", type="primary", width="stretch")

        if submitted:
            single_df = pd.DataFrame([user_record])
            proc_input = preprocess_for_inference(
                single_df, feature_cols, num_cols, cat_cols, results["preprocessor"]
            )
            ml_p, ml_prob, dl_p, dl_prob = run_predictions(
                task_type, proc_input, results["ml_model"], results["dl_model"]
            )

            r1, r2 = st.columns(2)
            if is_clf:
                with r1:
                    st.markdown("### 🤖 Simple ML (Logistic Regression)")
                    if ml_p[0] == 1:
                        st.error(f"**{lbl_1.upper()}** — Probability: {ml_prob[0]:.2%}")
                    else:
                        st.success(f"**{lbl_0.upper()}** — Positive Probability: {ml_prob[0]:.2%}")

                with r2:
                    st.markdown("### 🧠 Simple DL (Neural Network)")
                    if dl_p[0] == 1:
                        st.error(f"**{lbl_1.upper()}** — Probability: {dl_prob[0]:.2%}")
                    else:
                        st.success(f"**{lbl_0.upper()}** — Positive Probability: {dl_prob[0]:.2%}")
            else:
                with r1:
                    st.markdown("### 🤖 Simple ML (Linear Regression)")
                    st.info(f"Predicted **{target_col}**: **`{ml_p[0]:.2f}`**")
                with r2:
                    st.markdown("### 🧠 Simple DL (Neural Network)")
                    st.success(f"Predicted **{target_col}**: **`{dl_p[0]:.2f}`**")

    else:
        st.subheader(f"Upload Excel / CSV for Batch Prediction (`{target_col}`)")
        st.caption(f"Your file must contain all **{len(feature_cols)} feature columns**: `{', '.join(feature_cols)}` (extra ID columns are preserved in the output).")

        sample_template = df[feature_cols].head(10)
        csv_template = sample_template.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 Download Sample Template (10 Rows CSV)",
            data=csv_template,
            file_name=f"batch_template_{task_type.lower()}.csv",
            mime="text/csv",
        )

        uploaded_batch = st.file_uploader(
            "Upload File for Prediction (.xlsx, .xls, .csv)",
            type=["xlsx", "xls", "csv"],
            key="batch_uploader",
        )

        if uploaded_batch is not None:
            try:
                batch_raw_df = load_tabular_file(uploaded_batch, st, key_prefix="batch")
                if batch_raw_df is not None and len(batch_raw_df) > 0:
                    proc_batch = preprocess_for_inference(
                        batch_raw_df, feature_cols, num_cols, cat_cols, results["preprocessor"]
                    )
                    ml_preds, ml_probs, dl_preds, dl_probs = run_predictions(
                        task_type, proc_batch, results["ml_model"], results["dl_model"]
                    )

                    out_df = batch_raw_df.copy()
                    if is_clf:
                        out_df["ML_Prediction"] = np.where(ml_preds == 1, lbl_1, lbl_0)
                        out_df["ML_Positive_Prob"] = np.round(ml_probs, 4)
                        out_df["DL_Prediction"] = np.where(dl_preds == 1, lbl_1, lbl_0)
                        out_df["DL_Positive_Prob"] = np.round(dl_probs, 4)

                        st.success(f"✅ Successfully predicted **{len(out_df):,}** rows!")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Total Rows", f"{len(out_df):,}")
                        m2.metric(f"ML Predicted {lbl_1}", f"{int(ml_preds.sum()):,} ({ml_preds.mean()*100:.1f}%)")
                        m3.metric(f"DL Predicted {lbl_1}", f"{int(dl_preds.sum()):,} ({dl_preds.mean()*100:.1f}%)")
                    else:
                        out_df[f"ML_Pred_{target_col}"] = np.round(ml_preds, 2)
                        out_df[f"DL_Pred_{target_col}"] = np.round(dl_preds, 2)

                        st.success(f"✅ Successfully predicted **{len(out_df):,}** rows!")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Total Rows", f"{len(out_df):,}")
                        m2.metric(f"ML Mean Pred ({target_col})", f"{np.mean(ml_preds):.2f}")
                        m3.metric(f"DL Mean Pred ({target_col})", f"{np.mean(dl_preds):.2f}")

                    st.dataframe(out_df.head(50))

                    dl_col1, dl_col2 = st.columns(2)
                    with dl_col1:
                        csv_out = out_df.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            "⬇️ Download Predictions (.CSV)",
                            data=csv_out,
                            file_name=f"predictions_{task_type.lower()}.csv",
                            mime="text/csv",
                            width="stretch",
                        )
                    with dl_col2:
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                            out_df.to_excel(writer, index=False, sheet_name="Predictions")
                        st.download_button(
                            "⬇️ Download Predictions (.XLSX)",
                            data=excel_buffer.getvalue(),
                            file_name=f"predictions_{task_type.lower()}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            width="stretch",
                        )
                elif batch_raw_df is not None:
                    st.warning("⚠️ The uploaded file contains 0 rows.")
            except ValueError as ve:
                st.error(f"❌ Input Data Error: {ve}")
            except Exception as e:
                st.error(f"❌ An error occurred while processing the file: {e}")

# --- TAB 4: THEORY COMPARISON ---
with tab4:
    st.header(f"📚 Theoretical Foundation: ML vs. DL (Current Task: {task_type} — `{target_col}`)")
    h1, h2 = results["dl_layers"]

    col_t1, col_t2 = st.columns(2)
    if is_clf:
        with col_t1:
            st.info("### 🤖 Simple ML: Logistic Regression")
            st.markdown(
                rf"""
                * **Inputs:** {results['n_features_after_ohe']} preprocessed features.
                * **Core Mechanism:** A **linear** model — multiplies each feature by a fixed weight, sums them up, and applies a **Sigmoid** function to output a probability between `0` and `1`:
                  $$z = w_1 x_1 + w_2 x_2 + ... + w_{{{results['n_features_after_ohe']}}} x_{{{results['n_features_after_ohe']}}} + b$$
                  $$P({target_col}=1) = \frac{{1}}{{1 + e^{{-z}}}}$$
                * **Loss Function:** Binary Crossentropy.
                * **Strengths:** Ultra-fast, highly interpretable (inspect `w` directly), low overfitting risk.
                """
            )
        with col_t2:
            st.warning(f"### 🧠 Simple DL: Neural Network (MLP {h1} → {h2} → 1)")
            st.markdown(
                rf"""
                * **Inputs:** The exact same {results['n_features_after_ohe']} preprocessed features.
                * **Core Mechanism:** Stacks 2 hidden layers with non-linear **ReLU (`{h1}` and `{h2}` neurons)** before the **Sigmoid** output layer.
                * **How It Learns:** Automatically discovers cross-feature interactions in hidden layers (patterns a straight line cannot separate) and refines thousands of weights via **Backpropagation**.
                * **Strengths:** Captures complex, curved decision boundaries in behavioral data.
                """
            )
    else:
        with col_t1:
            st.info("### 🤖 Simple ML: Linear Regression")
            st.markdown(
                rf"""
                * **Inputs:** {results['n_features_after_ohe']} preprocessed features.
                * **Core Mechanism:** Fits a single **linear hyperplane** through the data:
                  $$\widehat{{{target_col}}} = w_1 x_1 + w_2 x_2 + ... + w_{{{results['n_features_after_ohe']}}} x_{{{results['n_features_after_ohe']}}} + b$$
                * **Loss Function:** MSE (Mean Squared Error).
                * **Limitation:** Assumes every feature has a constant, straight-line effect on `{target_col}` regardless of other feature values.
                """
            )
        with col_t2:
            st.warning(f"### 🧠 Simple DL: Neural Network (MLP {h1} → {h2} → 1)")
            st.markdown(
                rf"""
                * **Inputs:** The exact same {results['n_features_after_ohe']} preprocessed features.
                * **Core Mechanism:** Uses 2 **ReLU (`{h1}` and `{h2}` neurons)** hidden layers to combine linear pieces into **smooth non-linear curves**, followed by a **`linear`** output neuron.
                * **Loss Function:** MSE (Mean Squared Error).
                * **Strengths:** Captures saturation thresholds, curved trends, and interactions between categories and numeric behaviors.
                """
            )

    st.divider()
    st.subheader("📋 Side-by-Side Comparison")
    theory_table = pd.DataFrame([
        ["Pattern Complexity", "Linear (straight line / hyperplane)", "Non-linear (bends & curves flexibly)"],
        ["Feature Interactions", "Manual (requires manual feature engineering)", "Automatic (hidden ReLU layers combine features)"],
        ["Data Size Needed", "Works well even on small datasets (hundreds of rows)", "Shines on larger datasets (thousands to millions)"],
        ["Training Speed", "Instant (milliseconds)", "Slower (seconds to hours)"],
        ["Interpretability", "High (transparent weights w)", "Lower (often treated as a Black Box)"],
    ], columns=["Criterion", "Simple ML (Logistic / Linear Regression)", "Simple DL (Neural Network - MLP)"])
    st.dataframe(theory_table, hide_index=True)

# --- TAB 5: SIMPLE, BUSINESS-FRIENDLY DEEP LEARNING VISUALIZER ---
with tab5:
    st.header("🧠 How Deep Learning Works — A Visual Guide for Business Users")
    st.markdown(
        "No heavy math! Think of a **Neural Network (MLP)** like a **Smart Review Team** "
        "that reads customer data, spots patterns across multiple layers, and learns from its mistakes."
    )

    dl_model = results["dl_model"]
    dense_layers = [l for l in dl_model.layers if isinstance(l, keras.layers.Dense)]
    h1, h2 = results["dl_layers"]
    n_in = results["n_features_after_ohe"]
    total_params = dl_model.count_params()
    epochs_ran = len(results["history"]["loss"])

    # =====================================================
    # PART 1: VISUAL DIAGRAM OF ALL LAYERS & NODE COUNTS
    # =====================================================
    st.subheader("1️⃣ What Does Our Neural Network Look Like? (Layers & Neuron Counts)")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("1. Input Layer", f"{n_in} Features", help="All customer attributes after cleaning & encoding")
    m2.metric("2. Hidden Layer 1", f"{h1} Neurons", help="First team of pattern spotters (ReLU)")
    m3.metric("3. Hidden Layer 2", f"{h2} Neurons", help="Second team combining clues (ReLU)")
    m4.metric("4. Output Layer", "1 Answer", help=f"Final {'Risk %' if is_clf else target_col + ' Value'}")

    # Draw full layer diagram clearly showing node counts
    fig_arch, ax_arch = plt.subplots(figsize=(11.5, 4.4))
    ax_arch.axis("off")

    def get_draw_positions(count, max_draw=10):
        """Return y positions for nodes to draw, with a flag if truncated with '...'."""
        if count <= max_draw:
            return np.linspace(0.22, 0.84, count), False
        # Draw top 4 and bottom 4 with '...' gap in between
        top = np.linspace(0.84, 0.62, 4)
        bot = np.linspace(0.42, 0.22, 4)
        return np.concatenate([top, bot]), True

    layer_counts = [n_in, h1, h2, 1]
    x_pos_list = [0.12, 0.40, 0.68, 0.90]
    layer_colors = ["#64748b", "#3b82f6", "#8b5cf6", "#f97316"]
    layer_titles = [
        f"INPUT LAYER\n({n_in} Features)\nCustomer Info",
        f"HIDDEN LAYER 1\n({h1} Neurons)\nSpots Basic Clues",
        f"HIDDEN LAYER 2\n({h2} Neurons)\nCombines Patterns",
        f"OUTPUT LAYER\n(1 Neuron)\n{'Churn Risk %' if is_clf else target_col}",
    ]

    all_coords = []
    for idx_l, (cnt, xp, col, title) in enumerate(zip(layer_counts, x_pos_list, layer_colors, layer_titles)):
        ys, truncated = get_draw_positions(cnt, max_draw=10)
        coords = [(xp, y) for y in ys]
        all_coords.append(coords)

        # Draw soft box around each layer
        box_h = 0.72 if cnt > 1 else 0.20
        box_y = 0.17 if cnt > 1 else 0.43
        rect = plt.Rectangle((xp - 0.055, box_y), 0.11, box_h, facecolor=col, alpha=0.08, edgecolor=col, linewidth=1.2, zorder=0)
        ax_arch.add_patch(rect)

        for (x, y) in coords:
            c = plt.Circle((x, y), 0.020, facecolor=col, edgecolor="white", linewidth=1.2, zorder=3)
            ax_arch.add_patch(c)

        if truncated:
            ax_arch.text(xp, 0.52, f"⋮\n({cnt} total)", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#1e293b", zorder=4)

        ax_arch.text(xp, 0.07, title, ha="center", va="center", fontsize=9, fontweight="bold", color="#0f172a")

    # Connect adjacent layers
    for i in range(len(all_coords) - 1):
        for (x1, y1) in all_coords[i]:
            for (x2, y2) in all_coords[i + 1]:
                ax_arch.plot([x1, x2], [y1, y2], color="#94a3b8", linewidth=0.5, alpha=0.35, zorder=1)

    ax_arch.annotate(
        "Information Flows Forward (Left ➔ Right)",
        xy=(0.82, 0.94), xytext=(0.18, 0.94),
        arrowprops=dict(arrowstyle="->", color="#1e293b", lw=1.8),
        fontsize=9.5, fontweight="bold", va="center", ha="center", color="#1e293b"
    )

    ax_arch.set_xlim(0, 1.02)
    ax_arch.set_ylim(0, 1.0)
    plt.tight_layout()
    st.pyplot(fig_arch)
    plt.close(fig_arch)

    # Plain-English translation of Sidebar knobs
    st.info(
        f"""
        **🎛️ What Do Your Sidebar Settings Mean in Plain English?**
        * **Epochs (`{epochs}` max, ran `{epochs_ran}`):** How many times the AI reads through the entire training dataset to practice (just like studying a textbook multiple times).
        * **Learning Rate (`{lr}`):** How big of an adjustment the AI makes each time it gets an answer wrong (too small = learns slowly; too large = overshoots).
        * **Early Stopping (`{'ON' if use_es else 'OFF'}`):** Automatically stops studying once the AI stops improving on new data, preventing it from memorizing noise (*Overfitting*).
        * **Total Learned Knobs (`{total_params:,}` weights):** Unlike simple ML (1 weight per feature), this network tunes **{total_params:,} tiny connections** across its layers!
        """
    )

    with st.expander("💻 Optional: View the Exact Python Code Behind This Model"):
        st.code(
            f"dl_model = keras.Sequential([\n"
            f"    keras.layers.Input(shape=({n_in},)),\n"
            f"    keras.layers.Dense({h1}, activation='relu'),\n"
            f"    keras.layers.Dropout({results['dropout_rate']}),\n"
            f"    keras.layers.Dense({h2}, activation='relu'),\n"
            f"    keras.layers.Dense(1, activation='{results['dl_out_act']}'),\n"
            f"])\n"
            f"dl_model.compile(optimizer=Adam(learning_rate={lr}), loss='{results['dl_loss_name']}')\n"
            f"dl_model.fit(X_train, y_train, epochs={epochs}, batch_size={results['batch_size']})",
            language="python",
        )

    st.divider()

    # =====================================================
    # PART 2: WATCH 1 REAL CUSTOMER FLOW THROUGH THE NETWORK
    # =====================================================
    st.subheader("2️⃣ Follow One Real Customer Through the Network")
    st.caption("Pick any customer from the Test Set to see how the Hidden Layers light up and produce the final decision.")

    sample_idx = st.number_input(
        f"Choose Test Customer # (0 to {results['n_test'] - 1}):",
        min_value=0,
        max_value=results["n_test"] - 1,
        value=0,
        step=1,
        key="dl_simple_idx",
    )

    raw_row_s = results["X_test_raw"].iloc[sample_idx]
    x_vec = results["X_test_proc"][sample_idx]
    y_true_val = results["y_test"][sample_idx]

    x_tensor = tf.convert_to_tensor(x_vec.reshape(1, -1), dtype=tf.float32)
    dense_outputs = []
    for layer in dl_model.layers:
        x_tensor = layer(x_tensor, training=False)
        if isinstance(layer, keras.layers.Dense):
            dense_outputs.append(x_tensor.numpy().flatten())

    h1_acts, h2_acts = dense_outputs[0], dense_outputs[1]
    out_val = float(dense_outputs[-1][0])

    s_c1, s_c2, s_c3 = st.columns([1.1, 1.4, 0.9])

    with s_c1:
        st.markdown("**Step A: Customer Profile (Input)**")
        st.dataframe(raw_row_s.astype(str).to_frame(name="Customer Value"))

    with s_c2:
        st.markdown("**Step B: How Neurons React (Hidden Layers)**")
        n_on_1 = int(np.sum(h1_acts > 0))
        n_on_2 = int(np.sum(h2_acts > 0))
        st.caption(
            f"Each bar is 1 neuron. **Blue/Purple bars (> 0)** are neurons that 'woke up' because they spotted a relevant clue for this customer; "
            f"**Gray slots (= 0)** stayed quiet (ReLU filter)."
        )

        fig_react, (ax_r1, ax_r2) = plt.subplots(2, 1, figsize=(5.5, 4.0))
        ax_r1.bar(range(1, len(h1_acts) + 1), h1_acts, color=["#3b82f6" if v > 0 else "#e2e8f0" for v in h1_acts])
        ax_r1.set_title(f"Hidden Layer 1: {n_on_1} / {h1} neurons activated", fontsize=9.5, fontweight="bold")
        ax_r1.set_ylabel("Signal")
        ax_r1.grid(axis="y", alpha=0.3)

        ax_r2.bar(range(1, len(h2_acts) + 1), h2_acts, color=["#8b5cf6" if v > 0 else "#e2e8f0" for v in h2_acts])
        ax_r2.set_title(f"Hidden Layer 2: {n_on_2} / {h2} neurons activated", fontsize=9.5, fontweight="bold")
        ax_r2.set_xlabel("Neuron Number")
        ax_r2.set_ylabel("Signal")
        ax_r2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig_react)
        plt.close(fig_react)

    with s_c3:
        st.markdown("**Step C: Final Decision (Output)**")
        if is_clf:
            pred_cls = 1 if out_val >= 0.5 else 0
            st.metric("AI Risk Score", f"{out_val:.1%}")
            if pred_cls == 1:
                st.error(f"AI Predicts: **{lbl_1}**")
            else:
                st.success(f"AI Predicts: **{lbl_0}**")
            st.metric("Actual Reality", lbl_1 if y_true_val == 1 else lbl_0)
            if pred_cls == y_true_val:
                st.caption("✅ Correct prediction!")
            else:
                st.caption("❌ Missed — during training, mistakes like this trigger weight adjustments.")
        else:
            diff = abs(y_true_val - out_val)
            st.metric(f"AI Predicted {target_col}", f"{out_val:.2f}")
            st.metric(f"Actual {target_col}", f"{y_true_val:.2f}")
            st.metric("Difference (Error)", f"{diff:.2f}")

    st.divider()

    # =====================================================
    # PART 3: HOW THE AI IMPROVES OVER TIME (EPOCHS & LEARNING RATE)
    # =====================================================
    st.subheader("3️⃣ How Does the AI Learn? (Practice Rounds & Correction Speed)")

    lr_col, ep_col = st.columns(2)

    with lr_col:
        st.markdown("**🎯 Effect of `Learning Rate` (Step Size of Correction)**")
        st.caption("Imagine walking blindfolded down a valley to find the lowest mistake (Loss):")

        w_curve = np.linspace(-4, 4, 200)
        loss_curve = w_curve ** 2

        def sim_gd(start_w, step_lr, n_steps=7):
            ws = [start_w]
            w = start_w
            for _ in range(n_steps):
                w = w - step_lr * (2 * w)
                ws.append(w)
            return np.array(ws), np.array(ws) ** 2

        fig_gd, axes_gd = plt.subplots(1, 3, figsize=(9.5, 3.0))
        for ax, (ttl, lrd, col_g) in zip(
            axes_gd,
            [
                ("Too Small\n(Crawls slowly)", 0.03, "#3b82f6"),
                ("Just Right\n(Reaches best answer)", 0.25, "#16a34a"),
                ("Too Large\n(Bounces wildly!)", 0.95, "#dc2626"),
            ],
        ):
            ws_p, ls_p = sim_gd(-3.5, lrd)
            ax.plot(w_curve, loss_curve, color="#94a3b8", linewidth=1.5)
            ax.plot(ws_p, ls_p, marker="o", markersize=4.5, color=col_g, linewidth=2)
            ax.scatter([0], [0], color="gold", edgecolors="black", s=60, zorder=5)
            ax.set_title(ttl, fontsize=8.5, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim(-0.5, 16)
        axes_gd[0].set_ylabel("Mistakes (Loss)")
        plt.tight_layout()
        st.pyplot(fig_gd)
        plt.close(fig_gd)
        st.caption(f"Your current Learning Rate is **`{lr}`**.")

    with ep_col:
        st.markdown(f"**📈 Practice Progress Across `{epochs_ran}` Epochs**")
        st.caption("Watch how mistakes drop (or Accuracy improves) as the AI studies more rounds:")

        val_losses = results["history"]["val_loss"]
        best_epoch = int(np.argmin(val_losses)) + 1
        metric_k = "accuracy" if is_clf else "mae"
        ep_axis = range(1, epochs_ran + 1)

        fig_prog, ax_p = plt.subplots(figsize=(6, 3.3))
        ax_p.plot(ep_axis, results["history"][metric_k], label=f"Training {metric_k.upper()}", color="#2563eb", linewidth=2)
        ax_p.plot(ep_axis, results["history"][f"val_{metric_k}"], label=f"Validation {metric_k.upper()} (Unseen)", color="#f97316", linewidth=2)
        ax_p.axvline(best_epoch, color="#1e293b", linestyle="--", linewidth=1.2, label=f"Best Stop (#{best_epoch})")
        ax_p.set_xlabel("Epoch (Study Round)")
        ax_p.set_ylabel("Accuracy (Higher = Better)" if is_clf else "MAE Error (Lower = Better)")
        ax_p.legend(fontsize=8)
        ax_p.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_prog)
        plt.close(fig_prog)

        if use_es and epochs_ran < epochs:
            st.caption(f"🛑 Stopped early at **Epoch {epochs_ran}** (out of {epochs}) and kept the best model from **Epoch #{best_epoch}**.")

    st.divider()

    # =====================================================
    # PART 4: BUSINESS RECAP (LECTURE 7)
    # =====================================================
    st.subheader("4️⃣ Key Takeaways & Other Business Applications (Lecture 7)")
    recap_df = pd.DataFrame([
        ["Customer & Business Tables (This Demo)", "MLP (Feedforward Neural Network)", "Customer Churn, Cashback/Sales Forecasting, Credit Scoring"],
        ["Images & Visual Inspection", "CNN (Convolutional Neural Network)", "Product Defect Detection, Medical Scans, Face ID"],
        ["Text, Chat & Time Series", "RNN / LSTM / Transformer", "Customer Service Chatbots, Sentiment Analysis, Demand Forecasting"],
    ], columns=["Type of Business Data", "Deep Learning Model Used", "Real-World Business Examples"])
    st.dataframe(recap_df, hide_index=True)