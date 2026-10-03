"""
Loan Default Prediction - Naive Bayes & KNN  (Streamlit app)
Mirrors the workflow of Naive_Bayes.ipynb:
Load Data -> EDA -> Preprocessing -> Split (+SMOTE) -> Train -> Evaluate -> Predict
Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except Exception:
    HAS_SMOTE = False

st.set_page_config(page_title="Loan Default Prediction", page_icon="🏦", layout="wide")


# ----------------------------------------------------------------- data
@st.cache_data
def demo_data(n=950, seed=42):
    """Synthetic data with the same columns as the notebook's loan dataset."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "applicant_id": np.arange(1, n + 1),
        "age": rng.integers(21, 65, n),
        "annual_income": rng.normal(60000, 25000, n).clip(15000).round(),
        "loan_amount": rng.normal(180000, 90000, n).clip(10000).round(),
        "credit_score": rng.normal(680, 70, n).clip(300, 850).round(),
        "city": rng.choice(["Delhi", "Mumbai", "Bengaluru", "Pune", "Chennai"], n),
        "education": rng.choice(["Graduate", "Postgraduate", "High School"], n),
        "employment_type": rng.choice(["Salaried", "Self-Employed", "Business"], n),
    })
    score = (-0.012 * (df.credit_score - 680) + 0.000004 * (df.loan_amount - 180000)
             - 0.000012 * (df.annual_income - 60000) - 1.4 + rng.normal(0, 0.8, n))
    df["loan_status"] = np.where(score > 0, "Default", "Non-Default")
    return df


@st.cache_data
def load_csv(file):
    return pd.read_csv(file)


# ----------------------------------------------------------------- sidebar
st.sidebar.title("🏦 Loan Default App")
up = st.sidebar.file_uploader("Upload loan CSV", type="csv",
                              help="Needs a target column (e.g. loan_status).")
if up is not None:
    raw = load_csv(up)
    st.sidebar.success(f"Loaded {raw.shape[0]} rows")
else:
    raw = demo_data()
    st.sidebar.info("Using built-in demo data. Upload your own CSV to replace it.")

raw = raw.dropna().reset_index(drop=True)
default_t = raw.columns.get_loc("loan_status") if "loan_status" in raw.columns else len(raw.columns) - 1
target = st.sidebar.selectbox("Target column", raw.columns, index=default_t)
id_guess = [c for c in raw.columns if "id" == c.lower() or c.lower().endswith("_id")]
drop_cols = st.sidebar.multiselect("Columns to drop (IDs)", [c for c in raw.columns if c != target],
                                   default=[c for c in id_guess if c != target])

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Training settings")
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.33, 0.01)
seed = st.sidebar.number_input("Random state", 0, 9999, 42)
use_smote = st.sidebar.checkbox("Balance train set with SMOTE", value=HAS_SMOTE, disabled=not HAS_SMOTE)
k_neighbors = st.sidebar.slider("KNN: n_neighbors", 1, 25, 3)
var_smooth = st.sidebar.select_slider("GaussianNB var_smoothing",
                                      options=[1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6], value=1e-9)

if raw[target].nunique() < 2 or raw[target].nunique() > 10:
    st.error("Target must be a categorical column with 2–10 classes.")
    st.stop()


# ----------------------------------------------------------------- pipeline
@st.cache_data
def prepare(df, target, drop_cols, test_size, seed, use_smote):
    d = df.drop(columns=list(drop_cols))
    cat_cols = list(d.select_dtypes(include=["object", "string"]).columns.drop(target, errors="ignore"))
    num_cols = [c for c in d.columns if c not in cat_cols + [target]]
    X = pd.get_dummies(d.drop(columns=target), columns=cat_cols, dtype=int, drop_first=True)
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(d[target]), index=d.index)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
    n_before = len(y_tr)
    if use_smote:
        X_tr, y_tr = SMOTE(random_state=seed).fit_resample(X_tr, y_tr)
    scaler = StandardScaler().fit(X_tr)
    return dict(X=X, cat_cols=cat_cols, num_cols=num_cols, classes=list(le.classes_),
                X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te, n_before=n_before,
                Xs_tr=scaler.transform(X_tr), Xs_te=scaler.transform(X_te), scaler=scaler)


P = prepare(raw, target, tuple(drop_cols), test_size, int(seed), use_smote and HAS_SMOTE)
classes = P["classes"]

gnb = GaussianNB(var_smoothing=var_smooth).fit(P["X_tr"], P["y_tr"])
knn = KNeighborsClassifier(n_neighbors=k_neighbors).fit(P["Xs_tr"], P["y_tr"])
pred_nb = gnb.predict(P["X_te"])
pred_knn = knn.predict(P["Xs_te"])


def metrics(y_true, y_pred):
    avg = "binary" if len(classes) == 2 else "weighted"
    return {"Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
            "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
            "F1": f1_score(y_true, y_pred, average=avg, zero_division=0)}


def cm_plot(y_true, y_pred, title, cmap):
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap=cmap, cbar=False,
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------- header
st.title("🏦 Loan Default Prediction")
st.caption("Naïve Bayes & KNN  ·  Load Data → EDA → Preprocessing → Split → Train → Evaluate → Predict")

tabs = st.tabs(["📂 Data & EDA", "🛠️ Preprocessing", "🤖 Train & Evaluate",
                "⚖️ NB vs KNN", "🔮 Predict", "📉 Elbow (K-Means)"])

# ----------------------------------------------------------------- tab 1
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", raw.shape[0]); c2.metric("Columns", raw.shape[1])
    c3.metric("Classes", raw[target].nunique())
    st.subheader("Dataset preview")
    st.dataframe(raw.head(50), use_container_width=True)

    a, b = st.columns(2)
    with a:
        st.subheader("Target distribution")
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        sns.countplot(data=raw, x=target, ax=ax, palette="viridis", hue=target, legend=False)
        st.pyplot(fig)
    with b:
        st.subheader("Statistical summary")
        st.dataframe(raw.describe(), use_container_width=True)

    st.subheader("Explore a feature")
    feat = st.selectbox("Feature", [c for c in raw.columns if c != target])
    fig, ax = plt.subplots(figsize=(7, 3.2))
    if pd.api.types.is_numeric_dtype(raw[feat]):
        sns.histplot(data=raw, x=feat, hue=target, kde=True, ax=ax)
    else:
        sns.countplot(data=raw, x=feat, hue=target, ax=ax)
    st.pyplot(fig)

    num = raw.drop(columns=drop_cols).select_dtypes("number")
    if num.shape[1] > 1 and st.checkbox("Show correlation heatmap"):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# ----------------------------------------------------------------- tab 2
with tabs[1]:
    st.markdown("""**Why preprocessing?** GaussianNB and KNN work with numbers only, so categorical
columns are **one-hot encoded** (`drop_first=True`) and the target is **label-encoded**.
KNN additionally needs **standardised** features because it is distance-based.""")
    a, b, c = st.columns(3)
    a.metric("Numeric features", len(P["num_cols"])); b.metric("Categorical features", len(P["cat_cols"]))
    c.metric("Features after encoding", P["X"].shape[1])
    st.write("**Categorical columns:**", ", ".join(P["cat_cols"]) or "none")
    st.write("**Target mapping:**", {cl: i for i, cl in enumerate(classes)})
    st.subheader("Encoded feature matrix")
    st.dataframe(P["X"].head(10), use_container_width=True)

    st.subheader("Train / test split & class balance")
    before = pd.Series(P["y_tr"]).value_counts().sort_index()
    cnt = pd.DataFrame({"Train (used for fitting)": before,
                        "Test": P["y_te"].value_counts().sort_index()})
    cnt.index = [classes[i] for i in cnt.index]
    st.write(f"Train rows before balancing: **{P['n_before']}** → after: **{len(P['y_tr'])}**"
             + (" (SMOTE applied)" if use_smote and HAS_SMOTE else ""))
    st.bar_chart(cnt)
    if not HAS_SMOTE:
        st.warning("imbalanced-learn not installed; SMOTE disabled.")

# ----------------------------------------------------------------- tab 3
with tabs[2]:
    model_name = st.radio("Model", ["Gaussian Naïve Bayes", "KNN"], horizontal=True)
    y_pred = pred_nb if model_name.startswith("Gaussian") else pred_knn
    m = metrics(P["y_te"], y_pred)
    cols = st.columns(4)
    for col, (k, v) in zip(cols, m.items()):
        col.metric(k, f"{v:.3f}")
    a, b = st.columns(2)
    with a:
        st.pyplot(cm_plot(P["y_te"], y_pred, f"{model_name} – Confusion Matrix",
                          "Blues" if model_name == "KNN" else "Greens"))
    with b:
        st.text("Classification report")
        st.code(classification_report(P["y_te"], y_pred, target_names=[str(c) for c in classes],
                                      zero_division=0))
    st.subheader("Actual vs Predicted")
    cmp = pd.DataFrame({"Actual": [classes[i] for i in P["y_te"]],
                        "Predicted": [classes[i] for i in y_pred]}, index=P["y_te"].index)
    cmp["Correct"] = cmp.Actual == cmp.Predicted
    st.dataframe(cmp.head(25), use_container_width=True)

# ----------------------------------------------------------------- tab 4
with tabs[3]:
    res = pd.DataFrame({"Gaussian NB": metrics(P["y_te"], pred_nb),
                        f"KNN (k={k_neighbors})": metrics(P["y_te"], pred_knn)}).T
    st.dataframe(res.style.format("{:.3f}").highlight_max(axis=0, color="#c6efce"),
                 use_container_width=True)
    st.bar_chart(res.T)
    a, b = st.columns(2)
    a.pyplot(cm_plot(P["y_te"], pred_nb, "Naïve Bayes", "Greens"))
    b.pyplot(cm_plot(P["y_te"], pred_knn, "KNN", "Blues"))
    st.subheader("Accuracy vs K (KNN)")
    ks = range(1, 26)
    accs = [accuracy_score(P["y_te"], KNeighborsClassifier(n_neighbors=k).fit(P["Xs_tr"], P["y_tr"])
                           .predict(P["Xs_te"])) for k in ks]
    st.line_chart(pd.DataFrame({"Accuracy": accs}, index=list(ks)))

# ----------------------------------------------------------------- tab 5
with tabs[4]:
    st.subheader("Predict for a new applicant")
    base = raw.drop(columns=drop_cols + [target])
    with st.form("applicant"):
        vals, cols = {}, st.columns(3)
        for i, col in enumerate(base.columns):
            with cols[i % 3]:
                if col in P["cat_cols"]:
                    vals[col] = st.selectbox(col, sorted(base[col].unique()))
                else:
                    lo, hi, med = float(base[col].min()), float(base[col].max()), float(base[col].median())
                    vals[col] = st.number_input(col, value=med, min_value=lo * 0.5 if lo > 0 else lo * 2,
                                                max_value=hi * 2, step=max((hi - lo) / 100, 0.01))
        model_pick = st.radio("Model", ["Gaussian Naïve Bayes", "KNN", "Both"], horizontal=True)
        go = st.form_submit_button("🔮 Predict", type="primary")
    if go:
        row = pd.get_dummies(pd.DataFrame([vals]), columns=P["cat_cols"], dtype=int)
        row = row.reindex(columns=P["X"].columns, fill_value=0)
        out = st.columns(2)
        if model_pick in ("Gaussian Naïve Bayes", "Both"):
            p = gnb.predict_proba(row)[0]
            out[0].success(f"**Naïve Bayes → {classes[int(np.argmax(p))]}**")
            out[0].bar_chart(pd.Series(p, index=[str(c) for c in classes]))
        if model_pick in ("KNN", "Both"):
            p = knn.predict_proba(P["scaler"].transform(row))[0]
            out[1].success(f"**KNN → {classes[int(np.argmax(p))]}**")
            out[1].bar_chart(pd.Series(p, index=[str(c) for c in classes]))

    st.markdown("---")
    import pickle
    d1, d2 = st.columns(2)
    d1.download_button("💾 Download naive_bayes_model.pkl", pickle.dumps(gnb), "naive_bayes_model.pkl")
    d2.download_button("💾 Download knn_model.pkl", pickle.dumps(knn), "knn_model.pkl")

# ----------------------------------------------------------------- tab 6
with tabs[5]:
    st.markdown("Elbow method on the scaled training data to pick a good number of clusters.")
    kmax = st.slider("Max K", 5, 25, 19)
    if st.button("Run elbow analysis"):
        inertia = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(P["Xs_tr"]).inertia_
                   for k in range(1, kmax + 1)]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, kmax + 1), inertia, marker="o")
        ax.set_xlabel("Number of clusters (K)"); ax.set_ylabel("Inertia (WCSS)")
        ax.set_title("Elbow Method"); ax.grid(True, linestyle="--", alpha=.6)
        st.pyplot(fig)
