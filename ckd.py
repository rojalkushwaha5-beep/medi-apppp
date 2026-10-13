import streamlit as st
import pandas as pd
import os
from pathlib import Path
from urllib.request import urlopen, Request
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="CKD Prediction Demo", page_icon="🩺", layout="wide")
st.title("🩺 Chronic Kidney Disease (CKD) Prediction Demo")
st.warning(
    "Educational project only. This model is not a medical device and cannot diagnose CKD. "
    "Do not use its result for medical decisions."
)

DATA_FILE = Path(__file__).with_name("kidney.csv")
DATA_URLS = [
    "https://raw.githubusercontent.com/vamshikrishnajr/chronic-kidney-disease-diagnosis/master/kidney_disease.csv",
    "https://raw.githubusercontent.com/ArjunAnil2000/Chronic-Kidney-Disease-dataset/main/final.csv",
]

@st.cache_data(show_spinner=False)
def load_dataset():
    # Prefer the local CSV if the user has already downloaded or added it.
    if DATA_FILE.exists() and DATA_FILE.stat().st_size > 100:
        try:
            return pd.read_csv(DATA_FILE)
        except Exception:
            pass

    errors = []
    for url in DATA_URLS:
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=20) as response:
                raw = response.read()
            df = pd.read_csv(BytesIO(raw))
            if df.empty or len(df.columns) < 2:
                raise ValueError("The downloaded file does not look like a CSV dataset.")
            # Save a local copy beside app.py so next startup can use it offline.
            df.to_csv(DATA_FILE, index=False)
            return df
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError(
        "Could not download the CKD dataset. Check your internet connection, or put "
        "'kidney.csv' in the same folder as app.py. Details: " + " | ".join(errors)
    )

try:
    df = load_dataset()
except Exception as exc:
    st.error(str(exc))
    st.info(
        "To add the CSV manually: download a CKD dataset CSV from the linked public "
        "repository, rename it to kidney.csv, and place it beside app.py."
    )
    st.markdown(
        "[Open CKD dataset repository](https://github.com/vamshikrishnajr/chronic-kidney-disease-diagnosis)"
    )
    st.stop()

# Normalize column names and text values.
df.columns = [str(c).strip().lower() for c in df.columns]
df = df.loc[:, ~df.columns.str.match(r"^unnamed")]
df = df.replace(r"^\s*$", pd.NA, regex=True)
df = df.replace(r"^\s*\?\s*$", pd.NA, regex=True)

target_candidates = ["classification", "class", "target", "ckd"]
target = next((c for c in target_candidates if c in df.columns), None)
if target is None:
    st.error("Could not find a target column. Expected one of: classification, class, target.")
    st.write("Columns found:", df.columns.tolist())
    st.stop()

# Remove ID-like columns from model inputs.
X = df.drop(columns=[target])
X = X.drop(columns=[c for c in X.columns if c in ["id", "unnamed: 0"]], errors="ignore")
y = df[target].astype("string").str.strip().str.lower()
y = y.str.replace(r"\s+", "", regex=True).str.replace(".", "", regex=False)

# Drop rows without known labels and normalize common CKD labels.
valid = y.notna() & (y != "")
X, y = X.loc[valid].copy(), y.loc[valid].copy()
label_map = {
    "ckd": "CKD",
    "yes": "CKD",
    "positive": "CKD",
    "1": "CKD",
    "notckd": "Not CKD",
    "no": "Not CKD",
    "negative": "Not CKD",
    "0": "Not CKD",
}
y = y.map(lambda value: label_map.get(value, value))
if y.nunique() < 2:
    st.error("The target column must contain at least two classes. Please check the CSV labels.")
    st.write("Labels found:", y.unique().tolist())
    st.stop()

# Convert columns that are mostly numeric into numeric values.
for col in X.columns:
    converted = pd.to_numeric(X[col], errors="coerce")
    if converted.notna().sum() >= max(1, int(0.7 * X[col].notna().sum())):
        X[col] = converted

numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocess = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols),
])
model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
])

if len(X) < 10:
    st.error("The CSV has too few usable rows to train this demo.")
    st.stop()

stratify = y if y.value_counts().min() >= 2 else None
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
except Exception as exc:
    st.error(f"Could not train the model: {exc}")
    st.stop()

st.success(f"Dataset loaded: {len(df)} rows | {len(X.columns)} input features")
st.metric("Demo test-set accuracy", f"{accuracy:.1%}")
st.caption("Accuracy is only a demonstration metric on a small public dataset; it does not establish clinical reliability.")

st.subheader("Dataset preview")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("Try a sample prediction")
st.write("Enter values for the features. Missing fields use the dataset's typical value.")
user_values = {}
with st.form("prediction_form"):
    left, right = st.columns(2)
    for i, col in enumerate(X.columns):
        container = left if i % 2 == 0 else right
        with container:
            if col in numeric_cols:
                vals = pd.to_numeric(X[col], errors="coerce").dropna()
                default = float(vals.median()) if not vals.empty else 0.0
                user_values[col] = st.number_input(
                    col, value=default, key=f"num_{col}"
                )
            else:
                choices = sorted(X[col].dropna().astype(str).unique().tolist())
                if not choices:
                    choices = ["unknown"]
                user_values[col] = st.selectbox(col, choices, key=f"cat_{col}")
    submitted = st.form_submit_button("Run demo prediction")

if submitted:
    sample = pd.DataFrame([user_values], columns=X.columns)
    prediction = model.predict(sample)[0]
    st.subheader(f"Model output: {prediction}")
    st.info("This is a machine-learning demonstration, not a diagnosis. Consult a qualified clinician for medical concerns.")
