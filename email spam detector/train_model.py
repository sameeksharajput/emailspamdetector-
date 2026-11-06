"""
Train the spam model and save the files that app.py loads.

Usage:
    python train_model.py --data spam.csv
    (optional) --label-col v1 --text-col v2

Creates:
    dataset/spam_model.pkl
    dataset/tfidf_vectorizer.pkl
"""
import argparse
import os
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

OUT_DIR = "dataset"

KNOWN_COLUMNS = [
    ("label", "text"),
    ("v1", "v2"),
    ("category", "message"),
    ("class", "text"),
    ("spam", "text"),
    ("label", "message"),
    ("target", "text"),
]

LABEL_MAP = {
    "spam": "spam", "1": "spam", "1.0": "spam", "true": "spam",
    "ham": "ham", "0": "ham", "0.0": "ham", "false": "ham",
    "not spam": "ham", "legitimate": "ham",
}


def read_csv_safely(path):
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not read the CSV file with utf-8, latin-1 or cp1252.")


def detect_columns(df, label_col, text_col):
    if label_col and text_col:
        return label_col, text_col
    lowered = {str(c).lower().strip(): c for c in df.columns}
    for lab, txt in KNOWN_COLUMNS:
        if lab in lowered and txt in lowered:
            return lowered[lab], lowered[txt]
    print("Could not auto-detect columns. Your CSV columns are:", list(df.columns))
    print("Re-run with --label-col <name> --text-col <name>")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to the CSV dataset")
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--text-col", default=None)
    args = parser.parse_args()

    raw = read_csv_safely(args.data)
    label_col, text_col = detect_columns(raw, args.label_col, args.text_col)
    print(f"Using label column '{label_col}' and text column '{text_col}'")

    df = pd.DataFrame({"label": raw[label_col], "text": raw[text_col]}).dropna()
    df["label"] = df["label"].astype(str).str.strip().str.lower().map(LABEL_MAP)
    df = df.dropna(subset=["label"]).copy()
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip().str.len() > 0]

    if df.empty:
        print("No valid rows found. Labels must be spam/ham or 1/0.")
        sys.exit(1)

    print(f"Loaded {len(df)} messages | spam ratio: {(df['label'] == 'spam').mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
    )

    tfidf = TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=1, max_df=0.95
    )
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    print(f"\nAccuracy: {accuracy_score(y_test, preds):.4f}\n")
    print(classification_report(y_test, preds, zero_division=0))
    print("Confusion matrix (rows = actual, cols = predicted; order: ham, spam):")
    print(confusion_matrix(y_test, preds, labels=["ham", "spam"]))

    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(OUT_DIR, "spam_model.pkl"))
    joblib.dump(tfidf, os.path.join(OUT_DIR, "tfidf_vectorizer.pkl"))
    print(f"\nSaved model and vectorizer to the '{OUT_DIR}' folder.")


if __name__ == "__main__":
    main()