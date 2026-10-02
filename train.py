"""Explicit training entry point; never run by app startup."""
import hashlib
import json
import pickle
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from preprocessing import clean_text, PREPROCESSING_VERSION

BASE = Path(__file__).resolve().parent

def train():
    source = BASE / "email_data.csv"
    raw = pd.read_csv(source)
    if not {"Category", "Message"}.issubset(raw.columns):
        raise ValueError("CSV must contain Category and Message columns.")
    audit = {"raw_rows": len(raw), "missing_values": {c: int(raw[c].isna().sum()) for c in ["Category", "Message"]}, "raw_class_counts": {k: int(v) for k, v in raw.Category.value_counts().items()}}
    data = raw.dropna(subset=["Category", "Message"]).copy()
    data["Category"] = data.Category.astype(str).str.strip().str.lower()
    data["Message"] = data.Message.astype(str).str.strip()
    audit["blank_messages"] = int(data.Message.eq("").sum())
    data = data[data.Message.ne("")]
    if set(data.Category) - {"ham", "spam"}:
        raise ValueError("Labels must be ham or spam.")
    audit["duplicate_rows"] = int(data.duplicated(["Category", "Message"]).sum())
    audit["duplicate_message_texts"] = int(data.Message.duplicated().sum())
    groups = data.groupby("Message").Category.nunique()
    conflicts = groups[groups > 1].index
    audit["conflicting_message_texts"] = len(conflicts)
    audit["conflicting_rows_removed"] = int(data.Message.isin(conflicts).sum())
    data = data[~data.Message.isin(conflicts)].drop_duplicates("Message")
    audit["usable_rows"] = len(data)
    audit["class_counts"] = {k: int(v) for k, v in data.Category.value_counts().items()}
    train_text, test_text, y_train, y_test = train_test_split(data.Message, data.Category.map({"ham": 0, "spam": 1}), test_size=0.2, stratify=data.Category, random_state=42)
    vectorizer = TfidfVectorizer(max_features=3000)
    x_train = vectorizer.fit_transform(train_text.map(clean_text))
    x_test = vectorizer.transform(test_text.map(clean_text))
    models = {"Multinomial Naive Bayes": MultinomialNB(), "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42), "Linear SVM": LinearSVC(max_iter=10000, random_state=42)}
    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        results[name] = {"accuracy": float(accuracy_score(y_test, predicted)), "precision": float(precision_score(y_test, predicted, zero_division=0)), "recall": float(recall_score(y_test, predicted, zero_division=0)), "f1": float(f1_score(y_test, predicted, zero_division=0)), "confusion_matrix": confusion_matrix(y_test, predicted, labels=[0, 1]).tolist()}
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "sklearn_version": sklearn.__version__, "preprocessing_version": PREPROCESSING_VERSION, "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "audit": audit, "split": {"train_rows": len(train_text), "test_rows": len(test_text), "test_size": 0.2, "random_state": 42, "stratified": True}, "vocabulary_size": len(vectorizer.vocabulary_), "models": results}
    dest = BASE / "artifacts"
    dest.mkdir(exist_ok=True)
    for filename, obj in [("spam_model.pkl", models["Multinomial Naive Bayes"]), ("tfidf_vectorizer.pkl", vectorizer)]:
        with (dest / filename).open("wb") as handle:
            pickle.dump(obj, handle)
    (dest / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    train()
