"""Streamlit frontend: python -m streamlit run app.py."""
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import streamlit as st
from preprocessing import clean_text, PREPROCESSING_VERSION

BASE = Path(__file__).resolve().parent
ARTIFACTS = BASE / "artifacts"
st.set_page_config(page_title="Email Spam Detection", page_icon="✉️", layout="wide")
st.markdown("<style>.block-container{max-width:1120px;padding-top:2.5rem} h1{letter-spacing:-1px} div[data-testid='stMetric']{padding:1rem;border:1px solid #d6dce5;border-radius:12px} .stButton>button{border-radius:8px}</style>", unsafe_allow_html=True)

st.title("Email Spam Detection")
st.write("Check a message, explore measured performance, and understand the classification pipeline.")

@st.cache_resource
def load_artifacts(signatures):
    with (ARTIFACTS / "spam_model.pkl").open("rb") as f:
        model = pickle.load(f)
    with (ARTIFACTS / "tfidf_vectorizer.pkl").open("rb") as f:
        vectorizer = pickle.load(f)
    report = json.loads((ARTIFACTS / "evaluation.json").read_text(encoding="utf-8"))
    if report["preprocessing_version"] != PREPROCESSING_VERSION or report["sklearn_version"] != sklearn.__version__:
        raise ValueError("Preprocessing or scikit-learn version changed. Retrain models.")
    if model.n_features_in_ != len(vectorizer.vocabulary_):
        raise ValueError("Model and vectorizer do not match. Retrain models.")
    return model, vectorizer, report

try:
    files = [ARTIFACTS / f for f in ["spam_model.pkl", "tfidf_vectorizer.pkl", "evaluation.json"]]
    model, vectorizer, report = load_artifacts(tuple((f.stat().st_mtime_ns, f.stat().st_size) for f in files))
except Exception as exc:
    st.error("Artifacts are missing or could not be loaded. Run `python train.py` in the project folder, then reload this page.")
    with st.expander("Details"):
        st.code(str(exc))
    st.stop()

check, performance, how = st.tabs(["Check Message", "Model Performance", "How It Works"])

def example(text):
    st.session_state["message"] = text

with check:
    st.subheader("Is this message spam?")
    st.caption("Paste English text. Predictions use Multinomial Naive Bayes.")
    a, b = st.columns(2)
    a.button("Try Spam Example", on_click=example, args=("Congratulations! You won a free cash prize. Claim your reward now! Call today for this exclusive offer.",), width="stretch")
    b.button("Try Normal Example", on_click=example, args=("Hi, can we meet tomorrow after class to discuss the project? Please bring your notes.",), width="stretch")
    message = st.text_area("Message", height=220, key="message", placeholder="Enter or paste your message here…")
    if st.button("Check Spam", type="primary", width="stretch"):
        if not message.strip():
            st.warning("Please enter a message before checking.")
        else:
            cleaned = clean_text(message)
            features = vectorizer.transform([cleaned])
            if features.nnz == 0:
                st.warning("No recognised features: no words remain in the model's vocabulary after preprocessing. Try a longer English message. A reliable classification cannot be shown.")
            else:
                prediction = int(model.predict(features)[0])
                if prediction == 1:
                    st.error("Spam — the model flags this message as spam.")
                else:
                    st.success("Not Spam — the model classifies this message as ham.")
                probabilities = dict(zip(model.classes_, model.predict_proba(features)[0]))
                a, b = st.columns(2)
                a.metric("Estimated Spam probability", f"{probabilities[1]:.1%}")
                b.metric("Estimated Not Spam probability", f"{probabilities[0]:.1%}")
                st.caption("Model estimates are uncalibrated and are not guarantees.")
            with st.expander("View original and cleaned text"):
                st.text("Original text")
                st.code(message, language=None)
                st.text("Cleaned text")
                st.code(cleaned or "(No tokens remain.)", language=None)

with performance:
    st.subheader("Measured model performance")
    audit = report["audit"]
    a, b, c = st.columns(3)
    a.metric("Original dataset", f"{audit['raw_rows']:,}")
    b.metric("Original ham messages", f"{audit['raw_class_counts']['ham']:,}")
    c.metric("Original spam messages", f"{audit['raw_class_counts']['spam']:,}")
    st.caption(f"After cleaning and deduplication: {audit['usable_rows']:,} messages — {audit['class_counts']['ham']:,} ham and {audit['class_counts']['spam']:,} spam.")
    st.bar_chart(pd.DataFrame({"Messages": audit["class_counts"]}), color="#3b82f6")
    split = report["split"]
    st.info(f"Same stratified 80/20 held-out split for all models: {split['train_rows']:,} training messages and {split['test_rows']:,} test messages; random_state=42. Duplicate texts are removed before splitting. TF-IDF learns only from training messages.")
    scores = pd.DataFrame({name: {k: v for k, v in result.items() if k != "confusion_matrix"} for name, result in report["models"].items()}).T
    scores.columns = ["Accuracy", "Precision", "Recall", "F1 score"]
    st.dataframe(scores.style.format("{:.2%}"), width="stretch")
    st.caption("Precision, recall, and F1 use spam as the positive class. Accuracy covers both classes; class imbalance makes accuracy alone insufficient.")
    st.subheader("Confusion matrices")
    for column, (name, result) in zip(st.columns(3), report["models"].items()):
        with column:
            fig, ax = plt.subplots(figsize=(4, 3.5))
            matrix = result["confusion_matrix"]
            ax.imshow(matrix, cmap="Blues")
            for row in range(2):
                for col in range(2):
                    ax.text(col, row, str(matrix[row][col]), ha="center", va="center", color="white" if matrix[row][col] > max(map(max, matrix))/2 else "black")
            ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Not Spam", "Spam"], yticklabels=["Not Spam", "Spam"], xlabel="Predicted label", ylabel="Actual label", title=name)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    with st.expander("Dataset audit and reproducibility"):
        st.json(audit)
        st.caption("Duplicates use exact message text after trimming outer whitespace. All texts with conflicting labels are excluded.")
        st.write(f"Vocabulary: {report['vocabulary_size']:,} features. Trained: {report['created_at']}.")
        st.caption(f"Dataset SHA-256: {report['dataset_sha256']}")

with how:
    st.subheader("From message to prediction")
    st.info("Message → Preprocessing → TF-IDF → Classifier → Spam / Not Spam")
    st.markdown("""
1. **Cleaning:** Convert text to lowercase and extract letters and numbers, removing punctuation.
2. **Tokenisation:** Break the message into individual word and number tokens.
3. **Stopword removal:** Remove common English words such as “the” and “and” using a fixed stopword list.
4. **Stemming:** Shorten related words to a shared root: “winning” and “wins” become “win”.
5. **TF-IDF:** Turn words into numerical features, weighting distinctive words more heavily. Up to 3,000 features are learned from training messages only.
6. **Training:** Learn from labelled spam and ham messages. Compare Naive Bayes, Logistic Regression, and a linear support vector machine on the same unseen test messages.
7. **Prediction:** Apply the same preprocessing and saved vectorizer to new text. The saved Multinomial Naive Bayes model predicts Spam or Not Spam and supplies estimated probabilities.
""")
    
