# Email Spam Detection — DWM project

Three Streamlit tabs cover message prediction, actual held-out evaluation, and a simple pipeline explanation.

## Windows setup and run (PowerShell)

Use Python 3.13. These commands do not require virtual-environment activation:

```powershell
Set-Location 'D:\PRAPTI\PROJECTS\email-spam-detection'
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the URL printed by Streamlit, usually http://localhost:8501. Stop with Ctrl+C. On later runs, only the final command is needed. Retrain after changing the dataset, preprocessing, or scikit-learn version.

## Project files

- `app.py`: three-tab frontend with file-relative artifact paths and helpful missing-artifact errors.
- `train.py`: separate training entry point, never invoked by app startup.
- `preprocessing.py`: shared lowercase, regex tokenisation, fixed English stopwords, and Porter stemming. No NLTK resource downloads are needed.
- `artifacts/spam_model.pkl` and `artifacts/tfidf_vectorizer.pkl`: matching final Multinomial Naive Bayes model and vectorizer.
- `artifacts/evaluation.json`: measured scores, labelled confusion matrices, dataset audit, split details, versions, and dataset hash.

The original notebook and root-level pickle files are preserved. New artifacts use a separate directory. Only load pickle files from trusted sources.

## Evaluation method

The script reports missing values, blank messages, duplicate rows, duplicate message texts, and conflicting labels. It excludes missing/blank rows, removes all texts with conflicting labels, and deduplicates exact message texts after trimming outer whitespace.

Raw messages are split first using an 80/20 stratified split and `random_state=42`. TF-IDF with `max_features=3000` is fitted only on training messages. Multinomial Naive Bayes, Logistic Regression, and LinearSVC are trained and evaluated on that same split. Precision, recall, and F1 use spam as the positive class. The app uses identical preprocessing for prediction and reports uncalibrated probability estimates.

The notebook fitted TF-IDF before splitting and its sample inference skipped preprocessing; the separate training script corrects both. The dataset contains short messages, many resembling SMS, so performance on full emails may differ. This English classifier does not connect to inboxes or inspect attachments. No login or database is used.

## Verification

Run `python verify_app.py` after training to check all three tabs render, example predictions, empty input, unknown vocabulary, missing-artifact handling, and file-relative paths.
