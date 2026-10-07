# 🛡️ Spam-Ham Email Detector

A **production-style** email spam classification system powered by **DistilBERT**,
with SHAP explainability, a Streamlit UI, and a full audit trail via SQLite.

---

## 🚀 Features

| Feature | Details |
|---|---|
| 🤖 DistilBERT Classification | Fine-tuned `distilbert-base-uncased` for binary spam/ham detection |
| 🔍 Live Prediction | Instant SPAM/HAM verdict with confidence score |
| 📂 Batch Upload | Classify thousands of emails from a CSV in seconds |
| 💡 SHAP Explainability | Token-level heatmap showing *why* an email is spam |
| 📬 Gmail Scanner | Connect live Gmail inbox via IMAP and classify incoming emails |
| 🔐 Authentication Portal | Real-world login & sign-up portal with PBKDF2 password hashing & session guards |
| 📊 Dashboard | Model metrics, ROC curve, confusion matrix, prediction history |
| 🗄️ Audit Log | Every prediction persisted to SQLite with metadata |
| 🐳 Docker-ready | One-command deployment |

---

## 📁 Folder Structure

```
spam-ham-detector/
├── data/
│   ├── raw/                  # Raw email CSVs
│   ├── processed/            # Cleaned datasets
│   └── predictions.db        # SQLite prediction log (auto-created)
├── models/
│   └── distilbert_spam/      # Fine-tuned model weights + tokenizer
├── spam_detector/            # Core Python package
│   ├── config.py             # Config loader (config.yaml + .env)
│   ├── preprocess.py         # Email cleaning pipeline
│   ├── dataset.py            # PyTorch Dataset + data splitting
│   ├── train.py              # DistilBERT fine-tuning script
│   ├── predict.py            # Inference engine
│   ├── explain.py            # SHAP explainability
│   ├── evaluate.py           # Metrics + Plotly charts
│   ├── db.py                 # SQLite prediction logger
│   ├── gmail.py              # Gmail IMAP fetcher & classifier
│   └── auth.py               # Authentication & PBKDF2 password security
├── app/
│   ├── main.py               # Streamlit home page
│   └── pages/
│       ├── 1_live_predict.py
│       ├── 2_batch_upload.py
│       ├── 3_dashboard.py
│       ├── 4_explainability.py
│       └── 5_gmail_scanner.py
├── tests/                    # pytest unit tests
├── config.yaml               # Central configuration
├── .env.example              # Environment variable template
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.11+
- pip
- (Optional) CUDA GPU for faster training

### 1. Clone the project
```bash
cd D:\spam-ham-detector
```

### 2. Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
copy .env.example .env
# Edit .env if needed (HF_TOKEN is optional for public models)
```

---

## 📥 Download Dataset

The project works with **UCI SMS Spam Collection** (fast, ~5500 emails):

```bash
# Download directly
curl -L "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip" -o data/raw/sms.zip
cd data/raw && tar -xf sms.zip
# Rename: SMSSpamCollection -> spam.csv (tab-separated, v1=label, v2=text)
```

Or use any CSV with columns `text` (email body) and `label` (0=HAM, 1=SPAM).

---

## 🏋️ Train the Model

```bash
python -m spam_detector.train --csv data/raw/spam.csv
```

Training takes ~5 minutes on GPU, ~30–60 minutes on CPU.  
The best model is auto-saved to `models/distilbert_spam/`.

---

## 🌐 Run the Streamlit App

```bash
streamlit run app/main.py
```

Open your browser at: **http://localhost:8501**

---

## 🐳 Run with Docker

```bash
# Build and start
docker compose up --build

# Stop
docker compose down
```

App is available at **http://localhost:8501**.

---

## 🧪 Run Tests

```bash
# All tests (skips model tests if model not trained)
pytest tests/ -v --tb=short

# With coverage report
pytest tests/ --cov=spam_detector --cov-report=term-missing
```

---

## 📊 Evaluation Metrics (Expected on SMS Dataset)

| Metric | Expected Value |
|---|---|
| Accuracy | ~99% |
| Precision (SPAM) | ~98–99% |
| Recall (SPAM) | ~96–98% |
| F1 Score | ~97–99% |
| AUC-ROC | ~0.999 |

---

## 🎓 Viva / Demo Q&A

**Q1: Why DistilBERT over Naive Bayes?**  
A: DistilBERT understands context — "free offer" in a work email vs. spam email is treated differently. Naive Bayes is bag-of-words and ignores word order. DistilBERT achieves ~99% F1 vs ~97% for NB on this dataset.

**Q2: What is SHAP and why use it?**  
A: SHAP (SHapley Additive exPlanations) assigns each word a contribution value based on game theory. It tells us *why* the model predicted SPAM — making the system auditable and trustworthy.

**Q3: How does the preprocessing pipeline work?**  
A: It strips HTML (BeautifulSoup), removes URLs/email headers, normalises unicode, lowercases, and collapses whitespace — reducing noise and preventing the model from over-fitting on structural artefacts.

**Q4: Why truncate to 256 tokens?**  
A: DistilBERT's max is 512. Most spam signals are in the first 256 tokens (subject + opening lines). Halving the length doubles batch size and reduces training time with minimal accuracy loss.

**Q5: How is class imbalance handled?**  
A: The dataset is ~87% HAM / 13% SPAM. Stratified splitting preserves this ratio in train/val/test. We report macro-F1 (not accuracy) to catch underperformance on the minority class.

**Q6: What if someone submits adversarial spam (obfuscated text)?**  
A: DistilBERT handles character-level obfuscation better than TF-IDF because subword tokenisation (WordPiece) can still recognise "fr3e m0ney". Perfect robustness would require adversarial training.

**Q7: Why SQLite and not Postgres?**  
A: SQLite requires zero configuration — ideal for an MVP. The `db.py` module abstracts the connection string, so swapping to Postgres is a one-line change.

**Q8: How reproducible is the training?**  
A: `set_seed(42)` is called before all random operations (model init, data splitting, training). The seed is stored in `config.yaml` and applied consistently.

**Q9: What are the system's limitations?**  
A: (1) Multilingual spam is not handled — model is English-only. (2) Image-based spam (PDF attachments) is ignored. (3) SHAP runs on CPU and takes 10–30s per email.

**Q10: How would you deploy this to production?**  
A: (1) Replace Streamlit with a FastAPI REST endpoint for integrations. (2) Move SQLite to Postgres with connection pooling. (3) Set up MLflow for model versioning. (4) Add a retraining pipeline triggered when drift is detected.

---

## 🛠️ Configuration

All tuneable parameters live in `config.yaml` — no magic numbers in code:

- `training.epochs`, `training.batch_size`, `training.learning_rate`
- `model.max_length` — token truncation limit
- `shap.max_evals` — SHAP speed vs. accuracy
- `streamlit.confidence_threshold` — "Uncertain" badge threshold

---

## 📄 License

MIT License — free for academic, portfolio, and commercial use.


