# 📄 AI Resume Analyzer

A fully offline, AI-powered resume analyzer built with **TensorFlow**, **HuggingFace Transformers**, and **Streamlit** — no API keys, no subscriptions.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)

---

## ✨ Features

| Feature | Description |
|---|---|
| 📂 **Multi-format Upload** | Supports PDF, DOCX, DOC, and plain TXT |
| 🎯 **ATS Score (0–100)** | Weighted scoring across 5 dimensions |
| 🛠️ **Skill Detection** | 60+ skills across 5 categories |
| 🤖 **AI Summary** | Abstractive summary using DistilBART |
| 🎯 **Job Role Matching** | Match against 7 popular tech roles |
| 📋 **Section Checklist** | Detects 7 resume sections |
| 💡 **Smart Suggestions** | Actionable, specific improvement tips |

---

## 🤖 AI Models (Free, HuggingFace, No API Key)

| Model | Task | Framework |
|---|---|---|
| `cross-encoder/nli-MiniLM2-L6-H768` | Zero-shot classification | TensorFlow |
| `sshleifer/distilbart-cnn-6-6` | Abstractive summarization | TensorFlow |

> Both models download automatically on first run (~300 MB total) and are cached locally.

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/ai-resume-analyzer.git
cd ai-resume-analyzer

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

Open your browser at **http://localhost:8501**

> ⚠️ First launch downloads the HuggingFace models (~300 MB). Subsequent launches are fast.

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. **Fork / push** this repo to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `main`, and set **Main file path** to `app.py`
4. Click **Deploy** — done! 🎉

> Streamlit Cloud gives you 1 GB RAM. The app uses ~600 MB, so it fits comfortably.

---

## 📁 Project Structure

```
ai-resume-analyzer/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── .streamlit/
    └── config.toml     # Optional Streamlit theme config
```

---

## 🧠 How It Works

```
Resume File / Text
       │
       ▼
Text Extraction (pdfplumber / python-docx)
       │
       ├──► Section Detection    (regex keyword matching)
       ├──► Skill Extraction     (pattern matching across 60+ skills)
       ├──► Contact Info Check   (email, phone, LinkedIn, GitHub)
       ├──► Job Role Matching    (role-specific skill gap analysis)
       ├──► AI Summary           (DistilBART summarization via TF)
       └──► ATS Score (0–100)    (weighted 5-dimension scoring)
                │
                ▼
      Streamlit Dashboard
```

---

## 🗺️ Roadmap

- [ ] Multi-language resume support
- [ ] Resume vs Job Description comparison (paste JD)
- [ ] Export analysis as PDF report
- [ ] Interview question generator based on resume gaps
- [ ] Batch resume ranking for recruiters

---

## 📦 Dependencies

```
streamlit       — Web UI framework
pdfplumber      — PDF parsing
python-docx     — Word document parsing
tensorflow-cpu  — TensorFlow (CPU build for cloud)
transformers    — HuggingFace model hub
tokenizers      — Fast tokenization
sentencepiece   — Tokenizer support
numpy           — Numerical operations
```

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">
Built with ❤️ using TensorFlow · HuggingFace · Streamlit
</div>
