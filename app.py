import streamlit as st
import pdfplumber
import docx
import re
import numpy as np
from pathlib import Path

# ============================================
# CRITICAL: Force PyTorch Only - Disable TensorFlow
# ============================================
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # Disable TensorFlow in Transformers
os.environ["USE_TF"] = "0"              # Force no TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF warnings

# Page config
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

.hero-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}

.hero-sub {
    color: #8892b0;
    font-size: 1.05rem;
    font-weight: 400;
    letter-spacing: 0.02em;
    margin-bottom: 2rem;
}

.score-card {
    background: linear-gradient(145deg, #1e2140, #252a4a);
    border: 1px solid rgba(102, 126, 234, 0.3);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15);
}

.score-number {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 5rem;
    font-weight: 700;
    line-height: 1;
}

.score-label {
    color: #8892b0;
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

.section-card {
    background: linear-gradient(145deg, #1e2140, #252a4a);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: #ccd6f6;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.skill-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.5rem;
}

.skill-pill {
    background: linear-gradient(135deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2));
    border: 1px solid rgba(102,126,234,0.4);
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-size: 0.82rem;
    font-weight: 500;
    color: #a8b4f8;
    display: inline-block;
}

.skill-pill-found {
    background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(16,185,129,0.15));
    border: 1px solid rgba(34,197,94,0.35);
    color: #6ee7b7;
}

.skill-pill-missing {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(220,38,38,0.12));
    border: 1px solid rgba(239,68,68,0.3);
    color: #fca5a5;
}

.metric-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.metric-row:last-child { border-bottom: none; }
.metric-key { color: #8892b0; font-size: 0.9rem; }
.metric-val { color: #ccd6f6; font-weight: 600; font-size: 0.95rem; }

.progress-wrap { margin: 0.5rem 0; }
.progress-label {
    display: flex; justify-content: space-between;
    color: #8892b0; font-size: 0.8rem; margin-bottom: 0.3rem;
}
.progress-bar-bg {
    background: rgba(255,255,255,0.07);
    border-radius: 99px; height: 8px; overflow: hidden;
}
.progress-bar-fill {
    height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, #667eea, #764ba2);
    transition: width 0.5s ease;
}
.progress-bar-fill-green {
    background: linear-gradient(90deg, #10b981, #34d399);
}
.progress-bar-fill-yellow {
    background: linear-gradient(90deg, #f59e0b, #fbbf24);
}
.progress-bar-fill-red {
    background: linear-gradient(90deg, #ef4444, #f87171);
}

.suggestion-item {
    display: flex; gap: 0.75rem; align-items: flex-start;
    padding: 0.6rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    color: #a8b4c8; font-size: 0.9rem; line-height: 1.5;
}
.suggestion-item:last-child { border-bottom: none; }
.suggestion-icon { font-size: 1rem; flex-shrink: 0; margin-top: 2px; }

.upload-hint {
    color: #8892b0; font-size: 0.88rem; text-align: center;
    padding: 0.5rem 0;
}

.sidebar-label {
    color: #8892b0; font-size: 0.78rem;
    text-transform: uppercase; letter-spacing: 0.1em;
    font-weight: 600; margin-bottom: 0.4rem;
}

.badge {
    display: inline-block; border-radius: 99px;
    padding: 0.2rem 0.7rem; font-size: 0.75rem; font-weight: 600;
}
.badge-green { background: rgba(34,197,94,0.15); color: #6ee7b7; border: 1px solid rgba(34,197,94,0.3); }
.badge-yellow { background: rgba(245,158,11,0.15); color: #fcd34d; border: 1px solid rgba(245,158,11,0.3); }
.badge-red { background: rgba(239,68,68,0.12); color: #fca5a5; border: 1px solid rgba(239,68,68,0.3); }

.custom-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(102,126,234,0.4), transparent);
    margin: 1.5rem 0;
}

.stFileUploader > div > div {
    background: rgba(255,255,255,0.03) !important;
    border: 2px dashed rgba(102,126,234,0.4) !important;
    border-radius: 16px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(102,126,234,0.4) !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================
# FIXED: Lazy Model Loader (PyTorch Only)
# ============================================
@st.cache_resource(show_spinner=False)
def load_models():
    """Load NLP models using PyTorch only - NO TENSORFLOW"""
    try:
        from transformers import pipeline
        
        # Force PyTorch by setting device
        device = 0 if __import__('torch').cuda.is_available() else -1
        
        # Zero-shot classifier for section detection & skill matching
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            framework="pt",
            device=device
        )
        
        # Summarizer for resume summary section
        summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-6-6",
            framework="pt",
            device=device,
            max_length=120,
            min_length=30,
            truncation=True
        )
        
        return classifier, summarizer
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Falling back to keyword-based analysis...")
        return None, None

# ============================================
# TEXT EXTRACTION HELPERS
# ============================================

def extract_text_pdf(file) -> str:
    text = []
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text.append(t)
    except Exception as e:
        st.warning(f"PDF extraction warning: {str(e)}")
    return "\n".join(text)

def extract_text_docx(file) -> str:
    try:
        doc = docx.Document(file)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        st.warning(f"DOCX extraction warning: {str(e)}")
        return ""

def extract_text(file) -> str:
    name = file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_pdf(file)
    elif name.endswith((".docx", ".doc")):
        return extract_text_docx(file)
    else:
        try:
            return file.read().decode("utf-8", errors="ignore")
        except:
            return ""

# ============================================
# SKILL DATABASES
# ============================================

SKILL_CATEGORIES = {
    "Programming Languages": [
        "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go",
        "Rust", "Ruby", "PHP", "Swift", "Kotlin", "Scala", "R", "MATLAB",
    ],
    "Web & Frameworks": [
        "React", "Angular", "Vue", "Node.js", "Django", "Flask", "FastAPI",
        "Spring", "Laravel", "Express", "Next.js", "Svelte",
    ],
    "Data & AI/ML": [
        "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "Pandas", "NumPy",
        "Spark", "Hadoop", "Tableau", "Power BI", "SQL", "MongoDB", "Redis",
        "Elasticsearch", "Kafka", "Airflow",
    ],
    "Cloud & DevOps": [
        "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "CI/CD",
        "Jenkins", "GitHub Actions", "Linux", "Bash", "Ansible",
    ],
    "Soft Skills": [
        "Leadership", "Communication", "Teamwork", "Problem Solving",
        "Project Management", "Agile", "Scrum", "Presentation", "Mentoring",
    ],
}

SECTION_KEYWORDS = {
    "Education": ["education", "degree", "university", "college", "bachelor",
                  "master", "phd", "gpa", "coursework", "graduated"],
    "Experience": ["experience", "work history", "employment", "position",
                   "role", "internship", "job", "career"],
    "Skills": ["skills", "technologies", "tools", "competencies", "expertise",
               "proficiencies", "technical"],
    "Projects": ["projects", "portfolio", "built", "developed", "created",
                 "implemented", "launched"],
    "Certifications": ["certification", "certified", "certificate", "license",
                       "credential", "aws certified", "google certified"],
    "Summary": ["summary", "objective", "profile", "about me", "overview"],
}

JOB_ROLE_SKILLS = {
    "Data Scientist": ["Python", "R", "SQL", "TensorFlow", "PyTorch", "Pandas",
                       "NumPy", "Scikit-learn", "Statistics", "Machine Learning",
                       "Deep Learning", "Tableau", "Spark"],
    "Full Stack Developer": ["JavaScript", "React", "Node.js", "Python", "SQL",
                             "MongoDB", "Docker", "AWS", "Git", "REST API",
                             "TypeScript", "CSS", "HTML"],
    "DevOps Engineer": ["Docker", "Kubernetes", "AWS", "Azure", "Terraform",
                        "CI/CD", "Linux", "Bash", "Jenkins", "Ansible", "Git",
                        "Monitoring", "Security"],
    "ML Engineer": ["Python", "TensorFlow", "PyTorch", "Kubernetes", "Docker",
                    "AWS", "MLflow", "Spark", "SQL", "Airflow", "Pandas",
                    "Scikit-learn", "C++"],
    "Backend Developer": ["Python", "Java", "Go", "SQL", "PostgreSQL", "Redis",
                          "Docker", "AWS", "REST API", "Microservices",
                          "Kafka", "gRPC", "Linux"],
    "Frontend Developer": ["JavaScript", "TypeScript", "React", "Vue", "Angular",
                           "CSS", "HTML", "Next.js", "Webpack", "Testing",
                           "Figma", "Performance"],
    "Cloud Architect": ["AWS", "Azure", "GCP", "Terraform", "Kubernetes",
                        "Docker", "Security", "Networking", "Cost Optimization",
                        "Serverless", "Microservices"],
    "General / Other": [],
}

# ============================================
# ANALYSIS FUNCTIONS
# ============================================

def detect_sections(text: str) -> dict:
    text_lower = text.lower()
    found = {}
    for section, keywords in SECTION_KEYWORDS.items():
        found[section] = any(kw in text_lower for kw in keywords)
    return found

def extract_skills(text: str) -> dict:
    text_lower = text.lower()
    result = {}
    for category, skills in SKILL_CATEGORIES.items():
        found = [s for s in skills if re.search(r'\b' + re.escape(s.lower()) + r'\b', text_lower)]
        result[category] = found
    return result

def match_job_skills(resume_text: str, role: str) -> dict:
    required = JOB_ROLE_SKILLS.get(role, [])
    if not required:
        return {"required": [], "found": [], "missing": [], "match_pct": 100}
    text_lower = resume_text.lower()
    found = [s for s in required if re.search(r'\b' + re.escape(s.lower()) + r'\b', text_lower)]
    missing = [s for s in required if s not in found]
    pct = round(len(found) / len(required) * 100) if required else 100
    return {"required": required, "found": found, "missing": missing, "match_pct": pct}

def count_words(text: str) -> int:
    return len(text.split())

def detect_email(text: str) -> bool:
    return bool(re.search(r'[\w.+-]+@[\w-]+\.\w+', text))

def detect_phone(text: str) -> bool:
    return bool(re.search(r'(\+?\d[\d\s\-().]{7,}\d)', text))

def detect_linkedin(text: str) -> bool:
    return bool(re.search(r'linkedin\.com', text, re.I))

def detect_github(text: str) -> bool:
    return bool(re.search(r'github\.com', text, re.I))

def score_resume(text: str, sections: dict, skills: dict, job_match: dict) -> dict:
    score = 0
    breakdown = {}
    
    wc = count_words(text)
    if wc >= 400:
        length_score = 20
    elif wc >= 200:
        length_score = 12
    else:
        length_score = 5
    breakdown["Length & Content"] = (length_score, 20)
    score += length_score
    
    key_sections = ["Education", "Experience", "Skills", "Projects"]
    present = sum(1 for s in key_sections if sections.get(s))
    section_score = int(present / len(key_sections) * 20)
    breakdown["Sections Coverage"] = (section_score, 20)
    score += section_score
    
    total_skills = sum(len(v) for v in skills.values())
    if total_skills >= 15:
        skill_score = 20
    elif total_skills >= 8:
        skill_score = 13
    elif total_skills >= 3:
        skill_score = 7
    else:
        skill_score = 2
    breakdown["Skills Breadth"] = (skill_score, 20)
    score += skill_score
    
    contact_score = 0
    if detect_email(text): contact_score += 7
    if detect_phone(text): contact_score += 5
    if detect_linkedin(text): contact_score += 4
    if detect_github(text): contact_score += 4
    breakdown["Contact & Links"] = (contact_score, 20)
    score += contact_score
    
    if job_match["required"]:
        match_score = int(job_match["match_pct"] / 100 * 20)
    else:
        match_score = 15
    breakdown["Job Role Match"] = (match_score, 20)
    score += match_score
    
    return {"total": min(score, 100), "breakdown": breakdown}

def build_suggestions(text: str, sections: dict, skills: dict, job_match: dict, score: int) -> list:
    tips = []
    wc = count_words(text)
    if wc < 300:
        tips.append("Your resume is quite short. Aim for 400–700 words to give recruiters enough detail.")
    if not sections.get("Summary"):
        tips.append("Add a professional summary at the top — 2-3 sentences about who you are and what you bring.")
    if not sections.get("Projects"):
        tips.append("Include a Projects section with 2-3 key projects showcasing your work and impact.")
    if not sections.get("Certifications"):
        tips.append("Consider adding certifications relevant to your target role to stand out.")
    if not detect_email(text):
        tips.append("Add your email address — it's the primary contact method for recruiters.")
    if not detect_linkedin(text):
        tips.append("Add your LinkedIn profile URL to make it easy for recruiters to learn more.")
    if not detect_github(text):
        tips.append("Link your GitHub profile to showcase your code and projects.")
    if job_match.get("missing"):
        top_missing = ", ".join(job_match["missing"][:4])
        tips.append(f"For your target role, consider adding experience with: {top_missing}.")
    if score < 60:
        tips.append("Use strong action verbs like 'Led', 'Built', 'Designed', 'Improved' to start bullet points.")
    if not any(re.search(r'\d+%|\d+x|\$\d+', text)):
        tips.append("Quantify your achievements — e.g., 'Improved API response time by 40%' is far stronger.")
    if len(tips) == 0:
        tips.append("Great resume! Keep it updated and tailor it for each specific job application.")
    return tips

def get_score_color(score: int):
    if score >= 75:
        return "#10b981", "Excellent"
    elif score >= 55:
        return "#f59e0b", "Good"
    else:
        return "#ef4444", "Needs Work"

def ai_summarize(text: str, summarizer) -> str:
    if summarizer is None:
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 3:
            return ". ".join(sentences[:3]) + "."
        return text[:500]
    try:
        chunk = text[:3000]
        result = summarizer(chunk, max_length=120, min_length=40, do_sample=False)
        return result[0]["summary_text"]
    except Exception as e:
        return f"AI summary not available. {str(e)[:100]}"

def render_progress(label: str, value: int, max_val: int, color_class: str = ""):
    pct = int(value / max_val * 100)
    fill_class = f"progress-bar-fill{'-' + color_class if color_class else ''}"
    st.markdown(f"""
    <div class="progress-wrap">
        <div class="progress-label"><span>{label}</span><span>{value}/{max_val}</span></div>
        <div class="progress-bar-bg">
            <div class="{fill_class}" style="width:{pct}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <div style="font-size:2.5rem;">📄</div>
        <div style="font-family:'Space Grotesk',sans-serif; font-size:1.1rem;
             font-weight:700; color:#ccd6f6; margin-top:0.25rem;">Resume Analyzer</div>
        <div style="color:#8892b0; font-size:0.78rem; margin-top:0.2rem;">Powered by HuggingFace + PyTorch</div>
    </div>
    <hr style="border:none;height:1px;background:rgba(102,126,234,0.3);margin:1rem 0;">
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Target Job Role</div>', unsafe_allow_html=True)
    selected_role = st.selectbox(
        "Role",
        list(JOB_ROLE_SKILLS.keys()),
        label_visibility="collapsed",
    )

    st.markdown('<div style="margin-top:1rem;" class="sidebar-label">Upload Resume</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload",
        type=["pdf", "docx", "doc", "txt"],
        label_visibility="collapsed",
        help="PDF, DOCX, or TXT files supported",
    )
    st.markdown('<div class="upload-hint">PDF · DOCX · TXT</div>', unsafe_allow_html=True)

    st.markdown('<div style="margin-top:1rem;" class="sidebar-label">Or paste resume text</div>',
                unsafe_allow_html=True)
    pasted_text = st.text_area("Paste text", height=180, label_visibility="collapsed",
                               placeholder="Paste your resume content here...")

    st.markdown('<div style="margin-top:1.2rem;"></div>', unsafe_allow_html=True)
    analyze_btn = st.button("🔍  Analyze Resume")

    st.markdown("""
    <hr style="border:none;height:1px;background:rgba(102,126,234,0.2);margin:1.5rem 0 1rem;">
    <div style="color:#8892b0;font-size:0.78rem;line-height:1.6;">
    <b style="color:#a8b4c8;">Models used:</b><br>
    • facebook/bart-large-mnli<br>
    • sshleifer/distilbart-cnn-6-6<br><br>
    <b style="color:#a8b4c8;">Framework:</b> PyTorch<br>
    <b style="color:#a8b4c8;">100% offline</b> · No API keys
    </div>
    """, unsafe_allow_html=True)

# ============================================
# MAIN AREA
# ============================================

st.markdown('<div class="hero-header">AI Resume Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Upload your resume and get instant AI-powered feedback, '
    'skill gap analysis, and actionable suggestions.</div>',
    unsafe_allow_html=True,
)

if not analyze_btn:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="section-card" style="text-align:center;padding:2rem 1.5rem;">
            <div style="font-size:2.5rem;margin-bottom:0.75rem;">🎯</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-weight:600;
                 color:#ccd6f6;font-size:1.05rem;margin-bottom:0.5rem;">Role Match</div>
            <div style="color:#8892b0;font-size:0.88rem;line-height:1.6;">See exactly which skills you have vs what recruiters want</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="section-card" style="text-align:center;padding:2rem 1.5rem;">
            <div style="font-size:2.5rem;margin-bottom:0.75rem;">🧠</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-weight:600;
                 color:#ccd6f6;font-size:1.05rem;margin-bottom:0.5rem;">AI Insights</div>
            <div style="color:#8892b0;font-size:0.88rem;line-height:1.6;">Get a smart summary and skill analysis from NLP models</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="section-card" style="text-align:center;padding:2rem 1.5rem;">
            <div style="font-size:2.5rem;margin-bottom:0.75rem;">📈</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-weight:600;
                 color:#ccd6f6;font-size:1.05rem;margin-bottom:0.5rem;">Score & Tips</div>
            <div style="color:#8892b0;font-size:0.88rem;line-height:1.6;">Receive an ATS score with specific improvement suggestions</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.stop()

# ============================================
# ANALYSIS FLOW
# ============================================

resume_text = ""
if uploaded_file:
    with st.spinner("📂 Reading your resume..."):
        resume_text = extract_text(uploaded_file)
elif pasted_text.strip():
    resume_text = pasted_text.strip()

if not resume_text:
    st.warning("⚠️ Please upload a file or paste your resume text before analyzing.")
    st.stop()

with st.spinner("🤖 Loading AI models (first run takes ~30 sec)..."):
    classifier, summarizer = load_models()

with st.spinner("⚙️ Analyzing your resume..."):
    sections = detect_sections(resume_text)
    skills = extract_skills(resume_text)
    job_match = match_job_skills(resume_text, selected_role)
    score_data = score_resume(resume_text, sections, skills, job_match)
    suggestions = build_suggestions(resume_text, sections, skills, job_match, score_data["total"])
    ai_summary = ai_summarize(resume_text, summarizer)

total_score = score_data["total"]
score_color, score_label = get_score_color(total_score)

# Score and Breakdown
col_score, col_meta = st.columns([1, 2], gap="large")

with col_score:
    st.markdown(f"""
    <div class="score-card">
        <div class="score-number" style="color:{score_color};">{total_score}</div>
        <div style="color:{score_color};font-size:0.9rem;font-weight:600;margin-top:0.25rem;">{score_label}</div>
        <div class="score-label">ATS RESUME SCORE</div>
    </div>
    """, unsafe_allow_html=True)

with col_meta:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Score Breakdown</div>', unsafe_allow_html=True)
    for label, (val, max_val) in score_data["breakdown"].items():
        pct = val / max_val
        color = "green" if pct >= 0.75 else ("yellow" if pct >= 0.5 else "red")
        render_progress(label, val, max_val, color)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# AI Summary and Sections
col_ai, col_sec = st.columns(2, gap="large")

with col_ai:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🤖 AI Summary</div>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:#a8b4c8;font-size:0.92rem;line-height:1.7;">{ai_summary}</p>', unsafe_allow_html=True)
    wc = count_words(resume_text)
    
    email_found = detect_email(resume_text)
    phone_found = detect_phone(resume_text)
    linkedin_found = detect_linkedin(resume_text)
    github_found = detect_github(resume_text)
    
    email_badge = "badge-green" if email_found else "badge-red"
    phone_badge = "badge-green" if phone_found else "badge-red"
    linkedin_badge = "badge-green" if linkedin_found else "badge-yellow"
    github_badge = "badge-green" if github_found else "badge-yellow"
    
    email_status = "✓ Found" if email_found else "✗ Missing"
    phone_status = "✓ Found" if phone_found else "✗ Missing"
    linkedin_status = "✓ Found" if linkedin_found else "— Not found"
    github_status = "✓ Found" if github_found else "— Not found"
    
    st.markdown(f"""
    <div class="metric-row"><span class="metric-key">Word Count</span>
        <span class="metric-val">{wc}</span></div>
    <div class="metric-row"><span class="metric-key">Email</span>
        <span class="badge {email_badge}">{email_status}</span></div>
    <div class="metric-row"><span class="metric-key">Phone</span>
        <span class="badge {phone_badge}">{phone_status}</span></div>
    <div class="metric-row"><span class="metric-key">LinkedIn</span>
        <span class="badge {linkedin_badge}">{linkedin_status}</span></div>
    <div class="metric-row"><span class="metric-key">GitHub</span>
        <span class="badge {github_badge}">{github_status}</span></div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_sec:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Section Checklist</div>', unsafe_allow_html=True)
    for sec_name, present in sections.items():
        icon = "✅" if present else "❌"
        badge = "badge-green" if present else "badge-red"
        status = "Present" if present else "Missing"
        st.markdown(f"""
        <div class="metric-row">
            <span class="metric-key">{icon} {sec_name}</span>
            <span class="badge {badge}">{status}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# Skills Section
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🛠️ Detected Skills</div>', unsafe_allow_html=True)

skill_cols = st.columns(len(SKILL_CATEGORIES))
for col, (cat, found_skills) in zip(skill_cols, skills.items()):
    with col:
        st.markdown(f'<div style="color:#8892b0;font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem;">{cat}</div>', unsafe_allow_html=True)
        if found_skills:
            pills = "".join(f'<span class="skill-pill skill-pill-found">{s}</span>' for s in found_skills)
        else:
            pills = '<span style="color:#8892b0;font-size:0.82rem;">None detected</span>'
        st.markdown(f'<div class="skill-pills">{pills}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Job Match Section
if selected_role != "General / Other":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    pct = job_match["match_pct"]
    pct_color, pct_label = get_score_color(pct)
