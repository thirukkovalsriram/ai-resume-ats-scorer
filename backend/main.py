import re
import fitz
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="AI Resume ATS Scorer")

# Common technical & soft skills dictionary for accurate keyword extraction
TECH_SKILLS = {
    "python", "java", "javascript", "typescript", "c", "c++", "sql", "html", "html5",
    "css", "css3", "tailwind", "tailwindcss", "react", "reactjs", "node", "nodejs",
    "fastapi", "flask", "django", "rest", "api", "apis", "restful", "git", "github",
    "sqlite", "postgresql", "mysql", "mongodb", "docker", "aws", "linux", "pandas",
    "numpy", "scikit-learn", "machine learning", "nlp", "data structures", "algorithms",
    "oop", "json", "excel", "power bi", "tableau", "agile", "debugging", "testing"
}

class ATSResult(BaseModel):
    match_score: int
    matched_keywords: List[str]
    missing_keywords: List[str]
    bullet_improvements: List[str]
    overall_summary: str

def extract_text_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "\n".join(page.get_text("text") for page in doc).strip()

def extract_skills(text: str) -> set:
    text_lower = text.lower()
    found = set()
    for skill in TECH_SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.add(skill)
    return found

@app.get("/")
def serve_ui():
    return FileResponse("index.html")

@app.post("/api/analyze", response_model=ATSResult)
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    if not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    resume_text = extract_text_from_pdf(await resume.read())
    if len(resume_text) < 50:
        raise HTTPException(status_code=400, detail="Could not read text from PDF.")

    # 1. TF-IDF + Cosine Similarity score (0-100)
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
    text_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    # 2. Skill keyword match
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(job_description)

    matched = sorted(resume_skills.intersection(jd_skills))
    missing = sorted(jd_skills - resume_skills)

    skill_ratio = (len(matched) / len(jd_skills)) if jd_skills else text_similarity
    final_score = min(100, int(round((skill_ratio * 0.7 + text_similarity * 0.3) * 100)))

    # 3. Actionable suggestions
    tips = []
    if missing:
        tips.append(f"Add missing job keywords naturally into your skills or project bullets: {', '.join(missing[:5])}.")
    if not re.search(r"\d+%|\d+\+", resume_text):
        tips.append("Quantify your project impact with numbers (e.g., 'Reduced API response time by 40%').")
    tips.append("Start each project bullet with strong action verbs like Built, Engineered, Deployed, or Optimized.")

    summary = (
        f"Your resume matched {len(matched)} of {len(jd_skills)} key technical skills in the job description, "
        f"giving an overall ATS compatibility score of {final_score}%."
    )

    return ATSResult(
        match_score=final_score,
        matched_keywords=matched if matched else ["General Resume Structure"],
        missing_keywords=missing if missing else ["None — great keyword coverage!"],
        bullet_improvements=tips,
        overall_summary=summary,
    )