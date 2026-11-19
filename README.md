# AI Resume ATS Scorer

A full-stack web application that parses PDF resumes and evaluates them against job descriptions using TF-IDF, Cosine Similarity, and technical skill keyword matching.
## Screenshots<img width="1920" height="1080" alt="Screenshot (78)" src="https://github.com/user-attachments/assets/05468e45-4404-4884-aaf5-bab4aa159e45" />





## Tech Stack
- **Backend:** Python, FastAPI, PyMuPDF (fitz), Scikit-learn
- **Frontend:** HTML5, Tailwind CSS, Vanilla JavaScript

## Run Locally
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn main:app --reload
Open http://127.0.0.1:8000 in your browser.
