from flask import Flask, render_template, request
import joblib
from utils import extract_text_from_file
from sklearn.metrics.pairwise import cosine_similarity
from utils import suggest_missing_skills
from utils import semantic_similarity
app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files.get('resume_file')
    job_desc = request.form.get('job_desc', '')

    overall_score = None
    dynamic_score = None
    match_score = None
    missing_skills = []

    # Check if a file was uploaded
    if file:
        text = extract_text_from_file(file)

        # Check if extraction was successful
        if text.strip() != "":
            vector = vectorizer.transform([text])
            prob = model.predict_proba(vector)[0][1]
            overall_score = round(prob * 100, 2)

            # If job description provided:
            if job_desc.strip() != "":
                # Dynamic Resume Score (keyword match %)
                resume_words = set(text.lower().split())
                job_words = set(job_desc.lower().split())
                matching = resume_words.intersection(job_words)
                dynamic_score = round((len(matching) / len(job_words)) * 100, 2) if job_words else 0

                # Semantic Job Match % (using BERT similarity)
                match_score = semantic_similarity(text, job_desc)

                # Suggested Missing Skills
                missing_skills = suggest_missing_skills(text, job_desc)

    return render_template('index.html',
                           overall_score=overall_score,
                           dynamic_score=dynamic_score,
                           match_score=match_score,
                           missing_skills=missing_skills)

if __name__ == "__main__":
    app.run(debug=True)