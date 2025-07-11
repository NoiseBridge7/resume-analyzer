import docx
from pdfminer.high_level import extract_text

from sentence_transformers import SentenceTransformer, util
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_similarity(text1, text2):
    emb1 = bert_model.encode(text1, convert_to_tensor=True)
    emb2 = bert_model.encode(text2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb1, emb2).item()
    return round(similarity * 100, 2)


def suggest_missing_skills(resume_text, job_desc_text):
    known_skills = ['python', 'sql', 'machine learning', 'cloud', 'aws', 'java', 'project management']
    resume_words = set(resume_text.lower().split())
    job_words = set(job_desc_text.lower().split())
    missing = [skill for skill in known_skills if skill in job_words and skill not in resume_words]
    return missing


def extract_text_from_file(file):
    if file.filename.endswith('.pdf'):
        return extract_text(file.stream)
    elif file.filename.endswith('.docx'):
        doc = docx.Document(file)
        return '\n'.join([p.text for p in doc.paragraphs])
    else:
        return ""