import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

data = pd.DataFrame({
    'resume' : [
         'Experienced Python developer skilled in machine learning and data analysis.',
        'Software engineer with expertise in Java, Spring Boot, and cloud computing.',
        'Recent graduate looking for entry-level position in IT, no experience yet.',
        'Project manager with strong background in Agile and Scrum methodologies.',
        'No professional experience, seeking first job opportunity.'
    ],
    'label': [1, 1, 0, 1, 0]  # 1 = Good Resume, 0 = Needs Improvement
})

# Text Vecotoriztion (TF - IDF)

vectoriztion = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectoriztion.fit_transform(data['resume'])
y = data['label']

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, 'model.pkl')
joblib.dump(vectoriztion, 'vectorizer.pkl')

print("Model and vectorizer saved successfully.")