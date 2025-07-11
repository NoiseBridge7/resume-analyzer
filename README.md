1) Overview
    This is an AI-powered web application that analyzes resumes and provides insightful feedback on their quality and relevance to a given job description. 
    This has been built using Flask, Machine Learning , and has a responsive Frontend.

2)Project Structure
    resume-analyzer/
    │
    ├── app.py # Flask backend
    ├── utils.py # Text extraction & NLP helpers
    ├── model.pkl # Trained ML model
    ├── vectorizer.pkl # TF-IDF vectorizer
    ├── requirements.txt # Python dependencies
    ├── Procfile # For deployment
    ├── templates/
    │ └── index.html # Frontend
    └── static/
    └── styles.css # Custom styling

3) Install dependencies:
    pip install -r requirements.txt
3) Run the app:
    In the respective directory of the project folder run the following code
     python app.py
   
5) Open the Website
    http://127.0.0.1:5000

6)Deployment Options
    i)Deploy on Render (Recommended)
    ii) Deploy on Heroku

    Note: both Render and Heroku has a 512MB limit for free tier if you need to upload it is recomended for a paid subscription or you can just remove the BERT part of the code which does semantic analysis 
 
