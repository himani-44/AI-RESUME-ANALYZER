import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re

st.title("AI Resume Analyzer (Improved)")

model = SentenceTransformer('all-MiniLM-L6-v2')

resume_text = st.text_area("Paste Resume Text")
job_desc = st.text_area("Paste Job Description")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

def get_similarity(resume, job):
    resume = clean_text(resume)
    job = clean_text(job)
    
    embeddings = model.encode([resume, job])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    return score * 100

if st.button("Analyze"):
    score = get_similarity(resume_text, job_desc)
    
    st.subheader(f"Semantic Match Score: {score:.2f}%")