import streamlit as st              # both frontend &backend in one profile creates web ui quickly 
from sklearn.metrics.pairwise import cosine_similarity # cosine - used to maesre similarity between two vectors 
from sentence_transformers import SentenceTransformer #converting text into embeddings catching semantic meaning 
import re # data cleaning and text pre processing :- NLP

st.title("AI Resume Analyzer (Improved)")

model = SentenceTransformer('all-MiniLM-L6-v2') # lightweight , fast and optimized for semantic similarity while providing good accuracy 
# user input 
resume_text = st.text_area("Paste Resume Text ") 
job_desc = st.text_area("Paste Job Description")
# clean text function 
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

def get_similarity(resume, job):
    #preprocessing 
    resume = clean_text(resume)
    job = clean_text(job)


    # transformer converts texts into vectors 
    embeddings = model.encode([resume, job])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]   # calculating similarity 0,0 :- 2D array
    
    return score * 100 # decimal * percentage 
    # runds the analaysis when clicked 
if st.button("Analyze"):
    score = get_similarity(resume_text, job_desc)
    
    st.subheader(f"Semantic Match Score: {score:.2f}%")
