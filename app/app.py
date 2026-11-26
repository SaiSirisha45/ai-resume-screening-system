"""Streamlit app: upload resumes, paste a job description, get a ranked shortlist."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parser import extract_text
from src.screener import rank_resumes

st.set_page_config(page_title="AI Resume Screening System", page_icon="📄", layout="wide")
st.title("📄 AI Resume Screening System")
st.caption("NLP-based screening with TF-IDF, cosine similarity and skill matching.")

with st.sidebar:
    st.header("Settings")
    weight = st.slider("Weight of text similarity", 0.0, 1.0, 0.5, 0.05,
                       help="The rest of the score comes from required-skill coverage.")
    use_samples = st.checkbox("Use sample resumes and job description", value=False)

default_jd = (ROOT / "data" / "job_description.txt").read_text() if use_samples else ""
jd = st.text_area("Job description", value=default_jd, height=240,
                  placeholder="Paste the job description here...")

resumes: dict[str, str] = {}
if use_samples:
    for p in sorted((ROOT / "data" / "sample_resumes").glob("*.txt")):
        resumes[p.stem] = p.read_text()
else:
    files = st.file_uploader("Upload resumes (PDF, DOCX or TXT)",
                             type=["pdf", "docx", "txt"], accept_multiple_files=True)
    for f in files or []:
        try:
            resumes[f.name] = extract_text(f.name, f.read())
        except Exception as exc:
            st.warning(f"Could not read {f.name}: {exc}")

if st.button("Screen resumes", type="primary"):
    if not jd.strip():
        st.error("Please enter a job description.")
    elif not resumes:
        st.error("Please upload at least one resume.")
    else:
        df = rank_resumes(jd, resumes, similarity_weight=weight)
        st.subheader("Ranked shortlist")
        st.dataframe(df, width="stretch", hide_index=True)
        st.bar_chart(df.set_index("Candidate")["Final Score (%)"])
        st.download_button("Download results (CSV)", df.to_csv(index=False).encode(),
                           "ranking_results.csv", "text/csv")
