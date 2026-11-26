"""Resume screening engine: TF-IDF + cosine similarity + skill matching."""
import re
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocess import preprocess

SKILLS_FILE = Path(__file__).resolve().parents[1] / "data" / "skills.txt"


def load_skills(path: Path = SKILLS_FILE) -> list[str]:
    return [s.strip().lower() for s in path.read_text().splitlines() if s.strip()]


def find_skills(text: str, skills: list[str]) -> set[str]:
    """Return the skills from the master list that appear in the text."""
    text = text.lower()
    found = set()
    for skill in skills:
        if re.search(rf"(?<![a-z0-9+#]){re.escape(skill)}(?![a-z0-9+#])", text):
            found.add(skill)
    return found


def rank_resumes(job_description: str, resumes: dict[str, str],
                 skills: list[str] | None = None,
                 similarity_weight: float = 0.5) -> pd.DataFrame:
    """Rank resumes against a job description.

    Final score = w * TF-IDF cosine similarity + (1 - w) * required-skill coverage.
    If the job description lists no known skills, the score is the similarity alone.
    """
    if not resumes:
        raise ValueError("No resumes provided.")
    skills = skills or load_skills()
    jd_skills = find_skills(job_description, skills)

    names = list(resumes)
    corpus = [preprocess(job_description)] + [preprocess(resumes[n]) for n in names]
    tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    matrix = tfidf.fit_transform(corpus)
    sims = cosine_similarity(matrix[0:1], matrix[1:]).ravel()

    rows = []
    for name, sim in zip(names, sims):
        resume_skills = find_skills(resumes[name], skills)
        matched = sorted(jd_skills & resume_skills)
        missing = sorted(jd_skills - resume_skills)
        coverage = len(matched) / len(jd_skills) if jd_skills else None
        score = sim if coverage is None else similarity_weight * sim + (1 - similarity_weight) * coverage
        rows.append({
            "Candidate": name,
            "Final Score (%)": round(score * 100, 1),
            "Text Similarity (%)": round(sim * 100, 1),
            "Skill Coverage (%)": round(coverage * 100, 1) if coverage is not None else None,
            "Matched Skills": ", ".join(matched),
            "Missing Skills": ", ".join(missing),
        })
    df = pd.DataFrame(rows).sort_values("Final Score (%)", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)
    return df
