import pytest
from src.preprocess import preprocess, clean_text
from src.screener import find_skills, rank_resumes, load_skills


def test_clean_text_removes_email_and_url():
    out = clean_text("Contact me: a@b.com or https://x.com NOW!")
    assert "@" not in out and "http" not in out and out == out.lower()


def test_preprocess_removes_stopwords_and_lemmatizes():
    out = preprocess("The models are running in the pipelines")
    assert "the" not in out.split()
    assert "model" in out.split() and "pipeline" in out.split()


def test_find_skills_word_boundaries():
    skills = load_skills()
    found = find_skills("Experienced in Python and SQL, not JavaScript-free", skills)
    assert {"python", "sql"} <= found
    assert "r" not in find_skills("Python only", skills)


def test_ranking_prefers_relevant_resume():
    jd = "Looking for a Python machine learning engineer with NLP and SQL skills."
    resumes = {
        "good": "Python developer with machine learning, NLP and SQL experience.",
        "bad": "Graphic designer skilled in Photoshop and illustration.",
    }
    df = rank_resumes(jd, resumes)
    assert df.iloc[0]["Candidate"] == "good"
    assert df.iloc[0]["Final Score (%)"] > df.iloc[1]["Final Score (%)"]


def test_empty_resumes_raises():
    with pytest.raises(ValueError):
        rank_resumes("job", {})
