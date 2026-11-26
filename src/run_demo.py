"""Rank the sample resumes against the sample job description."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.screener import rank_resumes

ROOT = Path(__file__).resolve().parents[1]
jd = (ROOT / "data" / "job_description.txt").read_text()
resumes = {p.stem: p.read_text() for p in sorted((ROOT / "data" / "sample_resumes").glob("*.txt"))}

df = rank_resumes(jd, resumes)
print(df[["Rank", "Candidate", "Final Score (%)", "Text Similarity (%)", "Skill Coverage (%)"]].to_string(index=False))
df.to_csv(ROOT / "data" / "ranking_results.csv", index=False)

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(8, 4.5))
plot_df = df.iloc[::-1]
ax.barh(plot_df["Candidate"], plot_df["Final Score (%)"], color="#4C72B0")
for i, v in enumerate(plot_df["Final Score (%)"]):
    ax.text(v + 0.8, i, f"{v:.1f}%", va="center")
ax.set_xlabel("Final match score (%)"); ax.set_title("Candidate Ranking for Machine Learning Engineer")
ax.set_xlim(0, 100)
fig.tight_layout(); fig.savefig(ROOT / "images" / "ranking_chart.png", dpi=130)
