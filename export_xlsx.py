import pandas as pd
import json
from src.rule_ranker import RuleBasedRanker

# Load candidates
with open("data/sample_candidates.json") as f:
    candidates = json.load(f)

# Initialize ranker and rank
ranker = RuleBasedRanker()
result = ranker.rank_candidates_with_reports(candidates)

# Export to Excel
result.dataframe.to_excel("recommended_candidates.xlsx", index=False)
print("Saved to recommended_candidates.xlsx")
