"""
Generates a synthetic CSV with exactly the Kaggle schema so you can run the
whole pipeline before you drop the real file in. NOT real data - replace it.

    python src/make_sample_data.py
"""
import numpy as np
import pandas as pd

from config import DATA_DIR, QCHAT_CUTOFF

rng = np.random.default_rng(42)
N = 1054

rows = []
for i in range(1, N + 1):
    # latent "trait level" drives the item probabilities
    trait = rng.beta(2, 2)
    a = [int(rng.random() < 0.15 + 0.7 * trait) for _ in range(10)]
    score = sum(a)
    rows.append(
        {
            "Case_No": i,
            **{f"A{j+1}": a[j] for j in range(10)},
            "Age_Mons": int(rng.integers(12, 37)),
            "Qchat-10-Score": score,
            "Sex": rng.choice(["m", "f"], p=[0.7, 0.3]),
            "Ethnicity": rng.choice(
                ["White European", "asian", "middle eastern", "south asian",
                 "black", "Hispanic", "Latino", "mixed", "Others"],
                p=[0.30, 0.18, 0.16, 0.10, 0.06, 0.05, 0.05, 0.05, 0.05],
            ),
            "Jaundice": rng.choice(["Yes", "No"], p=[0.3, 0.7]),
            "Family_mem_with_ASD": rng.choice(["Yes", "No"], p=[0.2, 0.8]),
            "Who completed the test": rng.choice(
                ["family member", "Health Care Professional", "Self", "Others"],
                p=[0.75, 0.15, 0.05, 0.05],
            ),
            # the real dataset labels exactly this way
            "Class/ASD Traits": "Yes" if score > QCHAT_CUTOFF else "No",
        }
    )

df = pd.DataFrame(rows)
out = DATA_DIR / "sample_synthetic.csv"
df.to_csv(out, index=False)
print(f"Wrote {out}  ({len(df)} rows)")
print(df["Class/ASD Traits"].value_counts().to_string())
