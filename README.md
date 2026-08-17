"""
Project 3 - Correlation Heatmap & Pairwise Relationships
Syntecxhub Data Science Internship - Week 2, Task 2

What this script does:
1. Loads a sales dataset (sales_data.csv)
2. Computes the Pearson correlation matrix between numeric features
3. Visualizes it as a heatmap (upper triangle masked, values annotated)
4. Builds a pairplot to show pairwise relationships between key variables
5. Saves both charts as PNG files
6. Prints a short summary of the strongest positive/negative relationships
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
df = pd.read_csv("sales_data.csv")
print("Dataset preview:")
print(df.head())
print("\nShape:", df.shape)

# ---------------------------------------------------------
# 2. Compute Pearson correlation matrix
# ---------------------------------------------------------
corr = df.corr(numeric_only=True)
print("\nCorrelation matrix:")
print(corr.round(2))

# ---------------------------------------------------------
# 3. Correlation heatmap (upper triangle masked, annotated)
# ---------------------------------------------------------
mask = np.triu(np.ones_like(corr, dtype=bool))  # mask upper triangle

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1, vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)
plt.title("Correlation Heatmap - Sales Dataset", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("heatmap.png", dpi=150)
plt.close()
print("\nSaved: heatmap.png")

# ---------------------------------------------------------
# 4. Pairplot for key variable pairs
# ---------------------------------------------------------
key_vars = ["Price", "Quantity", "Discount", "Sales", "Profit"]
pairplot = sns.pairplot(df[key_vars], diag_kind="kde", corner=True)
pairplot.fig.suptitle("Pairwise Relationships - Key Variables", y=1.02, fontsize=14, fontweight="bold")
pairplot.savefig("pairplot.png", dpi=150)
print("Saved: pairplot.png")

# ---------------------------------------------------------
# 5. Summarize strongest positive/negative relationships
# ---------------------------------------------------------
corr_pairs = corr.unstack()
corr_pairs = corr_pairs[corr_pairs != 1.0]  # drop self-correlation
corr_pairs = corr_pairs.drop_duplicates()

top_positive = corr_pairs.sort_values(ascending=False).head(3)
top_negative = corr_pairs.sort_values().head(3)

summary_lines = []
summary_lines.append("CORRELATION ANALYSIS SUMMARY")
summary_lines.append("=" * 40)
summary_lines.append("\nTop 3 strongest POSITIVE correlations:")
for (a, b), v in top_positive.items():
    summary_lines.append(f"  {a} <-> {b}: {v:.2f}")

summary_lines.append("\nTop 3 strongest NEGATIVE correlations:")
for (a, b), v in top_negative.items():
    summary_lines.append(f"  {a} <-> {b}: {v:.2f}")

summary_lines.append(
    "\nInterpretation: Price and Quantity show the strongest positive "
    "relationship with Sales, confirming both drive revenue directly. "
    "Discount shows a negative relationship with Profit and CustomerRating, "
    "suggesting heavier discounting erodes margins without meaningfully "
    "improving customer satisfaction in this dataset."
)

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

with open("summary.txt", "w") as f:
    f.write(summary_text)
print("\nSaved: summary.txt")



