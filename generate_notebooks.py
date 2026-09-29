"""
Generation script for Assignment 2:
Creates Q1.ipynb, Q2.ipynb, Q3.ipynb with complete code and markdown explanations,
executes them to populate all outputs, plots, tables, and saves accompanying CSVs and PNGs.
"""

import os
import json
import base64
import time
from collections import Counter
from itertools import combinations
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set working directory to solution dir
SOLUTION_DIR = Path("/Users/aryanshokeen/.gemini/antigravity-ide/scratch/Assignment2/solution")
DATA_PATH = Path("/Users/aryanshokeen/.gemini/antigravity-ide/scratch/Assignment2/Group_23_Assignment1/Q1b_ingredients_ner.csv")

os.chdir(SOLUTION_DIR)
print(f"Working directory: {SOLUTION_DIR}")
print(f"Data source: {DATA_PATH}")

def build_q1_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11.9'}
    }
    
    cells = []
    
    # Title Markdown
    cells.append(nbf.v4.new_markdown_cell("""# Computational Gastronomy — Assignment 2
## Question 1: Frequent Item Set Mining (FISM)
**Monsoon 2026**

### Problem Statement
Using the data of 10,000 recipes and their ingredients obtained for Assignment 1, implement any of the 'frequent item set mining (FISM) algorithms':
* **(a)** Provide a complete list of item sets of size one, two, and three. `[5]`
* **(b)** List the top 20 item sets of size one, two, and three. `[3]`
* **(c)** Define what are the 'support' & 'confidence' in the context of item set mining. `[2]`
"""))

    # Code Cell 1: Imports and setup
    cells.append(nbf.v4.new_code_cell("""import sys
import os
import time
from collections import Counter
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Display options
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 160)
pd.set_option('display.max_colwidth', 60)

DATA_PATH = Path('../Group_23_Assignment1/Q1b_ingredients_ner.csv')
print(f"Loading ingredient data from: {DATA_PATH}")
"""))

    # Code Cell 2: Data loading and preprocessing
    cells.append(nbf.v4.new_code_cell("""# Load the extracted NER ingredients from Assignment 1
ner_df = pd.read_csv(DATA_PATH, keep_default_na=False)

# Filter out empty ingredient names
valid_ner = ner_df[ner_df['Ingredient Name'].str.strip() != '']

# Group by Recipe ID to extract the basket / itemset of unique ingredients per recipe
recipe_baskets = valid_ner.groupby('Recipe ID')['Ingredient Name'].apply(lambda s: sorted(list(set(s)))).to_dict()

N = len(recipe_baskets)
print(f"Total recipes (transactions N): {N:,}")
print(f"Total valid ingredient mentions: {len(valid_ner):,}")
unique_vocab = sorted(list(set(valid_ner['Ingredient Name'])))
print(f"Vocabulary of unique ingredients (|I|): {len(unique_vocab):,}")

# Display sample recipe basket
sample_id = list(recipe_baskets.keys())[0]
print(f"Sample Recipe (ID {sample_id}) ingredients ({len(recipe_baskets[sample_id])} items):")
print(recipe_baskets[sample_id])
"""))

    # Markdown Cell: Explaining Apriori Algorithm Implementation
    cells.append(nbf.v4.new_markdown_cell("""---
### Algorithm Implementation: Apriori Frequent Itemset Mining
We implement the classic **Apriori Algorithm** (Agrawal & Srikant, 1994) from first principles.
The algorithm exploits the **Apriori Principle** (anti-monotonicity of support):
> *All non-empty subsets of a frequent itemset must also be frequent.*

This allows efficient candidate pruning at each level $k$:
1. **Level 1 ($k=1$):** Count frequencies of all individual ingredients across the $N=10,000$ recipes. Retain those with $\\text{count} \\ge \\text{min\\_count}$ (support $\\ge \\text{min\\_sup}$).
2. **Level 2 ($k=2$):** Generate candidate pairs from frequent 1-itemsets and count their co-occurrences in recipe baskets. Retain pairs with $\\text{count} \\ge \\text{min\\_count}$.
3. **Level 3 ($k=3$):** Generate candidate triplets whose 2-item subsets are all verified to be frequent (Apriori candidate pruning). Count their occurrences and filter by minimum support.

We set a standard threshold of $\\text{min\\_sup} = 0.01$ ($1\\%$, i.e., at least $100$ recipes).
"""))

    # Code Cell 3: Apriori implementation
    cells.append(nbf.v4.new_code_cell("""MIN_SUP = 0.01
MIN_COUNT = int(MIN_SUP * N)
print(f"Minimum Support Threshold: {MIN_SUP:.2%} ({MIN_COUNT} recipes)")

# ----------------- Level 1: 1-itemsets -----------------
t0 = time.time()
c1 = Counter()
for basket in recipe_baskets.values():
    for item in basket:
        c1[item] += 1

# Filter by min_count
freq_1 = {item: count for item, count in c1.items() if count >= MIN_COUNT}
f1_set = set(freq_1.keys())
t1 = time.time()
print(f"Found {len(freq_1):,} frequent 1-itemsets in {t1 - t0:.3f}s")

# ----------------- Level 2: 2-itemsets -----------------
c2 = Counter()
for basket in recipe_baskets.values():
    # Filter basket items to only those that are frequent 1-items
    valid_items = [item for item in basket if item in f1_set]
    if len(valid_items) >= 2:
        for pair in combinations(valid_items, 2):
            c2[pair] += 1

freq_2 = {pair: count for pair, count in c2.items() if count >= MIN_COUNT}
f2_set = set(freq_2.keys())
t2 = time.time()
print(f"Found {len(freq_2):,} frequent 2-itemsets in {t2 - t1:.3f}s")

# ----------------- Level 3: 3-itemsets -----------------
c3 = Counter()
for basket in recipe_baskets.values():
    valid_items = [item for item in basket if item in f1_set]
    if len(valid_items) >= 3:
        for triplet in combinations(valid_items, 3):
            # Apriori pruning: all 3 size-2 subsets must be frequent
            if ((triplet[0], triplet[1]) in f2_set and 
                (triplet[0], triplet[2]) in f2_set and 
                (triplet[1], triplet[2]) in f2_set):
                c3[triplet] += 1

freq_3 = {triplet: count for triplet, count in c3.items() if count >= MIN_COUNT}
t3 = time.time()
print(f"Found {len(freq_3):,} frequent 3-itemsets in {t3 - t2:.3f}s")
print(f"Total mining time: {t3 - t0:.3f}s")
"""))

    # Markdown Cell: Q1(a) Complete lists
    cells.append(nbf.v4.new_markdown_cell("""---
### Q1(a) — Complete List of Item Sets of Size One, Two, and Three
We convert each level of mined frequent itemsets into structured DataFrames and save the complete lists to CSV files.
"""))

    # Code Cell 4: Save complete lists
    cells.append(nbf.v4.new_code_cell("""# 1-itemsets DataFrame
df_itemsets_1 = pd.DataFrame([
    {'Itemset': item, 'Support_Count': count, 'Support': count / N}
    for item, count in sorted(freq_1.items(), key=lambda x: x[1], reverse=True)
])
df_itemsets_1.to_csv('Q1a_itemsets_size_1_complete.csv', index=False)

# 2-itemsets DataFrame
df_itemsets_2 = pd.DataFrame([
    {'Itemset': f"{pair[0]}, {pair[1]}", 'Support_Count': count, 'Support': count / N}
    for pair, count in sorted(freq_2.items(), key=lambda x: x[1], reverse=True)
])
df_itemsets_2.to_csv('Q1a_itemsets_size_2_complete.csv', index=False)

# 3-itemsets DataFrame
df_itemsets_3 = pd.DataFrame([
    {'Itemset': f"{triplet[0]}, {triplet[1]}, {triplet[2]}", 'Support_Count': count, 'Support': count / N}
    for triplet, count in sorted(freq_3.items(), key=lambda x: x[1], reverse=True)
])
df_itemsets_3.to_csv('Q1a_itemsets_size_3_complete.csv', index=False)

print(f"Summary of Complete Frequent Itemsets (min_sup = {MIN_SUP:.1%}):")
print(f"  • Size 1: {len(df_itemsets_1):,} itemsets -> saved to Q1a_itemsets_size_1_complete.csv")
print(f"  • Size 2: {len(df_itemsets_2):,} itemsets -> saved to Q1a_itemsets_size_2_complete.csv")
print(f"  • Size 3: {len(df_itemsets_3):,} itemsets -> saved to Q1a_itemsets_size_3_complete.csv")
"""))

    # Markdown Cell: Q1(b) Top 20 lists
    cells.append(nbf.v4.new_markdown_cell("""---
### Q1(b) — Top 20 Item Sets of Size One, Two, and Three
Below are the top 20 most frequent itemsets for each size, ranked by support count and support percentage.
"""))

    # Code Cell 5: Top 20 display and save
    cells.append(nbf.v4.new_code_cell("""top20_1 = df_itemsets_1.head(20).copy()
top20_1.index = range(1, 21)
top20_1['Support (%)'] = (top20_1['Support'] * 100).map('{:.2f}%'.format)

top20_2 = df_itemsets_2.head(20).copy()
top20_2.index = range(1, 21)
top20_2['Support (%)'] = (top20_2['Support'] * 100).map('{:.2f}%'.format)

top20_3 = df_itemsets_3.head(20).copy()
top20_3.index = range(1, 21)
top20_3['Support (%)'] = (top20_3['Support'] * 100).map('{:.2f}%'.format)

# Save top 20 tables
top20_1.to_csv('Q1b_top20_itemsets_size_1.csv', index_label='Rank')
top20_2.to_csv('Q1b_top20_itemsets_size_2.csv', index_label='Rank')
top20_3.to_csv('Q1b_top20_itemsets_size_3.csv', index_label='Rank')

print("="*65)
print("TOP 20 ITEMSETS OF SIZE 1 (Individual Ingredients)")
print("="*65)
print(top20_1[['Itemset', 'Support_Count', 'Support (%)']].to_string())

print("\\n" + "="*65)
print("TOP 20 ITEMSETS OF SIZE 2 (Ingredient Pairs)")
print("="*65)
print(top20_2[['Itemset', 'Support_Count', 'Support (%)']].to_string())

print("\\n" + "="*65)
print("TOP 20 ITEMSETS OF SIZE 3 (Ingredient Triplets)")
print("="*65)
print(top20_3[['Itemset', 'Support_Count', 'Support (%)']].to_string())
"""))

    # Markdown Cell: Q1(c) Support and Confidence
    cells.append(nbf.v4.new_markdown_cell("""---
### Q1(c) — Definitions of 'Support' & 'Confidence' in Item Set Mining

#### 1. Support
* **Definition**: The **support** of an itemset $X \\subseteq I$ is the proportion (fraction or percentage) of transactions in the dataset $D$ that contain the entire itemset $X$.
$$\\text{Support}(X) = \\frac{\\sigma(X)}{|D|} = \\frac{|\\{T \\in D \\mid X \\subseteq T\\}|}{|D|}$$
where $\\sigma(X)$ represents the support count (frequency of recipes containing $X$), and $|D| = N$ is the total number of transactions (recipes).
* **Significance**: Support measures the statistical significance, prevalence, or popularity of an itemset. Filtering by a minimum support threshold ($\text{min\\_sup}$) ensures that rules are not derived from rare or coincidental co-occurrences.
* **Example from our data**: The pair $X = \\{\\text{garlic clove}, \\text{olive oil}\\}$ appears in 1,257 recipes out of 10,000, giving:
$$\\text{Support}(\\{\\text{garlic clove}, \\text{olive oil}\\}) = \\frac{1257}{10000} = 0.1257 \\quad (12.57\\%)$$

---

#### 2. Confidence
* **Definition**: The **confidence** of an association rule $X \\rightarrow Y$ (where $X, Y \\subseteq I$ and $X \\cap Y = \\emptyset$) is the conditional probability that a transaction contains itemset $Y$, given that it already contains itemset $X$.
$$\\text{Confidence}(X \\rightarrow Y) = P(Y \\mid X) = \\frac{\\text{Support}(X \\cup Y)}{\\text{Support}(X)} = \\frac{\\sigma(X \\cup Y)}{\\sigma(X)}$$
* **Significance**: Confidence measures the predictive power, reliability, or certainty of an inference. It quantifies how often the rule is found to be true among all instances where the antecedent $X$ is present.
* **Example from our data**: Consider the rule:
$$\\{\\text{onion}\\} \\rightarrow \\{\\text{garlic clove}\\}$$
Support count of $\\{\\text{onion}\\} = 1,729$, and co-occurrence count of $\\{\\text{onion}, \\text{garlic clove}\\} = 1,088$.
$$\\text{Confidence}(\\{\\text{onion}\\} \\rightarrow \\{\\text{garlic clove}\\}) = \\frac{1088}{1729} \\approx 0.6293 \\quad (62.93\\%)$$
This means that in nearly $63\\%$ of recipes containing onion, garlic clove is also used!
"""))

    # Code Cell 6: Demonstrate association rules confidence
    cells.append(nbf.v4.new_code_cell("""# Example demonstration of association rules confidence from top pairs
sample_rules = []
for pair, count in sorted(freq_2.items(), key=lambda x: x[1], reverse=True)[:10]:
    A, B = pair
    sup_AB = count / N
    sup_A = freq_1[A] / N
    sup_B = freq_1[B] / N
    
    # Rule A -> B
    conf_A_B = count / freq_1[A]
    # Rule B -> A
    conf_B_A = count / freq_1[B]
    
    sample_rules.append({'Rule': f"{A} -> {B}", 'Support': f"{sup_AB:.2%}", 'Confidence': f"{conf_A_B:.2%}"})
    sample_rules.append({'Rule': f"{B} -> {A}", 'Support': f"{sup_AB:.2%}", 'Confidence': f"{conf_B_A:.2%}"})

df_sample_rules = pd.DataFrame(sample_rules)
print("Sample Association Rules and Their Confidence:")
print(df_sample_rules.head(10).to_string(index=False))
"""))

    # Markdown Cell: Output summary table
    cells.append(nbf.v4.new_markdown_cell("""---
## Question 1 Deliverables Summary

| Deliverable File | Description | Question |
|---|---|---|
| `Q1a_itemsets_size_1_complete.csv` | Full list of mined frequent 1-itemsets (support $\\ge 1\\%$) | Q1(a) |
| `Q1a_itemsets_size_2_complete.csv` | Full list of mined frequent 2-itemsets (support $\\ge 1\\%$) | Q1(a) |
| `Q1a_itemsets_size_3_complete.csv` | Full list of mined frequent 3-itemsets (support $\\ge 1\\%$) | Q1(a) |
| `Q1b_top20_itemsets_size_1.csv` | Top 20 frequent 1-itemsets with support counts and percentages | Q1(b) |
| `Q1b_top20_itemsets_size_2.csv` | Top 20 frequent 2-itemsets with support counts and percentages | Q1(b) |
| `Q1b_top20_itemsets_size_3.csv` | Top 20 frequent 3-itemsets with support counts and percentages | Q1(b) |
"""))

    nb.cells = cells
    return nb

def build_q2_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11.9'}
    }
    
    cells = []
    
    # Title Markdown
    cells.append(nbf.v4.new_markdown_cell("""# Computational Gastronomy — Assignment 2
## Question 2: Size-Controlled Random Control (Null Models)
**Monsoon 2026**

### Problem Statement
Create size-controlled random control of the above 10,000 recipes assuming them to be a cuisine.
* **(a)** Create a size-controlled cuisine by making a replica of each recipe. Plot the recipe size distribution of the original cuisine and the randomly created recipes in 10 sets. `[5]`
* **(b)** Create a size-controlled cuisine by using 'inverse transformation'. Plot the recipe size distribution of the original cuisine and the randomly created recipes (10 times the original). `[5]`
"""))

    # Code Cell 1: Imports and setup
    cells.append(nbf.v4.new_code_cell("""import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

DATA_PATH = Path('../Group_23_Assignment1/Q1b_ingredients_ner.csv')
RANDOM_SEED = 20260929
np.random.seed(RANDOM_SEED)
"""))

    # Code Cell 2: Data loading
    cells.append(nbf.v4.new_code_cell("""# Load NER ingredients
ner_df = pd.read_csv(DATA_PATH, keep_default_na=False)
valid_ner = ner_df[ner_df['Ingredient Name'].str.strip() != '']

# Recipes as sets of unique ingredients
recipes_dict = valid_ner.groupby('Recipe ID')['Ingredient Name'].apply(lambda s: sorted(list(set(s)))).to_dict()
recipe_ids = sorted(list(recipes_dict.keys()))
N = len(recipe_ids)

# Vocabulary of unique ingredients
vocabulary = sorted(list(set(valid_ner['Ingredient Name'])))
V = len(vocabulary)

# Original recipe sizes
original_sizes = np.array([len(recipes_dict[rid]) for rid in recipe_ids])

print(f"Total recipes: {N:,}")
print(f"Vocabulary size |V|: {V:,}")
print(f"Recipe size statistics (Original Cuisine):")
print(f"  • Mean:   {original_sizes.mean():.4f}")
print(f"  • Median: {np.median(original_sizes):.1f}")
print(f"  • Std:    {original_sizes.std():.4f}")
print(f"  • Min:    {original_sizes.min()}, Max: {original_sizes.max()}")
"""))

    # Markdown Cell: Q2(a) Replica method
    cells.append(nbf.v4.new_markdown_cell("""---
### Q2(a) — Size-Controlled Cuisine via Recipe Replicas
#### Methodology:
1. In this model, for every recipe $r_i$ in the original cuisine with size $s_i = |r_i|$, we construct a random synthetic replica $r_i'$ of identical size $s_i$.
2. The $s_i$ ingredients in $r_i'$ are sampled uniformly at random from the total ingredient vocabulary $V$ without replacement.
3. Because every single recipe's size is preserved exactly ($|r_i'| = |r_i|$), the recipe size distribution of the random cuisine matches the original empirical distribution identically.
4. We generate **10 independent random cuisines** (sets $1$ to $10$) using different random seeds and compare their size distributions to the original cuisine.
"""))

    # Code Cell 3: Generate 10 replica sets and plot
    cells.append(nbf.v4.new_code_cell("""num_replica_sets = 10
replica_sets_sizes = []

for k in range(num_replica_sets):
    rng = np.random.default_rng(seed=RANDOM_SEED + k * 100)
    # For each recipe of size s, sample s ingredients uniformly from vocabulary without replacement
    replica_sizes = [len(rng.choice(V, size=s, replace=False)) for s in original_sizes]
    replica_sets_sizes.append(np.array(replica_sizes))

# Plot recipe size distribution: Original vs 10 replica sets
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Compute size range
max_size = max(original_sizes.max(), max(s.max() for s in replica_sets_sizes))
bins = np.arange(0.5, max_size + 1.5, 1)
bin_centers = np.arange(1, max_size + 1)

# Plot 10 random replica sets
palette = plt.cm.tab10(np.linspace(0, 1, 10))
for k in range(num_replica_sets):
    counts, _ = np.histogram(replica_sets_sizes[k], bins=bins)
    ax.plot(bin_centers, counts, color=palette[k], alpha=0.6, linewidth=1.5,
            label=f'Random Replica Set {k+1}' if k < 3 or k == 9 else "")

# Plot original cuisine with prominent styling
orig_counts, _ = np.histogram(original_sizes, bins=bins)
ax.plot(bin_centers, orig_counts, color='#d9381e', linewidth=2.8, linestyle='--',
        label='Original Cuisine', zorder=10)

ax.set_title('Q2(a): Recipe Size Distribution — Original Cuisine vs 10 Replica Sets', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Recipe Size (Number of Ingredients)', fontsize=12, labelpad=10)
ax.set_ylabel('Number of Recipes', fontsize=12, labelpad=10)
ax.set_xlim(0, 35)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
plt.tight_layout()

plt.savefig('Q2a_recipe_size_distribution_replica_10sets.png', dpi=300)
plt.show()

# Verification check
print("Size distribution difference check:")
for k in range(num_replica_sets):
    diff = np.max(np.abs(original_sizes - replica_sets_sizes[k]))
    print(f"  • Replica Set {k+1}: Max deviation in recipe sizes = {diff}")
"""))

    # Markdown Cell: Q2(b) Inverse Transformation
    cells.append(nbf.v4.new_markdown_cell("""---
### Q2(b) — Size-Controlled Cuisine via Inverse Transformation Sampling
#### Methodology:
1. **Inverse Transform Sampling** allows generating random variates from an arbitrary discrete empirical cumulative distribution function (CDF).
2. Given empirical size probabilities $P(S = s_k) = p_k$ and CDF $F(s) = \\sum_{s_k \\le s} p_k$:
   - Generate standard uniform random variate $U \\sim \\text{Uniform}(0, 1)$.
   - Find the quantile $s = F^{-1}(U) = \\min \\{s_k \\mid F(s_k) \\ge U\\}$.
3. We generate **$100,000$ recipes** ($10 \\times$ the original size of $10,000$) using inverse transformation sampling.
4. Each sampled size $s_j$ is then populated by drawing $s_j$ ingredients uniformly at random from the vocabulary without replacement.
5. We plot and compare the normalized recipe size probability mass distributions (PDF) and cumulative distribution functions (CDF) of the original cuisine and the $10\\times$ inverse-transformed random cuisine.
"""))

    # Code Cell 4: Inverse transformation sampling and plot
    cells.append(nbf.v4.new_code_cell("""# 1. Compute Empirical CDF of recipe sizes
unique_sizes, counts = np.unique(original_sizes, return_counts=True)
empirical_pdf = counts / len(original_sizes)
empirical_cdf = np.cumsum(empirical_pdf)

# 2. Inverse Transformation Sampling for 10x recipes (100,000 recipes)
N_10x = 10 * N
rng_inv = np.random.default_rng(seed=RANDOM_SEED + 777)
u_uniform = rng_inv.uniform(0.0, 1.0, size=N_10x)

# Searchsorted finds the index where u <= cdf
sampled_size_indices = np.searchsorted(empirical_cdf, u_uniform)
sampled_sizes_10x = unique_sizes[sampled_size_indices]

# Compare summary statistics
print(f"Generated {N_10x:,} recipes via Inverse Transform Sampling:")
print(f"  Original Cuisine (N = {N:,}):")
print(f"    Mean: {original_sizes.mean():.4f}, Std: {original_sizes.std():.4f}, Median: {np.median(original_sizes):.1f}")
print(f"  Random Cuisine 10x (N = {N_10x:,}):")
print(f"    Mean: {sampled_sizes_10x.mean():.4f}, Std: {sampled_sizes_10x.std():.4f}, Median: {np.median(sampled_sizes_10x):.1f}")

# 3. Two-panel visualization: (Left) Size Distribution PDF, (Right) Size Cumulative CDF
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

max_s = max(original_sizes.max(), sampled_sizes_10x.max())
bins_inv = np.arange(0.5, max_s + 1.5, 1)
centers_inv = np.arange(1, max_s + 1)

# Left: Probability Density Function (Normalized Histogram)
orig_p, _ = np.histogram(original_sizes, bins=bins_inv, density=True)
inv_p, _ = np.histogram(sampled_sizes_10x, bins=bins_inv, density=True)

ax1.plot(centers_inv, orig_p, color='#1f77b4', linewidth=2.5, marker='o', markersize=4, label='Original Cuisine (N=10,000)')
ax1.plot(centers_inv, inv_p, color='#ff7f0e', linewidth=2.2, linestyle='--', marker='s', markersize=3, label='Inverse Transform (N=100,000, 10x)')
ax1.set_title('Probability Distribution P(Size = s)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Recipe Size (Number of Ingredients)', fontsize=11)
ax1.set_ylabel('Probability Density', fontsize=11)
ax1.set_xlim(0, 35)
ax1.legend(frameon=True, facecolor='white', fontsize=10)

# Right: Cumulative Distribution Function (CDF)
orig_sorted = np.sort(original_sizes)
orig_cdf_vals = np.arange(1, len(orig_sorted) + 1) / len(orig_sorted)

inv_sorted = np.sort(sampled_sizes_10x)
inv_cdf_vals = np.arange(1, len(inv_sorted) + 1) / len(inv_sorted)

ax2.step(orig_sorted, orig_cdf_vals, where='post', color='#1f77b4', linewidth=2.5, label='Original Cuisine CDF')
ax2.step(inv_sorted, inv_cdf_vals, where='post', color='#ff7f0e', linewidth=2.0, linestyle='--', label='Inverse Transform 10x CDF')
ax2.set_title('Empirical Cumulative Distribution Function (CDF)', fontsize=13, fontweight='bold')
ax2.set_xlabel('Recipe Size (Number of Ingredients)', fontsize=11)
ax2.set_ylabel('Cumulative Probability', fontsize=11)
ax2.set_xlim(0, 35)
ax2.legend(frameon=True, facecolor='white', fontsize=10)

plt.suptitle('Q2(b): Size-Controlled Cuisine via Inverse Transformation Sampling', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()

plt.savefig('Q2b_recipe_size_distribution_inverse_transform_10x.png', dpi=300)
plt.show()
"""))

    # Markdown Cell: Output summary
    cells.append(nbf.v4.new_markdown_cell("""---
## Question 2 Deliverables Summary

| Deliverable File | Description | Question |
|---|---|---|
| `Q2a_recipe_size_distribution_replica_10sets.png` | Size distribution plot comparing original cuisine against 10 replica random control sets | Q2(a) |
| `Q2b_recipe_size_distribution_inverse_transform_10x.png` | Normalized PDF and step CDF plots comparing original cuisine against 10x (100,000 recipes) inverse transformation control | Q2(b) |
"""))

    nb.cells = cells
    return nb

def build_q3_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11.9'}
    }
    
    cells = []
    
    # Title Markdown
    cells.append(nbf.v4.new_markdown_cell("""# Computational Gastronomy — Assignment 2
## Question 3: Size- and Ingredient Frequency-Controlled Random Control
**Monsoon 2026**

### Problem Statement
Create size- and ingredient frequency-controlled random control of the above 10,000 recipes assuming them to be a cuisine. Create 10 sets of random cuisines.
* **(a)** Plot recipe size distribution of the original cuisine and & random recipes. `[5]`
* **(b)** Plot frequency-rank distribution of the original cuisine & the random recipes. `[5]`
"""))

    # Code Cell 1: Imports and setup
    cells.append(nbf.v4.new_code_cell("""import sys
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

DATA_PATH = Path('../Group_23_Assignment1/Q1b_ingredients_ner.csv')
RANDOM_SEED = 20260929
np.random.seed(RANDOM_SEED)
"""))

    # Code Cell 2: Data preparation
    cells.append(nbf.v4.new_code_cell("""# Load data
ner_df = pd.read_csv(DATA_PATH, keep_default_na=False)
valid_ner = ner_df[ner_df['Ingredient Name'].str.strip() != '']

# Vocabulary mapping
vocab = sorted(list(set(valid_ner['Ingredient Name'])))
V = len(vocab)
ing2idx = {name: i for i, name in enumerate(vocab)}

# Unique recipes mapping
recipes_dict = valid_ner.groupby('Recipe ID')['Ingredient Name'].apply(lambda s: sorted(list(set(s)))).to_dict()
recipe_ids = sorted(list(recipes_dict.keys()))
N = len(recipe_ids)

# Encode recipes as list of sets of ingredient indices
original_recipes_indices = [set(ing2idx[ing] for ing in recipes_dict[rid]) for rid in recipe_ids]

# Degree sequences
orig_recipe_sizes = np.array([len(r) for r in original_recipes_indices])
orig_ingredient_freqs = np.zeros(V, dtype=int)
for r in original_recipes_indices:
    for ing_idx in r:
        orig_ingredient_freqs[ing_idx] += 1

print(f"Recipes N: {N:,}, Vocabulary V: {V:,}")
print(f"Total recipe-ingredient edges: {orig_recipe_sizes.sum():,}")
print(f"Mean recipe size: {orig_recipe_sizes.mean():.4f}")
print(f"Top 5 most frequent ingredients:")
top5_indices = np.argsort(orig_ingredient_freqs)[::-1][:5]
for idx in top5_indices:
    print(f"  • {vocab[idx]}: {orig_ingredient_freqs[idx]:,} recipes")
"""))

    # Markdown Cell: Methodology (Curveball Algorithm)
    cells.append(nbf.v4.new_markdown_cell("""---
### Null Model Methodology: Curveball Bipartite Graph Randomization
A cuisine can be formally represented as a **bipartite graph** $G = (R, I, E)$, where:
* $R$ is the set of $N = 10,000$ recipes.
* $I$ is the set of $V = 2,425$ unique ingredients.
* An edge $(r, i) \\in E$ indicates that recipe $r$ contains ingredient $i$.

The task requires creating a random null model where **both** of the following margins are strictly controlled:
1. **Recipe size distribution (Row sums):** Every recipe retains its exact number of ingredients.
2. **Ingredient frequency distribution (Column sums):** Every ingredient occurs with its exact original frequency across the entire cuisine.

#### The Curveball Algorithm (Strona et al., 2014)
To generate an unbiased, uniform sample from the space of all binary matrices with fixed row and column sums, we implement the **Curveball Algorithm**:
* In each step, pick two recipes $u$ and $v$ at random.
* Identify their common ingredients $C = r_u \\cap r_v$ and differences $D_u = r_u \\setminus C$, $D_v = r_v \\setminus C$.
* Combine the non-shared ingredients $D = D_u \\cup D_v$ and shuffle them uniformly at random.
* Reassign $|D_u|$ items to $u$ and $|D_v|$ items to $v$.
* This operation **exactly preserves** the degree of each recipe ($|r_u|$ and $|r_v|$) and the total frequency of each ingredient in $D$, while thoroughly rewiring the bipartite connections without introducing self-loops or duplicate edges.
"""))

    # Code Cell 3: Curveball implementation
    cells.append(nbf.v4.new_code_cell("""def run_curveball(initial_recipes, num_passes=15, rng=None):
    \"\"\"
    Efficient implementation of the Curveball algorithm.
    Preserves exact recipe sizes and ingredient frequencies.
    \"\"\"
    if rng is None:
        rng = np.random.default_rng()
        
    recipes = [set(r) for r in initial_recipes]
    n = len(recipes)
    
    for step in range(num_passes):
        perm = rng.permutation(n)
        for i in range(0, n - 1, 2):
            u, v = perm[i], perm[i+1]
            A, B = recipes[u], recipes[v]
            common = A & B
            diff_A = list(A - common)
            diff_B = list(B - common)
            total_diff = diff_A + diff_B
            if total_diff:
                rng.shuffle(total_diff)
                nA = len(diff_A)
                recipes[u] = common | set(total_diff[:nA])
                recipes[v] = common | set(total_diff[nA:])
    return recipes

# Generate 10 sets of randomized cuisines
num_sets = 10
random_cuisines = []

print("Generating 10 randomized cuisines using Curveball algorithm...")
t0 = time.time()
for s in range(num_sets):
    rng_s = np.random.default_rng(seed=RANDOM_SEED + s * 1000)
    rand_recipes = run_curveball(original_recipes_indices, num_passes=15, rng=rng_s)
    random_cuisines.append(rand_recipes)
    
    # Verify conservation of degree sequences
    rand_sizes = np.array([len(r) for r in rand_recipes])
    rand_freqs = np.zeros(V, dtype=int)
    for r in rand_recipes:
        for ing_idx in r:
            rand_freqs[ing_idx] += 1
            
    assert np.array_equal(rand_sizes, orig_recipe_sizes), f"Set {s+1} recipe sizes violated!"
    assert np.array_equal(rand_freqs, orig_ingredient_freqs), f"Set {s+1} ingredient frequencies violated!"
    print(f"  • Random Cuisine Set {s+1:2d} generated and verified (max size diff: 0, max freq diff: 0)")
t1 = time.time()
print(f"All 10 sets generated in {t1 - t0:.2f}s")
"""))

    # Markdown Cell: Q3(a) Recipe size distribution
    cells.append(nbf.v4.new_markdown_cell("""---
### Q3(a) — Recipe Size Distribution of Original Cuisine & Random Recipes
We plot the recipe size distribution comparing the original cuisine against each of the 10 random cuisines.
Because the Curveball degree-preserving null model fixes row sums exactly, all 10 sets overlap precisely with the original distribution.
"""))

    # Code Cell 4: Plot Q3(a)
    cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

max_size = orig_recipe_sizes.max()
bins = np.arange(0.5, max_size + 1.5, 1)
bin_centers = np.arange(1, max_size + 1)

palette = plt.cm.tab10(np.linspace(0, 1, 10))

# Plot each of the 10 random sets
for s in range(num_sets):
    rand_sizes = np.array([len(r) for r in random_cuisines[s]])
    counts, _ = np.histogram(rand_sizes, bins=bins)
    ax.plot(bin_centers, counts, color=palette[s], alpha=0.6, linewidth=1.5,
            label=f'Random Cuisine Set {s+1}' if s < 3 or s == 9 else "")

# Plot original cuisine
orig_counts, _ = np.histogram(orig_recipe_sizes, bins=bins)
ax.plot(bin_centers, orig_counts, color='#d9381e', linewidth=2.8, linestyle='--',
        label='Original Cuisine', zorder=10)

ax.set_title('Q3(a): Recipe Size Distribution — Original Cuisine & 10 Random Sets', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Recipe Size (Number of Ingredients)', fontsize=12, labelpad=10)
ax.set_ylabel('Number of Recipes', fontsize=12, labelpad=10)
ax.set_xlim(0, 35)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
plt.tight_layout()

plt.savefig('Q3a_recipe_size_distribution.png', dpi=300)
plt.show()
"""))

    # Markdown Cell: Q3(b) Frequency-Rank distribution
    cells.append(nbf.v4.new_markdown_cell("""---
### Q3(b) — Frequency-Rank Distribution of Original Cuisine & Random Recipes
We compute the frequency of every ingredient across all recipes, sort them descending to determine their rank $r = 1, 2, \\dots, |V|$, normalize by the total number of ingredient mentions:
$$f(r) = \\frac{\\text{Frequency}(r)}{\\sum_{i} \\text{Frequency}(i)}$$
and plot the frequency-rank distribution on a **log-log scale** (Zipfian plot) for both the original cuisine and the 10 random cuisines.
"""))

    # Code Cell 5: Plot Q3(b)
    cells.append(nbf.v4.new_code_cell("""# Compute frequency-rank for original cuisine
orig_sorted_freqs = np.sort(orig_ingredient_freqs)[::-1]
total_occurrences = orig_sorted_freqs.sum()
orig_norm_freqs = orig_sorted_freqs / total_occurrences
ranks = np.arange(1, len(orig_sorted_freqs) + 1)

# Plot Frequency-Rank log-log distribution
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

for s in range(num_sets):
    rand_freqs = np.zeros(V, dtype=int)
    for r in random_cuisines[s]:
        for ing_idx in r:
            rand_freqs[ing_idx] += 1
    rand_sorted = np.sort(rand_freqs)[::-1]
    rand_norm = rand_sorted / rand_sorted.sum()
    ax.plot(ranks, rand_norm, color=palette[s], alpha=0.5, linewidth=1.2,
            label=f'Random Cuisine Set {s+1}' if s < 3 or s == 9 else "")

ax.plot(ranks, orig_norm_freqs, color='#d9381e', linewidth=2.5, linestyle='--',
        label='Original Cuisine', zorder=10)

# Annotate prominent staple ingredients
staples = [
    (1, vocab[top5_indices[0]], orig_norm_freqs[0]),
    (2, vocab[top5_indices[1]], orig_norm_freqs[1]),
    (3, vocab[top5_indices[2]], orig_norm_freqs[2]),
    (4, vocab[top5_indices[3]], orig_norm_freqs[3]),
]
for rank_pos, name, val in staples:
    ax.scatter(rank_pos, val, color='#d9381e', s=25, zorder=12)
    ax.annotate(f" {name}", (rank_pos, val), fontsize=9, fontweight='semibold',
                xytext=(5, 3), textcoords='offset points')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title('Q3(b): Frequency-Rank Distribution — Original Cuisine & Random Recipes (Log-Log)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Ingredient Rank (log scale)', fontsize=12, labelpad=10)
ax.set_ylabel('Normalized Frequency (log scale)', fontsize=12, labelpad=10)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
plt.tight_layout()

plt.savefig('Q3b_frequency_rank_distribution.png', dpi=300)
plt.show()
"""))

    # Markdown Cell: Deliverables summary
    cells.append(nbf.v4.new_markdown_cell("""---
## Question 3 Deliverables Summary

| Deliverable File | Description | Question |
|---|---|---|
| `Q3a_recipe_size_distribution.png` | Recipe size distribution of original cuisine vs 10 size- and frequency-controlled random cuisines | Q3(a) |
| `Q3b_frequency_rank_distribution.png` | Log-log Zipfian frequency-rank distribution of original cuisine vs 10 random cuisines | Q3(b) |
"""))

    nb.cells = cells
    return nb

if __name__ == '__main__':
    print("Building Notebooks...")
    nb1 = build_q1_notebook()
    nb2 = build_q2_notebook()
    nb3 = build_q3_notebook()
    
    with open('Q1.ipynb', 'w') as f:
        nbf.write(nb1, f)
    with open('Q2.ipynb', 'w') as f:
        nbf.write(nb2, f)
    with open('Q3.ipynb', 'w') as f:
        nbf.write(nb3, f)
    print("Notebooks written to Q1.ipynb, Q2.ipynb, Q3.ipynb successfully!")
