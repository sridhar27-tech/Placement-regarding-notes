# Sampling Techniques in Statistics and Machine Learning

**A Comprehensive Guide with Mathematical Examples and Python Code**

Sampling is a fundamental concept in statistics and machine learning. It allows us to draw conclusions about a large population from a smaller subset of data, saving time and computational resources while maintaining statistical validity.

---

## Table of Contents
1. [Introduction to Sampling](#introduction-to-sampling)
2. [Key Sampling Techniques](#key-sampling-techniques)
3. [Simple Random Sampling](#simple-random-sampling)
4. [Stratified Sampling](#stratified-sampling)
5. [Cluster Sampling](#cluster-sampling)
6. [Systematic Sampling](#systematic-sampling)
7. [Comparison of Sampling Methods](#comparison-of-sampling-methods)
8. [Sampling in Machine Learning](#sampling-in-machine-learning)
9. [Conclusion](#conclusion)

---

## Introduction to Sampling

**Population** vs **Sample**:
- Population: Complete set of all possible observations.
- Sample: Subset of the population used for analysis.

**Goal**: Obtain a representative sample that minimizes **sampling bias** and **sampling error**.

### Key Concepts
- **Sampling Frame**: List of all units in the population.
- **Sampling Error**: Difference between sample statistic and population parameter.
- **Bias**: Systematic error in sampling method.

---

## Key Sampling Techniques

### Probability Sampling (Random)
- Every unit has a known, non-zero probability of being selected.
- Allows statistical inference.

### Non-Probability Sampling
- Convenience, Purposive, Snowball, etc. (Not covered in depth here).

---

## Simple Random Sampling (SRS)

**Definition**: Every member of the population has an equal chance of being selected.

**Mathematical Representation**:
For a population of size \( N \), sample of size \( n \):

Probability of selecting any unit = \( \frac{n}{N} \)

**Sample Mean**:
$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

**Python Implementation**

```python
import numpy as np
import pandas as pd

# Create sample population
np.random.seed(42)
population = np.random.normal(50, 15, 10000)  # 10,000 data points

# Simple Random Sampling
sample_size = 500
simple_random_sample = np.random.choice(population, size=sample_size, replace=False)

print("Population Mean:", population.mean())
print("Sample Mean (SRS):", simple_random_sample.mean())
print("Sampling Error:", abs(population.mean() - simple_random_sample.mean()))


Stratified Sampling
Definition: Population is divided into homogeneous subgroups (strata) based on a key characteristic, then random samples are drawn from each stratum proportional to its size.
Best Used When: Population has distinct subgroups (e.g., age groups, gender, classes in classification).
Mathematical Formula:
Let there be ( K ) strata. For stratum ( h ):

Stratum size: ( N_h )
Sample size from stratum: ( n_h = n \times \frac{N_h}{N} )

Stratified Sample Mean:
$$\bar{x}_{str} = \sum_{h=1}^{K} W_h \bar{x}_h \quad \text{where} \quad W_h = \frac{N_h}{N}$$
Python Example (Stratified Sampling)
PythonCopyimport pandas as pd
import numpy as np

# Create population with strata (e.g., customer segments)
data = {
    'id': range(10000),
    'segment': np.random.choice(['Low', 'Medium', 'High'], 10000, p=[0.5, 0.3, 0.2]),
    'value': np.random.normal(100, 30, 10000)
}
df = pd.DataFrame(data)

# Stratified Sampling
def stratified_sample(df, strata_col, sample_size):
    strata = df[strata_col].unique()
    sample = pd.DataFrame()
    n_per_strata = sample_size // len(strata)
    
    for stratum in strata:
        stratum_df = df[df[strata_col] == stratum]
        sample_stratum = stratum_df.sample(n=min(n_per_strata, len(stratum_df)), random_state=42)
        sample = pd.concat([sample, sample_stratum])
    return sample

strat_sample = stratified_sample(df, 'segment', 600)

print("Original Segment Distribution:\n", df['segment'].value_counts(normalize=True))
print("\nSample Segment Distribution:\n", strat_sample['segment'].value_counts(normalize=True))

Cluster Sampling
Definition: Population is divided into clusters (groups). A random sample of clusters is selected, and then all units within the selected clusters are included (one-stage) or a sample is taken (two-stage).
Best Used When: Population is geographically spread out or natural clusters exist (e.g., schools, cities, warehouses).
Mathematical Representation:

Number of clusters in population: ( M )
Number of clusters selected: ( m )
Average cluster size: ( \bar{N} )

Two-stage Cluster Sampling is common in practice.
Python Example (Cluster Sampling)
PythonCopyimport numpy as np
import pandas as pd

# Simulate population with clusters (e.g., 100 schools, each with ~100 students)
np.random.seed(42)
clusters = []
for i in range(100):  # 100 clusters
    cluster_size = np.random.randint(80, 120)
    cluster_data = pd.DataFrame({
        'cluster_id': [i] * cluster_size,
        'score': np.random.normal(70 + i%10, 10, cluster_size)
    })
    clusters.append(cluster_data)

population = pd.concat(clusters, ignore_index=True)

# One-stage Cluster Sampling
num_clusters_to_sample = 15
selected_clusters = np.random.choice(population['cluster_id'].unique(), 
                                   size=num_clusters_to_sample, replace=False)

cluster_sample = population[population['cluster_id'].isin(selected_clusters)]

print("Population Mean Score:", population['score'].mean())
print("Cluster Sample Mean Score:", cluster_sample['score'].mean())
print("Sample Size:", len(cluster_sample))

Systematic Sampling
Definition: Select every ( k )-th member from the population after a random starting point.
Formula:
$k = \frac{N}{n}$
Python Implementation
PythonCopydef systematic_sample(population, sample_size):
    N = len(population)
    k = N // sample_size
    start = np.random.randint(0, k)
    indices = np.arange(start, N, k)[:sample_size]
    return population[indices]

# Using earlier population
sys_sample = systematic_sample(population['score'].values, 500)
print("Systematic Sample Mean:", sys_sample.mean())

Comparison of Sampling Methods








































MethodBias RiskCostRepresentativenessBest ForSimple RandomLowMediumGoodHomogeneous populationsStratifiedVery LowMediumExcellentHeterogeneous populationsClusterMediumLowModerateGeographically dispersedSystematicLow-MediumLowGoodOrdered lists
Advantages of Stratified:

More precise than SRS when strata are internally homogeneous.

Advantages of Cluster:

Cheaper and more practical for large-scale surveys.


Sampling in Machine Learning

Train-Test Split (Simple Random)
Stratified K-Fold Cross Validation (for imbalanced classification)
Bootstrap Sampling (Bagging, Random Forest)
Undersampling / Oversampling in imbalanced datasets
Reservoir Sampling for streaming data

Stratified Train-Test Split Example
PythonCopyfrom sklearn.model_selection import train_test_split

# Assuming df has target column
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Target distribution preserved:")
print(y_train.value_counts(normalize=True))

Conclusion
Choosing the right sampling technique is critical for reliable statistical inference and robust machine learning models.

Use Stratified Sampling when dealing with important subgroups.
Use Cluster Sampling when cost and logistics are major constraints.
Always validate your sample against known population characteristics.

Key Takeaway:
A well-designed sample leads to better models, more accurate predictions, and trustworthy insights.
Recommended Reading:

"The Elements of Statistical Learning"
"Sampling: Design and Analysis" by Sharon Lohr

"Sampling: Design and Analysis" by Sharon Lohr
