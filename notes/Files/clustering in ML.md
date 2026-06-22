# Sampling Techniques in Statistics and Machine Learning

**Extended Comprehensive Guide with Mathematical Examples and Python Code**

Sampling is crucial for statistical inference, efficient model training, and handling large-scale data in machine learning.

---

## Table of Contents
1. [Introduction to Sampling](#introduction-to-sampling)
2. [Key Sampling Techniques](#key-sampling-techniques)
3. [Simple Random Sampling](#simple-random-sampling)
4. [Stratified Sampling](#stratified-sampling)
5. [Cluster Sampling](#cluster-sampling)
6. [Systematic Sampling](#systematic-sampling)
7. [Bootstrap Sampling](#bootstrap-sampling)
8. [Reservoir Sampling](#reservoir-sampling)
9. [Advanced Sampling for Imbalanced Data](#advanced-sampling-for-imbalanced-data)
10. [Comparison of Sampling Methods](#comparison-of-sampling-methods)
11. [Sampling in Machine Learning Practice](#sampling-in-machine-learning-practice)
12. [Conclusion](#conclusion)

---

## Introduction to Sampling

**Population** vs **Sample**  
- **Population**: Entire set of interest.  
- **Sample**: Subset used for analysis.

**Main Goals**:
- Reduce computational cost
- Enable statistical inference
- Minimize sampling bias and variance

**Types of Sampling**:
- **Probability Sampling** (preferred for inference)
- **Non-Probability Sampling**

---

## Simple Random Sampling (SRS)

Every unit has an equal probability of being selected.

**Python Example**

```python
import numpy as np

np.random.seed(42)
population = np.random.normal(100, 20, 50000)

sample_size = 1000
srs_sample = np.random.choice(population, size=sample_size, replace=False)

print("Population Mean:", population.mean().round(3))
print("SRS Sample Mean:", srs_sample.mean().round(3))
Stratified Sampling
Divides the population into homogeneous strata, then takes random samples from each stratum proportionally.
Mathematical Formula:
For ( K ) strata:
$$n_h = n \cdot \frac{N_h}{N}, \quad h = 1,2,\dots,K$$
$$\bar{x}_{str} = \sum_{h=1}^{K} W_h \bar{x}_h, \quad W_h = \frac{N_h}{N}$$
Extended Python Example
PythonCopyimport pandas as pd
import numpy as np

# Create realistic dataset
np.random.seed(42)
n = 100000
df = pd.DataFrame({
    'age_group': np.random.choice(['18-30', '31-45', '46-60', '60+'], n, p=[0.3, 0.35, 0.25, 0.1]),
    'income': np.random.normal(60000, 20000, n),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n)
})

def stratified_sample(df, strata_cols, sample_size, random_state=42):
    df = df.copy()
    df['_strata'] = df[strata_cols].astype(str).agg('_'.join, axis=1)
    sample = df.groupby('_strata', group_keys=False).apply(
        lambda x: x.sample(min(len(x), max(1, int(sample_size * len(x)/len(df)))), random_state=random_state)
    )
    return sample.drop(columns=['_strata'])

strat_sample = stratified_sample(df, ['age_group', 'region'], 2000)

print("Population Age Distribution:\n", df['age_group'].value_counts(normalize=True))
print("\nSample Age Distribution:\n", strat_sample['age_group'].value_counts(normalize=True))

Cluster Sampling
Population is divided into clusters. Randomly select entire clusters (or samples within them).
Types:

One-stage: Take all units from selected clusters
Two-stage: Take a sample from selected clusters

Python Example (Two-stage Cluster Sampling)
PythonCopyimport numpy as np
import pandas as pd

np.random.seed(42)
# 500 clusters
clusters = []
for i in range(500):
    size = np.random.randint(50, 300)
    cluster = pd.DataFrame({
        'cluster_id': i,
        'value': np.random.normal(50 + (i % 30), 12, size)
    })
    clusters.append(cluster)

population = pd.concat(clusters, ignore_index=True)

# Two-stage Cluster Sampling
n_clusters = 40
selected_clusters = np.random.choice(population['cluster_id'].unique(), n_clusters, replace=False)

cluster_sample = population[population['cluster_id'].isin(selected_clusters)]

# Second stage: sample 30% from each selected cluster
final_sample = cluster_sample.groupby('cluster_id', group_keys=False).apply(
    lambda x: x.sample(frac=0.3, random_state=42)
)

print("Population Mean:", population['value'].mean().round(3))
print("Cluster Sample Mean:", final_sample['value'].mean().round(3))

Systematic Sampling
Select every ( k )-th unit after a random starting point.
PythonCopydef systematic_sampling(data, sample_size):
    N = len(data)
    k = N // sample_size
    start = np.random.randint(0, k)
    indices = np.arange(start, N, k)[:sample_size]
    return data.iloc[indices] if isinstance(data, pd.DataFrame) else data[indices]

# Usage
# sys_sample = systematic_sampling(df, 2000)

Bootstrap Sampling
Sampling with replacement to estimate variability and confidence intervals.
PythonCopydef bootstrap_mean(data, n_bootstraps=10000):
    bootstraps = np.random.choice(data, size=(n_bootstraps, len(data)), replace=True)
    means = bootstraps.mean(axis=1)
    return means.mean(), np.percentile(means, [2.5, 97.5])

sample = np.random.normal(100, 15, 500)
mean_est, ci = bootstrap_mean(sample)

print("Bootstrap Mean:", mean_est.round(3))
print("95% Confidence Interval:", ci.round(3))

Reservoir Sampling
Useful for streaming data or unknown population size.
PythonCopyimport random

def reservoir_sampling(stream, k):
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir

Advanced Sampling for Imbalanced Data
PythonCopyfrom imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=10000, weights=[0.9, 0.1], random_state=42)

print("Original class distribution:", np.bincount(y))

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

print("After SMOTE:", np.bincount(y_res))

Comparison of Sampling Methods

TechniqueBiasVarianceCostBest Use CaseSimple RandomLowMediumMediumHomogeneous dataStratifiedVery LowLowMediumHeterogeneous / ClassificationClusterMediumHigherLowGeographically grouped dataSystematicLowLowLowOrdered listsBootstrapLowControlledMediumEnsembles & UncertaintyReservoirLowLowLowStreaming / Big Data

Sampling in Machine Learning Practice

Use stratify=y in train_test_split
Use StratifiedKFold for cross-validation
Bootstrap in Bagging / Random Forest
SMOTE / ADASYN for imbalanced classes
TimeSeriesSplit for time series data


Conclusion
Mastering sampling techniques leads to more robust, efficient, and statistically sound machine learning models.
Key Takeaway:
The quality of your model depends heavily on the quality of your sample.
Recommended Resources:

The Elements of Statistical Learning (Hastie et al.)
Sampling: Design and Analysis (Sharon Lohr)
