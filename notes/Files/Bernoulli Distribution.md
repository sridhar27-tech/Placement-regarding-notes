# Bernoulli Distribution in Machine Learning

## 1. What is the Bernoulli Distribution?

The **Bernoulli Distribution** is the **simplest discrete probability distribution**. It models the outcome of a **single trial** (experiment) that has exactly **two possible outcomes**: success (1) or failure (0).

It is the building block for many other distributions (especially the Binomial) and is fundamental in binary classification problems in machine learning.

### Key Characteristics:
- Only **two possible values**: 0 (failure) or 1 (success)
- Defined by a single parameter **p** (probability of success)
- Each trial is independent
- **Mean (Expected Value)**:  
  $$
  \mu = p
  $$
- **Variance**:  
  $$
  \sigma^2 = p(1-p)
  $$

---

## 2. Probability Mass Function (PMF)

$$
P(X = k) = 
\begin{cases} 
p & \text{if } k = 1 \\
1-p & \text{if } k = 0 
\end{cases}
$$

Or more compactly:

$$
P(X = k) = p^k (1-p)^{1-k} \quad \text{for } k \in \{0, 1\}
$$

---

## 3. Mathematical Example

**Scenario**:  
A biased coin has a probability of landing heads (success) of **p = 0.65**.

#### Questions & Solutions:

**a)** Probability of getting heads (success) on a single flip?

$$
P(X=1) = 0.65 \quad (65\%)
$$

**b)** Probability of getting tails (failure)?

$$
P(X=0) = 1 - 0.65 = 0.35 \quad (35\%)
$$

**c)** Expected outcome if we define heads as 1?

$$
E[X] = p = 0.65
$$

**d)** Variance of the outcome?

$$
\text{Var}(X) = p(1-p) = 0.65 \times 0.35 = 0.2275
$$

**Python Code**:

```python
from scipy.stats import bernoulli
import numpy as np

p = 0.65

# Probability mass
print("P(X=1) =", bernoulli.pmf(1, p))
print("P(X=0) =", bernoulli.pmf(0, p))

# Generate 10,000 simulated flips (common in ML simulations)



from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate binary classification data
X, y = make_classification(n_samples=1000, random_state=42)

model = LogisticRegression()
model.fit(X, y)

# Predict probability (Bernoulli parameter p)
new_sample = X[0:1]
prob_spam = model.predict_proba(new_sample)[0][1]
print(f"Predicted P(Spam) = {prob_spam:.4f}")
samples = bernoulli.rvs(p, size=10000)
print("Empirical mean:", np.mean(samples))
print("Empirical variance:", np.var(samples))


The Bernoulli Distribution is the fundamental distribution for modeling single binary events. It serves as the cornerstone for many machine learning algorithms, especially in binary classification tasks like Logistic Regression and Bernoulli Naive Bayes. Its simplicity makes it easy to understand and implement, while its connection to more complex distributions (Binomial, Normal) makes it essential for probabilistic modeling in ML.
