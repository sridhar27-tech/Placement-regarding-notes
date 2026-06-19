# Probability Distributions in Machine Learning

## 1. What is a Probability Distribution?

A **Probability Distribution** describes how probabilities are distributed over the possible outcomes of a random variable. It is a fundamental concept in statistics and machine learning that allows us to model uncertainty, make predictions, and perform inference.

Probability distributions tell us:
- What outcomes are possible
- How likely each outcome (or range of outcomes) is
- Key summary statistics (mean, variance, etc.)

### Two Main Types:

**1. Discrete Probability Distributions**  
- For countable outcomes (integers)  
- Examples: Binomial, Poisson, Bernoulli, Categorical  
- Use **Probability Mass Function (PMF)**

**2. Continuous Probability Distributions**  
- For uncountable outcomes (real numbers)  
- Examples: Normal (Gaussian), Uniform, Exponential, t-distribution  
- Use **Probability Density Function (PDF)**

---

## 2. Key Functions

### Probability Mass Function (PMF) – Discrete
$$
P(X = k)
$$
Gives the probability that the discrete random variable equals exactly *k*.

### Probability Density Function (PDF) – Continuous
$$
f(x)
$$
Gives the relative likelihood of the random variable taking value *x*. The area under the curve equals 1.

### Cumulative Distribution Function (CDF)
For both types:
$$
F(x) = P(X \leq x)
$$
- Probability that the variable is less than or equal to *x*
- Always ranges from 0 to 1

### Expected Value (Mean) and Variance
- **Mean (μ)**: Expected average outcome
- **Variance (σ²)**: Measure of spread

---

## 3. Mathematical Example

**Scenario**:  
A fair six-sided die is rolled **twice**. Let X be the sum of the two rolls.

X follows a discrete probability distribution (range: 2 to 12).

#### a) Probability of getting a sum of exactly 7?

Possible outcomes: (1,6), (2,5), (3,4), (4,3), (5,2), (6,1) → 6 ways

Total possible outcomes: 6 × 6 = 36

$$
P(X=7) = \frac{6}{36} = \frac{1}{6} \approx 0.1667
$$

#### b) Probability that the sum is 8 or more? (using CDF)

$$
P(X \geq 8) = 1 - P(X \leq 7)
$$

(Computed by summing individual probabilities: 15/36 ≈ 0.4167)

**Python Code**:

```python
import numpy as np
from scipy.stats import randint

# Simulate rolling two dice
rolls = np.random.randint(1, 7, size=(10000, 2))
sums = rolls.sum(axis=1)
