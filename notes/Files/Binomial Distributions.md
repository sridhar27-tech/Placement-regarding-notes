# Binomial Distribution in Machine Learning

## 1. What is the Binomial Distribution?

The **Binomial Distribution** is a **discrete probability distribution** that models the number of **successes** in a fixed number of **independent Bernoulli trials** (yes/no experiments), each having the same probability of success.

It is widely used when outcomes are binary (0 or 1, success or failure).

### Key Characteristics:
- **Discrete** (countable outcomes: 0 to n successes)
- Fixed number of trials **n**
- Each trial is independent
- Constant probability of success **p** per trial
- Two parameters: **n** (trials) and **p** (success probability)

**Mean (Expected Value)**:  
$$
\mu = np
$$

**Variance**:  
$$
\sigma^2 = np(1-p)
$$

---

## 2. Probability Mass Function (PMF)

The probability of getting exactly **k** successes in **n** trials is:

$$
P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}
$$

Where:
- $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ (Binomial coefficient)
- $k = 0, 1, 2, ..., n$

---

## 3. Mathematical Example

**Scenario**:  
A company runs an A/B test on a new website button. The current click-through rate is **p = 0.12** (12%). They show the new button to **n = 20** randomly selected users.

#### Questions & Solutions:

**a)** What is the probability of getting **exactly 4 clicks**?

$$
P(X=4) = \binom{20}{4} (0.12)^4 (0.88)^{16}
$$

$$
\binom{20}{4} = 4845
$$

$$
P(X=4) \approx 4845 \times 0.00020736 \times 0.118 \approx 0.118 \quad (11.8\%)
$$

**b)** What is the probability of getting **3 or fewer clicks**?

$$
P(X \leq 3) = P(0) + P(1) + P(2) + P(3)
$$

(This is usually computed using cumulative distribution functions in practice.)

**c)** Expected number of clicks:

$$
\mu = np = 20 \times 0.12 = 2.4
$$

**Python Code for Calculation**:

```python
from scipy.stats import binom

n = 20
p = 0.12

# Probability of exactly 4 successes
prob_exact = binom.pmf(4, n, p)
print(f"P(X=4) = {prob_exact:.4f}")

# Probability of 3 or fewer
prob_leq3 = binom.cdf(3, n, p)
print(f"P(X ≤ 3) = {prob_leq3:.4f}")

# Generate random samples (common in simulations)
samples = binom.rvs(n, p, size=10000)
