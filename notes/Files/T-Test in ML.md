# T-Test in Machine Learning: Complete Guide

## 1. Introduction

The **T-Test** (Student's T-Test) is a statistical hypothesis test used to determine if there is a significant difference between the means of two groups when the **population standard deviation is unknown** and/or the **sample size is small** (n < 30).

It is one of the most commonly used statistical tests in Machine Learning and Data Science.

---

## 2. When to Use T-Test in ML

- Comparing performance of two ML models on small validation sets
- A/B testing with limited samples
- Testing if a new feature improves model performance (small experiments)
- Comparing before/after metrics in model experiments
- Checking statistical significance in hyperparameter tuning results

---

## 3. Types of T-Tests

1. **One-Sample T-Test** — Compare sample mean to a known value
2. **Independent Two-Sample T-Test** — Compare means of two independent groups
3. **Paired T-Test** — Compare means of the same group at different times

---

## 4. Mathematical Foundation

### T-Statistic (One-Sample)

$$
t = \frac{\bar{x} - \mu_0}{\frac{s}{\sqrt{n}}}
$$

Where:
- $\bar{x}$ = Sample mean
- $\mu_0$ = Hypothesized population mean
- $s$ = Sample standard deviation
- $n$ = Sample size

**Degrees of Freedom (df)** = $n - 1$

---

## 5. Detailed Numerical Example

**Problem**:  
A new ML model was tested on 25 runs and achieved an average accuracy of **91.2%**.  
The baseline model has a known mean accuracy of **89%**.  
Sample standard deviation = **2.8%**

**Calculation**:

$$
t = \frac{91.2 - 89}{\frac{2.8}{\sqrt{25}}} = \frac{2.2}{0.56} \approx 3.929
$$

**Degrees of Freedom** = 24  
**Critical t-value** (two-tailed, α=0.05) ≈ **2.064**

Since 3.929 > 2.064 → **Reject Null Hypothesis**. The new model performs significantly better.

---

## 6. Python Implementation

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def one_sample_t_test(sample_data, pop_mean, alpha=0.05, alternative='two-sided'):
    """Perform One-Sample T-Test"""
    t_stat, p_value = stats.ttest_1samp(sample_data, pop_mean, alternative=alternative)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'sample_mean': np.mean(sample_data),
        'sample_size': len(sample_data),
        'df': len(sample_data) - 1,
        'reject_h0': p_value < alpha
    }

def two_sample_t_test(group1, group2, alpha=0.05, equal_var=True):
    """Independent Two-Sample T-Test"""
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'mean1': np.mean(group1),
        'mean2': np.mean(group2),
        'reject_h0': p_value < alpha
    }

# ====================== Example Usage ======================
np.random.seed(42)

# Simulate model accuracies
baseline = np.random.normal(0.89, 0.028, 25)
new_model = np.random.normal(0.912, 0.025, 25)

result = one_sample_t_test(new_model, pop_mean=0.89)

print("T-Test Results:")
print(f"T-Statistic     : {result['t_statistic']:.4f}")
print(f"P-Value         : {result['p_value']:.6f}")
print(f"Sample Mean     : {result['sample_mean']:.4f}")
print(f"Degrees of Freedom: {result['df']}")
print(f"Reject H0       : {result['reject_h0']}")

# Visualization
plt.figure(figsize=(10, 6))
stats.probplot(new_model, dist="norm", plot=plt)
plt.title("Q-Q Plot for Normality Check (T-Test Assumption)")
plt.show()
