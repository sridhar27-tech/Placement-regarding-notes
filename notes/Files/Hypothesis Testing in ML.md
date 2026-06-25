# Hypothesis Testing in Machine Learning

**A Comprehensive Guide**

---

## 📌 Introduction

**Hypothesis Testing** is a fundamental statistical technique used in Machine Learning and Data Science to make inferences about populations based on sample data. It helps us determine whether observed effects are statistically significant or just due to random chance.

In ML, hypothesis testing is commonly used for:
- Feature selection
- Model evaluation and comparison
- A/B testing
- Detecting data drift
- Validating assumptions

---

## 🧠 Key Concepts

### 1. Null Hypothesis (H₀)
- The default assumption that there is **no effect** or **no difference**.
- Example: "The new model has the same accuracy as the old model."

### 2. Alternative Hypothesis (H₁ or Hₐ)
- The opposite of the null hypothesis.
- Example: "The new model performs better than the old model."

### 3. Significance Level (α)
- Usually set to **0.05** (5%).
- The probability of rejecting the null hypothesis when it is actually true (Type I Error).

### 4. p-value
- The probability of observing the test results (or more extreme) assuming the null hypothesis is true.
- If **p-value < α**, we reject H₀.

### 5. Types of Errors
- **Type I Error**: Rejecting H₀ when it is true (False Positive)
- **Type II Error**: Failing to reject H₀ when it is false (False Negative)

---

## 🔢 Steps in Hypothesis Testing

1. State the Null and Alternative Hypotheses
2. Choose the significance level (α)
3. Select the appropriate statistical test
4. Calculate the test statistic and p-value
5. Make a decision (Reject or Fail to Reject H₀)
6. Interpret the results in business/ML context

---

## Common Hypothesis Tests in ML

| Test                  | Use Case                              | Data Type          |
|-----------------------|---------------------------------------|--------------------|
| Z-test                | Large samples, known variance         | Numerical          |
| T-test                | Small samples, unknown variance       | Numerical          |
| Chi-Square Test       | Independence / Goodness of fit        | Categorical        |
| ANOVA                 | Comparing multiple groups             | Numerical          |
| Mann-Whitney U        | Non-parametric comparison             | Numerical          |

---

## 📐 Mathematical Example: One-Sample T-Test

**Problem**: A company claims their ML model's accuracy is 85%. We test 30 predictions and get a sample mean accuracy of 82.5% with a sample standard deviation of 6%.

**Hypotheses**:
- H₀: μ = 85% (mean accuracy is 85%)
- H₁: μ ≠ 85% (mean accuracy is different)

**Test Statistic Formula**:
$$
t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}
$$

Where:
- $\bar{x}$ = sample mean = 82.5
- $\mu_0$ = hypothesized mean = 85
- $s$ = sample std = 6
- $n$ = sample size = 30

**Calculation**:
$$
t = \frac{82.5 - 85}{6 / \sqrt{30}} = \frac{-2.5}{1.095} \approx -2.28
$$

**Degrees of Freedom**: n-1 = 29

If the p-value < 0.05, we reject the null hypothesis.

---

## 💻 Python Example: Hypothesis Testing with SciPy

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# ==================== One-Sample T-Test ====================

# Sample data: Model accuracies from 30 runs
np.random.seed(42)
model_accuracies = np.random.normal(loc=82.5, scale=6, size=30)

# Hypothesized population mean
hypothesized_mean = 85.0

# Perform One-Sample T-Test
t_stat, p_value = stats.ttest_1samp(model_accuracies, hypothesized_mean)

print("=== One-Sample T-Test Results ===")
print(f"Sample Mean: {model_accuracies.mean():.3f}%")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.6f}")

alpha = 0.05
if p_value < alpha:
    print("✅ Result: Reject Null Hypothesis - Model accuracy is significantly different from 85%")
else:
    print("❌ Result: Fail to Reject Null Hypothesis")

📊 Two-Sample T-Test Example (A/B Testing in ML)

Python# Comparing two ML models
model_a = np.random.normal(84.5, 5, 40)
model_b = np.random.normal(87.2, 5, 40)

t_stat, p_value = stats.ttest_ind(model_a, model_b)

print(f"Model A Mean: {model_a.mean():.2f}")
print(f"Model B Mean: {model_b.mean():.2f}")
print(f"P-value: {p_value:.5f}")

if p_value < 0.05:
    print("Model B performs significantly better than Model A")




🔍 Applications in Machine Learning

Feature Selection: Testing if a feature has significant predictive power
Model Comparison: A/B testing between different algorithms
Hyperparameter Tuning: Validating improvements
Data Drift Detection: Checking if production data distribution changed
Bias Detection: Testing for fairness across demographic groups


⚠️ Important Considerations

Always check assumptions (normality, independence, equal variance)
Large sample sizes can make even tiny differences statistically significant
Statistical significance ≠ Practical significance
Multiple testing problem (use Bonferroni correction, etc.)
