# Chi-Square Test in Machine Learning: Complete Guide

## 1. Introduction

The **Chi-Square Test** ($\chi^2$ Test) is a statistical test used to determine whether there is a significant association between **categorical variables**. Unlike Z-test and T-test which deal with means, Chi-Square works with **frequencies** and **proportions**.

It is one of the most important tests for **categorical data analysis** in Machine Learning.

---

## 2. When to Use Chi-Square Test in ML

- Feature selection (categorical features vs target)
- Testing independence between two categorical variables
- Goodness of fit test (does observed distribution match expected?)
- Evaluating classification model performance (Confusion Matrix analysis)
- A/B testing with categorical outcomes (click vs no-click)
- Checking data bias or distribution shifts

---

## 3. Types of Chi-Square Tests

1. **Chi-Square Test of Independence** — Tests if two categorical variables are independent
2. **Chi-Square Goodness of Fit Test** — Tests if observed frequencies match expected frequencies

---

## 4. Mathematical Foundation

### Chi-Square Statistic

$$
\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}
$$

Where:
- $O_i$ = Observed frequency
- $E_i$ = Expected frequency

**Degrees of Freedom (df)**:
- Independence: $(r-1) \times (c-1)$
- Goodness of Fit: $k - 1$ (k = number of categories)

---

## 5. Detailed Numerical Example

**Problem**:  
We want to check if there is an association between **Model Type** (Random Forest vs XGBoost) and **Prediction Outcome** (Correct vs Incorrect) on a test set.

|                | Correct | Incorrect | Total |
|----------------|---------|-----------|-------|
| Random Forest  | 85      | 15        | 100   |
| XGBoost        | 92      | 8         | 100   |
| **Total**      | 177     | 23        | 200   |

**Expected Frequencies** calculated → Chi-Square statistic = **2.78**

**Degrees of Freedom** = 1  
**Critical Value** (α=0.05) = 3.841

Since 2.78 < 3.841 → **Fail to reject H₀**. No significant association between model type and prediction accuracy.

---

## 6. Python Implementation

```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def chi_square_independence(observed):
    """
    Perform Chi-Square Test of Independence
    """
    chi2, p_value, dof, expected = stats.chi2_contingency(observed)
    
    return {
        'chi2_statistic': chi2,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'expected_frequencies': expected,
        'reject_h0': p_value < 0.05
    }

# ====================== Example Usage ======================
# Contingency Table: Model vs Prediction Result
observed = np.array([
    [85, 15],   # Random Forest
    [92, 8]     # XGBoost
])

result = chi_square_independence(observed)

print("Chi-Square Test of Independence Results:")
print(f"Chi-Square Statistic : {result['chi2_statistic']:.4f}")
print(f"P-Value              : {result['p_value']:.6f}")
print(f"Degrees of Freedom   : {result['degrees_of_freedom']}")
print(f"Reject Null Hypothesis: {result['reject_h0']}")

# Optional: Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(observed, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Correct', 'Incorrect'],
            yticklabels=['Random Forest', 'XGBoost'])
plt.title('Observed Frequencies')
plt.ylabel('Model')
plt.xlabel('Prediction Outcome')
plt.show()
