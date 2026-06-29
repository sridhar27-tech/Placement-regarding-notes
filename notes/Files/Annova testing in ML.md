# ANOVA Test in Machine Learning: Complete Guide

## 1. Introduction

**ANOVA** (Analysis of Variance) is a statistical method used to compare the **means of three or more groups** to determine whether there is a statistically significant difference among them.

While a T-test is limited to comparing **two groups**, ANOVA is the generalization used when you have **multiple groups**.

---

## 2. When to Use ANOVA in Machine Learning

- Comparing performance of multiple ML models (e.g., RF, XGBoost, LightGBM, Neural Net)
- Hyperparameter tuning — testing multiple learning rates or depths
- Evaluating different feature engineering techniques
- Comparing cross-validation results across several algorithms
- A/B/C testing with more than two variants
- Analyzing training loss across different optimizers

---

## 3. Types of ANOVA

1. **One-Way ANOVA** — One independent variable (most common in ML)
2. **Two-Way ANOVA** — Two independent variables + interaction
3. **Repeated Measures ANOVA** — Same subjects measured multiple times

---

## 4. Mathematical Foundation

### F-Statistic

$$
F = \frac{\text{Between-Group Variance}}{\text{Within-Group Variance}} = \frac{\text{MSB}}{\text{MSW}}
$$

Where:
- MSB = Mean Square Between groups
- MSW = Mean Square Within groups

**Null Hypothesis ($H_0$)**: All group means are equal  
**Alternative Hypothesis ($H_1$)**: At least one group mean is different

---

## 5. Detailed Numerical Example

**Problem**:  
We compared accuracy of 4 different models on the same dataset (each run 30 times):

- Model A: mean = 88.2%
- Model B: mean = 91.5%
- Model C: mean = 89.7%
- Model D: mean = 92.8%

**Result**:  
F-statistic = 12.45  
p-value = 0.000003  
Degrees of Freedom (between) = 3, (within) = 116

Since p-value < 0.05 → **Reject H₀**. There is a significant difference in performance among the models.

---

## 6. Python Implementation

```python
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def one_way_anova(*groups):
    """Perform One-Way ANOVA"""
    f_stat, p_value = stats.f_oneway(*groups)
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'reject_h0': p_value < 0.05
    }

# ====================== Example Usage ======================
np.random.seed(42)

model_a = np.random.normal(0.882, 0.025, 30)
model_b = np.random.normal(0.915, 0.022, 30)
model_c = np.random.normal(0.897, 0.028, 30)
model_d = np.random.normal(0.928, 0.020, 30)

result = one_way_anova(model_a, model_b, model_c, model_d)

print("One-Way ANOVA Results:")
print(f"F-Statistic     : {result['f_statistic']:.4f}")
print(f"P-Value         : {result['p_value']:.8f}")
print(f"Reject H0       : {result['reject_h0']}")

# Post-hoc test (Tukey HSD) - to see which groups differ
data = pd.DataFrame({
    'accuracy': np.concatenate([model_a, model_b, model_c, model_d]),
    'model': ['A']*30 + ['B']*30 + ['C']*30 + ['D']*30
})

# Visualization
plt.figure(figsize=(10, 6))
sns.boxplot(x='model', y='accuracy', data=data)
plt.title('Model Performance Comparison (ANOVA)')
plt.ylabel('Accuracy')
plt.show()
