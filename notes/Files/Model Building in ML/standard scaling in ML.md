# 🔧 Standard Scaling in Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)

A complete guide to **Standard Scaling** (also known as Z-score normalization) and other feature scaling techniques in Machine Learning.

## Why Scaling Matters

Many machine learning algorithms perform better or converge faster when features are on a similar scale:
- Distance-based algorithms (**KNN**, **K-Means**, **SVM**)
- Gradient-based algorithms (**Neural Networks**, **Logistic Regression**)
- Regularization techniques (**Lasso**, **Ridge**)

**Standard Scaling** transforms data to have **mean = 0** and **standard deviation = 1**.

---

## Standard Scaler Formula

$$
z = \frac{x - \mu}{\sigma}
$$

Where:
- $x$ = original value
- $\mu$ = mean of the feature
- $\sigma$ = standard deviation of the feature

---



('knn', KNeighborsClassifier())])

When to Use Which Scaler?





























ScalerBest ForHandles Outliers?Mean/StdStandardScalerMost algorithms, Normal-like dataNo0 / 1MinMaxScalerNeural Networks, Image dataNo0-1RobustScalerData with outliersYesMedian

Common Mistakes to Avoid

Fitting scaler on entire data before splitting → Data Leakage
Using scaling on categorical features
Forgetting to apply the same scaler on test/new data
Assuming all algorithms need scaling (e.g., Tree-based models like Random Forest usually don't)


Advanced Usage
Inverse Transform
PythonX_original = scaler.inverse_transform(X_scaled)
Custom Scaling with ColumnTransformer
Pythonfrom sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

Best Practices

Always fit on train set only
Use Pipeline to avoid mistakes
Scale after train-test split
Check distributions before and after scaling
## Code Examples

### 1. Basic Standard Scaling

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Sample data
X = np.array([[1, 50], [2, 30], [3, 80], [4, 40], [5, 60]])

# Split data first (Important!)
X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

# Initialize scaler
scaler = StandardScaler()

# Fit on training data only, then transform both
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Original:\n", X_train)
print("Scaled:\n", X_train_scaled)
