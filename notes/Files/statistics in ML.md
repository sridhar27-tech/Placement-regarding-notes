# Statistics in Machine Learning

**A Comprehensive Guide with Mathematical Examples and Python Code**

Statistics is the backbone of Machine Learning. This guide covers core concepts with **mathematical formulations**, **worked examples**, and **executable Python code**.

---

## Table of Contents
1. [Descriptive Statistics](#descriptive-statistics)
2. [Probability Fundamentals](#probability-fundamentals)
3. [Probability Distributions](#probability-distributions)
4. [Hypothesis Testing](#hypothesis-testing)
5. [Regression Analysis](#regression-analysis)
6. [Bias-Variance Tradeoff](#bias-variance-tradeoff)
7. [Dimensionality Reduction (PCA)](#dimensionality-reduction-pca)
8. [Conclusion](#conclusion)

---

## Descriptive Statistics

### Mathematical Definitions

Let \( X = \{x_1, x_2, \dots, x_n\} \) be a dataset.

- **Mean** (Arithmetic):  
  $$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

- **Variance**:  
  $$\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2 \quad \text{(Population)}$$
  Sample variance uses \( n-1 \) in denominator.

- **Standard Deviation**:  
  $$\sigma = \sqrt{\sigma^2}$$

### Python Example

```python
import numpy as np
import matplotlib.pyplot as plt

# Sample data
data = np.array([2.5, 3.1, 4.8, 5.2, 6.7, 7.1, 8.3])

print("Mean:", np.mean(data))
print("Median:", np.median(data))
print("Std Dev:", np.std(data, ddof=1))  # sample std
print("Variance:", np.var(data, ddof=1))

# Visualization
plt.hist(data, bins=5, alpha=0.7)
plt.axvline(np.mean(data), color='r', label='Mean')
plt.legend()
plt.title("Data Distribution")
plt.show()

Python Code - Sampling & Visualization

import numpy as np
import seaborn as sns
from scipy.stats import norm

# Generate data from Normal distribution
mu, sigma = 100, 15
samples = np.random.normal(mu, sigma, 10000)

print("Sample Mean:", np.mean(samples))
print("Sample Std:", np.std(samples))

# Plot
sns.histplot(samples, kde=True, stat="density")
x = np.linspace(mu-3*sigma, mu+3*sigma, 100)
plt.plot(x, norm.pdf(x, mu, sigma), 'r-', label='Theoretical')
plt.title("Normal Distribution")
plt.legend()
plt.show()

Python Code (t-test)

from scipy import stats
sample = [168, 172, 175, 169, 171, 174, 173, 170, 176, 172] * 3  # n=30
t_stat, p_value = stats.ttest_1samp(sample, 170)

print("t-statistic:", t_stat)
print("p-value:", p_value)

if p_value < 0.05:
    print("Reject H0 - Mean is significantly different from 170")
else:
    print("Fail to reject H0")


Bias-Variance Tradeoff
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate nonlinear data
np.random.seed(42)
X = np.random.uniform(-3, 3, 100).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.3, 100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# High Bias (Linear)
model_low = LinearRegression()
model_low.fit(X_train, y_train)
print("Linear (High Bias) Test MSE:", np.mean((model_low.predict(X_test) - y_test)**2))

# High Variance (High Degree Polynomial)
model_high = make_pipeline(PolynomialFeatures(degree=15), LinearRegression())
model_high.fit(X_train, y_train)
print("High Degree Poly (High Variance) Test MSE:", np.mean((model_high.predict(X_test) - y_test)**2))
