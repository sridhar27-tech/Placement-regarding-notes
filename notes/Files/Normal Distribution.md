# Normal Distribution in Machine Learning

## 1. What is the Normal Distribution?

The **Normal Distribution** (Gaussian Distribution) is a continuous probability distribution that is **symmetric and bell-shaped**. It is extremely important in machine learning because many natural phenomena, measurement errors, and model assumptions tend to follow this distribution.

### Key Properties:
- Perfectly symmetric around the mean (μ)
- Mean = Median = Mode
- The curve is unimodal (one peak)
- Tails extend to infinity (but probability decreases rapidly)
- Defined by only **two parameters**:
  - **μ** (mean): Determines the center
  - **σ** (standard deviation): Determines the spread/width

---

## 2. Probability Density Function (PDF)

$$
f(x \mid \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$

- The **exp** term makes the function peak at x = μ and decay exponentially as we move away.
- The denominator normalizes the total area under the curve to 1.

---

## 3. Standard Normal Distribution

When μ = 0 and σ = 1, it becomes the **Standard Normal Distribution** (Z-distribution).

**Z-score formula** (standardization):

$$
Z = \frac{X - \mu}{\sigma}
$$

This transformation allows us to use standard normal tables or functions for any normal distribution.

---

## 4. Empirical Rule (68-95-99.7 Rule)

For any normal distribution:
- **68%** of data lies within **±1σ** of the mean
- **95%** of data lies within **±2σ** of the mean
- **99.7%** of data lies within **±3σ** of the mean

This rule is very useful for quick outlier detection and understanding data spread.

---

## 5. Mathematical Example 1: Student Heights

**Scenario**:  
Heights of adult males in a population follow a normal distribution with **μ = 170 cm** and **σ = 8 cm**.

#### Questions & Solutions:

**a)** Probability that a student’s height is between **162 cm and 178 cm**?

$$
Z_1 = \frac{162 - 170}{8} = -1, \quad Z_2 = \frac{178 - 170}{8} = 1
$$

$$
P(-1 < Z < 1) \approx 0.6827 \quad (68.27\%)
$$

**b)** Probability that a student is taller than **186 cm**?

$$
Z = \frac{186 - 170}{8} = 2
$$

$$
P(Z > 2) = 1 - 0.9772 = 0.0228 \quad (2.28\%)
$$

**c)** What height corresponds to the **top 5%** of students?

We need the Z-score where cumulative probability = 0.95 → Z ≈ **1.645**

$$
X = \mu + Z \cdot \sigma = 170 + 1.645 \times 8 \approx 183.16 \text{ cm}
$$

---

## 6. Machine Learning Example: Linear Regression Residuals

In Linear Regression, we assume that the **residuals (errors)** are normally distributed with mean = 0 and constant variance (homoscedasticity).

**Scenario**:  
You built a model to predict house prices. After training, the residuals have:
- Mean ≈ 0
- Standard deviation σ = 25,000 USD

**Question**: What is the probability that the model’s prediction error is within **±50,000 USD**?

$$
Z = \frac{50000}{25000} = 2
$$

$$
P(-2 < Z < 2) \approx 0.9545 \quad (\approx 95.45\%)
$$

**Interpretation**:  
Your model’s predictions are expected to be within $50,000 of the actual price **about 95% of the time**.

**Python Code** (Common in ML workflows):

```python
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Generate synthetic residuals (as in real ML pipeline)
np.random.seed(42)
residuals = np.random.normal(0, 25000, 10000)

# Probability calculation
prob = norm.cdf(50000, 0, 25000) - norm.cdf(-50000, 0, 25000)
print(f"Probability within ±50k: {prob:.4f}")

# Visualization (very common in ML notebooks)
plt.hist(residuals, bins=100, density=True, alpha=0.7)
plt.title("Distribution of Residuals")
plt.xlabel("Residual Error ($)")
plt.ylabel("Density")
plt.show()
