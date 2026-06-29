# Z-Test in Machine Learning: Complete Guide

## 1. Introduction

The **Z-test** is one of the fundamental statistical hypothesis tests used in Machine Learning and Data Science. It helps determine whether observed differences in data are statistically significant or due to random chance.

It is particularly useful when working with **large datasets**, which is very common in ML.

---

## 2. When to Use Z-Test in ML

- Comparing model accuracy / F1-score / AUC against a baseline
- A/B testing between two models or algorithms
- Testing if a new feature significantly improves model performance
- Quality control of model predictions
- Drift detection (concept drift in production models)

---

## 3. Assumptions of Z-Test

1. Sample size is large (**n ≥ 30**)
2. Population standard deviation (**σ**) is known
3. Data follows (approximately) normal distribution
4. Observations are independent
5. Random sampling

---

## 4. Mathematical Foundation

### Z-Statistic

$$
Z = \frac{\bar{x} - \mu_0}{\frac{\sigma}{\sqrt{n}}}
$$

### Hypothesis Testing

**Null Hypothesis ($H_0$)**: $\bar{x} = \mu_0$ (no significant difference)  
**Alternative Hypothesis ($H_1$)**: $\bar{x} \neq \mu_0$ (two-tailed) / $\bar{x} > \mu_0$ (right-tailed) / $\bar{x} < \mu_0$ (left-tailed)

**Critical Values (α = 0.05)**:
- Two-tailed: ±1.96
- One-tailed (right): +1.645
- One-tailed (left): -1.645

---

## 5. Detailed Numerical Example

**Scenario**:  
An old recommendation model has an average click-through rate (CTR) of **3.2%** with known standard deviation **0.8%**.  
A new model was tested on 120 users and achieved **3.75%** CTR.  

Is the improvement statistically significant?

```math
Z = \frac{3.75 - 3.2}{\frac{0.8}{\sqrt{120}}} = \frac{0.55}{0.073} \approx 7.53
