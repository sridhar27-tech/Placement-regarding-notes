# Probability in Machine Learning

## Introduction
Probability is the foundation of Machine Learning. Most ML algorithms rely on probabilistic reasoning to make predictions, handle uncertainty, and learn from data. Instead of deterministic rules, ML models often work with probabilities.

## Basic Concepts

### 1. Probability Basics
- **Random Variable**: A variable whose possible values are outcomes of a random phenomenon.
  - Discrete (e.g., coin toss: Heads/Tails)
  - Continuous (e.g., height of students)

- **Probability of an Event**:
  $$
  P(A) = \frac{\text{Number of favorable outcomes}}{\text{Total number of possible outcomes}}
  $$

- **Joint Probability**: Probability of two events happening together.
  $$
  P(A \cap B) = P(A \text{ and } B)
  $$

- **Marginal Probability**: Probability of an event regardless of others.
  $$
  P(A) = \sum_{b} P(A, B=b)
  $$

### 2. Conditional Probability
The probability of an event given that another event has occurred.

$$
P(A|B) = \frac{P(A \cap B)}{P(B)}
$$

**Example from your previous question**:
- Class: 100 students (60 boys, 40 girls)
- 30 boys passed
- Probability that a randomly chosen boy has passed = \( P(\text{Passed} | \text{Boy}) = \frac{30}{60} = 0.5 \)

### 3. Bayes' Theorem
One of the most important concepts in ML (used in Naive Bayes, Bayesian Networks, etc.)

$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$

- **Prior**: $P(A)$
- **Likelihood**: $P(B|A)$
- **Posterior**: $P(A|B)$
- **Evidence**: $P(B)$

#### Detailed Explanation with Example
**Bayes' Theorem** tells us how to update our beliefs when we get new evidence.

**Medical Test Example**:
- Prior: Probability of having a disease $P(D) = 0.01$ (1%)
- Likelihood: Test is positive given disease $P(+|D) = 0.95$
- False positive rate $P(+|\neg D) = 0.05$

**Question**: If the test is positive, what's the probability you actually have the disease?

$$
P(D|+) = \frac{0.95 \times 0.01}{0.95 \times 0.01 + 0.05 \times 0.99} = \frac{0.0095}{0.059} \approx 0.161 \ (16.1\%)
$$

Even with a positive test, the probability is only ~16% because the disease is rare.

#### Connection to Neural Networks
Traditional Neural Networks give **point estimates**.  
**Bayesian Neural Networks (BNNs)** treat weights as probability distributions and use Bayes’ Theorem:

$$
P(\text{weights} | \text{data}) = \frac{P(\text{data} | \text{weights}) \cdot P(\text{weights})}{P(\text{data})}
$$

This provides uncertainty estimates.

### 4. Probability Distributions
Common distributions in ML:

- **Bernoulli**: Binary outcomes (e.g., logistic regression)
- **Binomial**: Number of successes in n trials
- **Gaussian/Normal**: Continuous data (used in linear regression, GMM)
- **Categorical/Multinomial**: Multi-class classification
- **Uniform**: Equal probability

### 5. Expectation and Variance
- **Expectation** (Mean): $E[X] = \sum x \cdot P(X=x)$
- **Variance**: Measures spread $Var(X) = E[(X - E[X])^2]$

## Probability in Machine Learning Algorithms

### 1. Supervised Learning
- **Classification**: Predict class probabilities (Softmax in Neural Networks)
- **Regression**: Often modeled with Gaussian noise

### 2. Naive Bayes Classifier
Assumes features are independent given the class.
$$
P(Class|Features) \propto P(Class) \prod P(Feature_i | Class)
$$

### 3. Bayesian Inference
- Used in Bayesian Neural Networks
- Handles uncertainty in model parameters

### 4. Generative Models
- Gaussian Mixture Models (GMM)
- Hidden Markov Models (HMM)
- Variational Autoencoders (VAE)

### 5. Reinforcement Learning
- Reward is often modeled probabilistically
- Policy as probability distribution over actions

## Python Code Examples

### A. Bayes' Theorem Implementation
```python
def bayes_theorem(prior, likelihood, evidence):
    return (likelihood * prior) / evidence

# Medical test example
prior_disease = 0.01
likelihood_positive = 0.95
false_positive = 0.05
evidence = (likelihood_positive * prior_disease) + (false_positive * 0.99)

posterior = bayes_theorem(prior_disease, likelihood_positive, evidence)
print(f"Probability of disease given positive test: {posterior:.4f} ({posterior*100:.1f}%)")


Gaussian Naive Bayes
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)
probs = model.predict_proba(X_test)
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))


Simple Neural Network (PyTorch)
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

model = SimpleNN()
sample = torch.randn(1, 4)
print("Predicted probabilities:", model(sample))


