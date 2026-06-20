# Probability in Machine Learning

## Introduction
Probability is the foundation of Machine Learning. Most ML algorithms rely on probabilistic reasoning to make predictions, handle uncertainty, and learn from data.

## Basic Concepts

### 1. Probability Basics
- **Random Variable**: A variable whose possible values are outcomes of a random phenomenon.
- **Probability of an Event**:
  $$
  P(A) = \frac{\text{Number of favorable outcomes}}{\text{Total number of possible outcomes}}
  $$

### 2. Conditional Probability
$$
P(A|B) = \frac{P(A \cap B)}{P(B)}
$$

### 3. Bayes' Theorem
$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$
- **Prior**: $P(A)$
- **Likelihood**: $P(B|A)$
- **Posterior**: $P(A|B)$
- **Evidence**: $P(B)$

**Medical Test Example**:
- $P(D) = 0.01$, $P(+|D) = 0.95$, $P(+|\neg D) = 0.05$
- $P(D|+) \approx 0.161$ (16.1%)

## Bayesian Neural Networks (BNNs)

### Mathematical Explanation
Traditional Neural Networks learn a **single set of weights** (point estimate).  
Bayesian Neural Networks model the **weights as distributions** and compute the **posterior** using Bayes' Theorem.

#### Core Bayesian Formula for Neural Networks
$$
P(\theta | D) = \frac{P(D | \theta) \cdot P(\theta)}{P(D)}
$$

Where:
- $\theta$ = all weights and biases of the network
- $D$ = training dataset
- $P(\theta | D)$ = **Posterior distribution** over weights (what we want)
- $P(D | \theta)$ = **Likelihood** of data given weights (usually Gaussian for regression)
- $P(\theta)$ = **Prior** (e.g., Gaussian prior $\theta \sim \mathcal{N}(0, \sigma^2)$)
- $P(D)$ = **Marginal Likelihood** (evidence) — usually intractable

#### Predictive Distribution
At inference time, instead of one prediction, we integrate over all possible weights:
$$
P(y^* | x^*, D) = \int P(y^* | x^*, \theta) \, P(\theta | D) \, d\theta
$$

This integral is intractable, so we use approximations.

#### Common Approximation: Variational Inference (VI)
We approximate the true posterior $q(\theta) \approx P(\theta | D)$ by minimizing KL divergence:
$$
\text{KL}(q(\theta) || P(\theta | D)) \rightarrow \min
$$

**Loss Function (ELBO)**:
$$
\mathcal{L} = \mathbb{E}_{q(\theta)}[\log P(D|\theta)] - \text{KL}(q(\theta) || P(\theta))
$$

### Practical Approximation: Monte Carlo Dropout

### Python Code Examples

#### 1. Simple Bayesian Neural Network with Monte Carlo Dropout (PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, output_size=3, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, apply_dropout=True):
        x = F.relu(self.fc1(x))
        if apply_dropout:
            x = self.dropout(x)          # Enable dropout at inference for uncertainty
        x = self.fc2(x)
        return F.softmax(x, dim=1)
    
    # Get prediction with uncertainty
    def predict_with_uncertainty(self, x, n_samples=50):
        self.train()  # Enable dropout
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x, apply_dropout=True)
                predictions.append(pred)
        predictions = torch.stack(predictions)
        
        mean_pred = predictions.mean(dim=0)           # Average prediction
        uncertainty = predictions.std(dim=0)          # Uncertainty (std dev)
        return mean_pred, uncertainty

# Usage Example
model = BayesianNN()
sample = torch.randn(1, 4)

mean_prob, uncertainty = model.predict_with_uncertainty(sample, n_samples=100)
print("Mean Prediction:", mean_prob)
print("Uncertainty (std):", uncertainty)

Basic Bayes Theorem Function

def bayes_theorem(prior, likelihood, evidence):
    """Basic Bayes' Theorem implementation"""
    return (likelihood * prior) / evidence
