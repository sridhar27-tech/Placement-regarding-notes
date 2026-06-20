# Bayes' Theorem: Explanation, Mathematics, and Python Implementation

## What is Bayes' Theorem?

**Bayes' Theorem** is a fundamental result in probability theory that describes how to update the probability of a hypothesis based on new evidence. It allows us to compute **conditional probabilities** in a reverse direction — from the effect back to the cause.

It is named after Thomas Bayes and is widely used in:
- Machine Learning (Naive Bayes classifiers)
- Medical diagnosis
- Spam filtering
- A/B testing
- Risk assessment

---

## Mathematical Explanation

### The Formula

Bayes' Theorem states:

$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$

### Breakdown of Terms

- **$P(A|B)$** — **Posterior probability**: Probability of event $A$ (hypothesis) occurring given that $B$ (evidence) has occurred.
- **$P(B|A)$** — **Likelihood**: Probability of observing evidence $B$ given that hypothesis $A$ is true.
- **$P(A)$** — **Prior probability**: Initial belief about the probability of hypothesis $A$ before seeing the evidence.
- **$P(B)$** — **Marginal probability** (or evidence): Total probability of observing evidence $B$, regardless of $A$.

### How to Compute $P(B)$

Using the **law of total probability**:

$$
P(B) = P(B|A) \cdot P(A) + P(B|\neg A) \cdot P(\neg A)
$$

Where $\neg A$ means "not $A$".

---

## Simple Example: Medical Test

Suppose a disease affects 1% of the population. A test is 99% accurate (true positive rate), but has a 5% false positive rate.

- Let $D$ = has disease
- Let $T^+$ = tests positive

We want $P(D|T^+)$ — probability that a person has the disease given a positive test.

Given:
- $P(D) = 0.01$ (prior)
- $P(\neg D) = 0.99$
- $P(T^+|D) = 0.99$ (sensitivity)
- $P(T^+|\neg D) = 0.05$ (false positive rate)

Using Bayes' Theorem:

$$
P(D|T^+) = \frac{P(T^+|D) \cdot P(D)}{P(T^+)} = \frac{0.99 \times 0.01}{P(T^+)}
$$

First compute denominator:

$$
P(T^+) = (0.99 \times 0.01) + (0.05 \times 0.99) = 0.0099 + 0.0495 = 0.0594
$$

So:

$$
P(D|T^+) = \frac{0.0099}{0.0594} \approx 0.1667 \ (16.67\%)
$$

Even with a positive test, the probability of actually having the disease is only about **16.7%** due to the low base rate.

---

## Python Implementation

```python
def bayes_theorem(prior_A, likelihood_B_given_A, likelihood_B_given_not_A):
    """
    Calculate posterior probability P(A|B) using Bayes' Theorem.
    
    Parameters:
        prior_A (float): P(A)
        likelihood_B_given_A (float): P(B|A)
        likelihood_B_given_not_A (float): P(B|¬A)
    
    Returns:
        float: P(A|B)
    """
    # P(¬A)
    prior_not_A = 1 - prior_A
    
    # P(B) = P(B|A)P(A) + P(B|¬A)P(¬A)
    total_prob_B = (likelihood_B_given_A * prior_A) + \
                   (likelihood_B_given_not_A * prior_not_A)
    
    # Bayes' Theorem
    posterior_A_given_B = (likelihood_B_given_A * prior_A) / total_prob_B
    
    return posterior_A_given_B

# Example: Disease test
prior_disease = 0.01
true_positive_rate = 0.99
false_positive_rate = 0.05

result = bayes_theorem(prior_disease, true_positive_rate, false_positive_rate)
print(f"Probability of having disease given positive test: {result:.4f} ({result*100:.2f}%)")
