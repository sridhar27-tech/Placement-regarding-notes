# Conditional Probability in Machine Learning

## 1. What is Conditional Probability?

**Conditional Probability** is the probability of an event occurring **given that another event has already occurred**. It helps us update our beliefs based on new evidence.

It is denoted as **P(A|B)**, read as *"probability of A given B"*.

### Key Formula:

$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)} = \frac{P(A \text{ and } B)}{P(B)}
$$

Where:
- $P(A \cap B)$ = Joint probability of both A and B occurring
- $P(B)$ = Probability of the conditioning event B (must be > 0)

---

## 2. Important Related Concepts

### 2.1 Bayes' Theorem

One of the most powerful results in probability and machine learning:

$$
P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}
$$

This allows us to invert conditional probabilities — extremely useful when direct computation of P(A|B) is hard.

### 2.2 Chain Rule (Product Rule)

For multiple events:

$$
P(A \cap B) = P(A \mid B) \cdot P(B)
$$

For three or more variables:

$$
P(A, B, C) = P(A \mid B, C) \cdot P(B \mid C) \cdot P(C)
$$

### 2.3 Independence

Two events A and B are **independent** if:

$$
P(A \mid B) = P(A)
$$

Which also implies:

$$
P(A \cap B) = P(A) \cdot P(B)
$$

---

## 3. Mathematical Example

**Scenario**:  
A medical test for a rare disease.

- Disease prevalence: **P(D) = 0.01** (1% of population has the disease)
- Test sensitivity: **P(T+|D) = 0.95** (true positive rate)
- False positive rate: **P(T+|No D) = 0.05**

#### Question 1: What is the probability that a person has the disease given a positive test? i.e., P(D|T+)

Using **Bayes' Theorem**:

First, calculate total probability of positive test P(T+):

$$
P(T+) = P(T+|D) \cdot P(D) + P(T+|No D) \cdot P(No D)
$$

$$
P(T+) = (0.95 \times 0.01) + (0.05 \times 0.99) = 0.0095 + 0.0495 = 0.059
$$

Now apply Bayes:

$$
P(D \mid T+) = \frac{P(T+|D) \cdot P(D)}{P(T+)} = \frac{0.95 \times 0.01}{0.059} \approx 0.161 \quad (16.1\%)
$$

**Interpretation**: Even with a positive test, there is only **16.1%** chance the person actually has the disease due to the low base rate.

#### Question 2: Probability of disease given negative test

$$
P(D \mid T-) = \frac{P(T-|D) \cdot P(D)}{P(T-)}
$$

(This is much lower, highlighting the importance of base rates.)

---

## 4. Machine Learning Example: Naive Bayes Classifier

Conditional probability is the **foundation** of the Naive Bayes algorithm, widely used for text classification, spam detection, and sentiment analysis.

**Scenario**: Email Spam Classification

- Let S = Spam, NS = Not Spam
- Let W = "contains word 'free'"

Given:
- P(S) = 0.3 (30% emails are spam)
- P(W|S) = 0.7 (70% of spam contain "free")
- P(W|NS) = 0.1 (10% of non-spam contain "free")

**Goal**: Compute P(S|W) — probability email is spam given it contains "free".

Using Bayes' Theorem:

$$
P(S \mid W) = \frac{P(W \mid S) \cdot P(S)}{P(W)}
$$

$$
P(W) = P(W|S)P(S) + P(W|NS)P(NS) = (0.7 \times 0.3) + (0.1 \times 0.7) = 0.21 + 0.07 = 0.28
$$

$$
P(S \mid W) = \frac{0.7 \times 0.3}{0.28} \approx 0.75 \quad (75\%)
$$

**Naive Bayes in Practice (Python)**:

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Example with Gaussian Naive Bayes (assumes features ~ Normal)
X_train = np.array([[1.2], [0.8], [3.1], [2.9]])  # features
y_train = np.array([0, 0, 1, 1])                 # 0=class A, 1=class B

model = GaussianNB()
model.fit(X_train, y_train)
