# Logistic Regression in Machine Learning

## Introduction
Logistic Regression is one of the most popular and fundamental classification algorithms in Machine Learning. Despite the name "regression", it is used for **classification** tasks.

It predicts the probability that a given input belongs to a particular class using the sigmoid function.

### Sigmoid Function
$$
P(y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1x_1 + ... + \beta_nx_n)}}
$$

---

## When to Use Logistic Regression
- Binary classification (Yes/No, Spam/Not Spam)
- Multiclass classification (with softmax or one-vs-rest)
- When you need interpretable probabilities
- Baseline model for many problems

---



Hyperparameter Tuning Example
Pythonfrom sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear', 'saga'],
    'penalty': ['l2']
}

grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)

Advantages and Disadvantages
Advantages:

Easy to implement and interpret
Works well with linearly separable data
Provides probability scores
Computationally efficient

Disadvantages:

Cannot handle complex non-linear relationships well
Sensitive to outliers
Assumes features are linearly related to log-odds
Performance suffers with highly correlated features


Tips

Always scale features when using Logistic Regression
Use Pipeline to avoid data leakage
Check for multicollinearity before training
Use Regularization (C parameter) to prevent overfitting


3. Random Forest Classifier

Pythonfrom sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

4. Support Vector Machine (SVM)

Pythonfrom sklearn.svm import SVC
model = SVC(kernel='rbf', C=1.0, probability=True)

5. K-Nearest Neighbors (KNN)

Pythonfrom sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)

6. Naive Bayes

Pythonfrom sklearn.naive_bayes import GaussianNB
model = GaussianNB()

7. XGBoost Classifier

Pythonimport xgboost as xgb
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1)

Model Comparison Framework
Pythonfrom sklearn.model_selection import cross_val_score

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

Best Practices for Classification

Handle class imbalance (SMOTE, class weights)
Use proper evaluation metrics (Precision, Recall, F1, ROC-AUC)
Feature scaling for distance-based models
Ensemble methods for better performance
Always use cross-validation


Evaluation Metrics

Accuracy
Precision & Recall
F1-Score
ROC Curve & AUC
Confusion Matrix

---


**Here is the complete Markdown content for Classification Algorithms:**

```markdown
# Classification Algorithms in Machine Learning

## Overview of Classification
Classification is a supervised learning task where the model predicts discrete class labels.

### Types of Classification
- Binary Classification
- Multiclass Classification
- Multilabel Classification
- Imbalanced Classification

---

## Major Classification Algorithms

### 1. Logistic Regression
(See previous markdown for details)

### 2. Decision Tree Classifier

```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=5, random_state=42)

## Complete Code Example

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Feature Scaling (Recommended)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
model = LogisticRegression(max_iter=1000, multi_class='auto', solver='lbfgs')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
