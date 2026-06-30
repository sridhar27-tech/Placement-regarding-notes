# K-Nearest Neighbors (KNN) Algorithm

**K-Nearest Neighbors** is a simple, versatile, and **supervised machine learning algorithm** used for both **classification** and **regression** tasks.

---

## 1. What is KNN?

- **Non-parametric** and **instance-based** learning algorithm.
- Makes predictions based on the **K closest data points** (neighbors) in the feature space.
- No explicit training phase — it "memorizes" the training data (lazy learner).
- Works on the principle: **"Birds of a feather flock together"**.

---

## 2. How KNN Works

1. **Store** all training examples.
2. For a new query point:
   - Calculate **distance** to every training point.
   - Find the **K nearest neighbors**.
   - **Classification**: Take majority vote of neighbors.
   - **Regression**: Take average (or weighted average) of neighbors' values.

---

## 3. Distance Metrics

Common distance measures:

- **Euclidean Distance** (most common):
  $$
  d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
  $$

- **Manhattan Distance**:
  $$
  d(x, y) = \sum_{i=1}^{n} |x_i - y_i|
  $$

- **Minkowski Distance** (general form):
  $$
  d(x, y) = \left( \sum_{i=1}^{n} |x_i - y_i|^p \right)^{\frac{1}{p}}
  $$

- **Hamming Distance** (for categorical data)
- **Cosine Similarity** (for text/high-dimensional data)

---
give me the markdown format for Knn K nearest neighbor in MLMarkdown# K-Nearest Neighbors (KNN) Algorithm

**K-Nearest Neighbors** is a simple, versatile, and **supervised machine learning algorithm** used for both **classification** and **regression** tasks.

---

## 1. What is KNN?

- **Non-parametric** and **instance-based** learning algorithm.
- Makes predictions based on the **K closest data points** (neighbors) in the feature space.
- No explicit training phase — it "memorizes" the training data (lazy learner).
- Works on the principle: **"Birds of a feather flock together"**.

---

## 2. How KNN Works

1. **Store** all training examples.
2. For a new query point:
   - Calculate **distance** to every training point.
   - Find the **K nearest neighbors**.
   - **Classification**: Take majority vote of neighbors.
   - **Regression**: Take average (or weighted average) of neighbors' values.

---

## 3. Distance Metrics

Common distance measures:

- **Euclidean Distance** (most common):
  $$
  d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
  $$

- **Manhattan Distance**:
  $$
  d(x, y) = \sum_{i=1}^{n} |x_i - y_i|
  $$

- **Minkowski Distance** (general form):
  $$
  d(x, y) = \left( \sum_{i=1}^{n} |x_i - y_i|^p \right)^{\frac{1}{p}}
  $$

- **Hamming Distance** (for categorical data)
- **Cosine Similarity** (for text/high-dimensional data)

---

## 4. Algorithm Steps (Pseudocode)

```python
def knn_predict(X_train, y_train, x_test, k=3, task='classification'):
    distances = []
    
    for i, x_train in enumerate(X_train):
        dist = euclidean_distance(x_test, x_train)
        distances.append((dist, y_train[i]))
    
    # Sort by distance
    distances.sort(key=lambda x: x[0])
    
    # Get K nearest neighbors
    k_nearest = distances[:k]
    
    if task == 'classification':
        # Majority vote
        votes = [label for _, label in k_nearest]
        return most_common(votes)
    else:
        # Average for regression
        values = [value for _, value in k_nearest]
        return sum(values) / k

5. Choosing the Value of K

Small K → More flexible, higher variance, risk of overfitting.
Large K → Smoother decision boundary, higher bias, risk of underfitting.
Common practice: Start with K = sqrt(n) (n = number of samples).
Use cross-validation to find optimal K.
Usually odd numbers for classification to avoid ties.


6. Advantages

✅ Very simple and intuitive
✅ No training phase (lazy learning)
✅ Works well with small datasets
✅ Can be used for both classification and regression
✅ Naturally handles multi-class problems


7. Disadvantages

❌ Computationally expensive (slow prediction on large datasets)
❌ Sensitive to irrelevant and redundant features
❌ Curse of dimensionality (performance drops in high dimensions)
❌ Needs feature scaling (StandardScaler recommended)
❌ Sensitive to imbalanced data


8. Implementation in scikit-learn
Pythonfrom sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Classification
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')
knn.fit(X_train, y_train)

# Regression
knn_reg = KNeighborsRegressor(n_neighbors=5)

9. Key Hyperparameters

ParameterDescriptionCommon Valuesn_neighborsNumber of neighbors (K)3, 5, 7, 11weightsVoting weightuniform, distancemetricDistance metriceuclidean, manhattan, minkowskialgorithmSearch algorithmauto, ball_tree, kd_tree, brute

10. When to Use KNN

Small to medium-sized datasets
Low-dimensional data
When interpretability is important
Baseline model for comparison
Recommendation systems
Anomaly detection







## 4. Algorithm Steps (Pseudocode)

```python
def knn_predict(X_train, y_train, x_test, k=3, task='classification'):
    distances = []
    
    for i, x_train in enumerate(X_train):
        dist = euclidean_distance(x_test, x_train)
        distances.append((dist, y_train[i]))
    
    # Sort by distance
    distances.sort(key=lambda x: x[0])
    
    # Get K nearest neighbors
    k_nearest = distances[:k]
    
    if task == 'classification':
        # Majority vote
        votes = [label for _, label in k_nearest]
        return most_common(votes)
    else:
        # Average for regression
        values = [value for _, value in k_nearest]
        return sum(values) / k
