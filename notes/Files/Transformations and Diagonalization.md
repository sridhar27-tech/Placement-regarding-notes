# Transformations and Diagonalization in Machine Learning

## 1. Brief Explanation

### Linear Transformations
A linear transformation maps vectors from one space to another while preserving vector addition and scalar multiplication. In ML, they are used for:
- Feature scaling and normalization
- Dimensionality reduction (PCA, LDA)
- Data projection and whitening
- Changing basis for easier computation

Mathematically:  
$$
\mathbf{y} = A \mathbf{x}
$$

### Diagonalization
Diagonalization decomposes a square matrix $A$ into:
$$
A = P D P^{-1}
$$
where $P$ contains eigenvectors and $D$ is a diagonal matrix of eigenvalues.

**Why it matters in ML**:
- Simplifies matrix operations ($A^k = P D^k P^{-1}$)
- Foundation of **Principal Component Analysis (PCA)**
- Used in spectral clustering, graph analysis, and optimization

---

## 2. Mathematical Calculations

### Eigenvalues & Eigenvectors
$$
A \mathbf{v} = \lambda \mathbf{v} \quad \Rightarrow \quad \det(A - \lambda I) = 0
$$

### Diagonalization
If $A$ has $n$ linearly independent eigenvectors:
$$
A = P D P^{-1}, \quad D = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_n)
$$

---

## 3. Prioritized Example: PCA using Diagonalization

**Dataset** (House price related 2D data):

| Sample | $x_1$ | $x_2$ |
|--------|-------|-------|
| 1      | 2     | 3     |
| 2      | 3     | 4     |
| 3      | 5     | 6     |
| 4      | 8     | 9     |
| 5      | 9     | 10    |

**Step-by-Step:**

1. **Mean Center**:
   $\mu = [5.4, 6.4]$

2. **Covariance Matrix** $S \approx \begin{bmatrix} 8.3 & 8.3 \\ 8.3 & 8.3 \end{bmatrix}$

3. **Eigendecomposition** (Diagonalization):
   - Eigenvalues: $\lambda_1 = 16.6$, $\lambda_2 = 0$
   - Eigenvectors: $\mathbf{v_1} = [0.707, 0.707]^T$, $\mathbf{v_2} = [-0.707, 0.707]^T$

4. **Projection** onto first PC: Significant dimensionality reduction with almost no information loss.

---

## 4. Additional Example: Singular Value Decomposition (SVD)

**SVD** is a generalization of diagonalization for any matrix (not just square):

$$
A = U \Sigma V^T
$$

- $U$: Left singular vectors
- $\Sigma$: Diagonal matrix of singular values
- $V$: Right singular vectors

**Use in ML**: 
- PCA (without computing covariance)
- Matrix completion
- Latent Semantic Analysis (LSA) in NLP
- Recommendation systems

---

## 5. Linear Discriminant Analysis (LDA) – Supervised Transformation

LDA finds linear combinations of features that best separate classes.

**Key Step**: Solve generalized eigenvalue problem:
$$
S_B \mathbf{w} = \lambda S_W \mathbf{w}
$$

Where $S_B$ = Between-class scatter, $S_W$ = Within-class scatter.

**Difference from PCA**: PCA is unsupervised (max variance), LDA is supervised (max class separation).

---

## 6. Python Code Implementation (NumPy)

```python
import numpy as np

# Sample Data
X = np.array([[2,3], [3,4], [5,6], [8,9], [9,10]])

# Step 1: Center the data
X_mean = np.mean(X, axis=0)
X_centered = X - X_mean

# Step 2: Covariance Matrix
cov_matrix = np.cov(X_centered.T)

# Step 3: Eigendecomposition (Diagonalization)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort by eigenvalues
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Step 4: Project data onto top principal component
X_pca = X_centered @ eigenvectors[:, 0]
print("Projected Data:", X_pca)



