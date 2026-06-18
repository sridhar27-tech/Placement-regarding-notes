# Singular Value Decomposition (SVD) in Machine Learning

## 1. Brief Explanation

**Singular Value Decomposition (SVD)** is a fundamental matrix factorization technique that decomposes **any rectangular matrix** $A$ into three matrices:

$$
A = U \Sigma V^T
$$

- **$U$**: Left singular vectors (orthogonal) — captures row information
- **$\Sigma$**: Diagonal matrix of singular values (non-negative, sorted descending)
- **$V^T$**: Right singular vectors (orthogonal) — captures column information

**Why SVD is Extremely Important in ML**:
- Foundation of **Principal Component Analysis (PCA)**
- Dimensionality reduction
- Matrix completion & recommendation systems
- Latent Semantic Analysis (LSA) in NLP
- Image compression
- Data denoising
- Low-rank approximation

SVD is more general than eigendecomposition because it works on **any matrix** (not just square).

---

## 2. Mathematical Calculations

### SVD Formula
For a matrix $A \in \mathbb{R}^{m \times n}$:

$$
A = U \Sigma V^T
$$

Where:
- $U \in \mathbb{R}^{m \times m}$ (orthogonal: $U^T U = I$)
- $\Sigma \in \mathbb{R}^{m \times n}$ (diagonal with singular values $\sigma_1 \geq \sigma_2 \geq \dots \geq \sigma_r \geq 0$)
- $V \in \mathbb{R}^{n \times n}$ (orthogonal: $V^T V = I$)

**Singular Values** ($\sigma_i$) are the square roots of the eigenvalues of $A^TA$ or $AA^T$.

**Low-Rank Approximation** (Truncated SVD):
$$
A_k = U_k \Sigma_k V_k^T \quad (k \ll \text{rank}(A))
$$

This is the best rank-$k$ approximation of $A$ (Eckart–Young Theorem).

---

## 3. Prioritized Example: Image Compression using SVD

**Problem**: Compress a grayscale image represented as a 5×5 matrix using SVD.

**Original Matrix** $A$ (simplified pixel intensities):

$$
A = \begin{bmatrix}
100 & 110 & 120 & 105 & 95 \\
105 & 115 & 125 & 110 & 100 \\
110 & 120 & 130 & 115 & 105 \\
100 & 105 & 115 & 100 & 95 \\
90 & 100 & 110 & 95 & 85
\end{bmatrix}
$$

---

### Step-by-Step Calculation:

**Step 1**: Compute SVD  
(Using NumPy in practice — shown conceptually)

- Compute $A^TA$ and $AA^T$ to find eigenvalues → singular values.
- Resulting singular values (approx):  
  $\sigma = [480.2, 28.4, 12.1, 5.3, 1.8]$

**Step 2**: Truncated SVD with $k=2$ (keep top 2 singular values)

$$
A_2 = U_2 \Sigma_2 V_2^T
$$

**Step 3**: Reconstruction Error  
$$
\text{Frobenius Error} = \|A - A_2\|_F = \sqrt{\sum_{i=3}^{5} \sigma_i^2} \approx 14.8
$$

**Interpretation**:
- With just **2 components**, we retain most of the image structure.
- Storage reduced dramatically (from 25 numbers to ~20 numbers for rank-2 approximation).

**Real-world Impact**: This technique is used to compress large images while preserving visual quality.

---

## 4. Additional Example: Recommendation Systems (Matrix Completion)

In collaborative filtering:
- $A$ = User-Item rating matrix (mostly sparse)
- SVD decomposes it into:
  - $U$: User latent factors
  - $V^T$: Item latent factors
- Predict missing ratings using low-rank approximation.

---

## 5. Python Code Implementation (NumPy)

```python
import numpy as np

# Sample Data: User-Item Rating Matrix (5 users, 4 items)
A = np.array([
    [5, 4, 0, 0],
    [4, 5, 0, 1],
    [0, 0, 5, 4],
    [0, 1, 4, 5],
    [1, 0, 4, 5]
])

# Full SVD
U, Sigma, Vt = np.linalg.svd(A, full_matrices=False)

print("Singular Values:", Sigma)

# Truncated SVD (k=2)
k = 2
U_k = U[:, :k]
Sigma_k = np.diag(Sigma[:k])
Vt_k = Vt[:k, :]

# Reconstructed Matrix
A_reconstructed = U_k @ Sigma_k @ Vt_k

print("Original Matrix:\n", A)
print("Reconstructed (k=2):\n", np.round(A_reconstructed, 2))
