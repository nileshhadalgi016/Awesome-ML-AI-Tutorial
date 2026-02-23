# Part 1 — LINEAR ALGEBRA

Linear algebra is the **language of data** in ML. Every dataset is a matrix, every data point is a vector, and every model transformation is a matrix operation.

---

## 1.1 Scalars, Vectors, Matrices, and Tensors

### Scalar
A single number.

$$a = 5, \quad \alpha = 0.01$$

In ML: a learning rate, a bias term, a single pixel intensity.

### Vector
An ordered list of numbers (a 1-D array). Written as a column by convention.

$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^n$$

In ML: a single training example with $n$ features, a row of weights, a gradient direction.

**Key operations on vectors:**

| Operation | Formula | Geometric Meaning |
|---|---|---|
| Addition | $\mathbf{a} + \mathbf{b} = [a_i + b_i]$ | Tip-to-tail placement |
| Scalar multiplication | $c\mathbf{a} = [ca_i]$ | Stretch / shrink |
| Dot product | $\mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i$ | Projection of one onto other |
| Norm (length) | $\|\mathbf{a}\| = \sqrt{\sum_i a_i^2}$ | Distance from origin |

**The Dot Product — your most important operation:**

$$\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$$

- If $\cos\theta = 1$ → vectors point the same way (similar).
- If $\cos\theta = 0$ → vectors are orthogonal (independent).
- If $\cos\theta = -1$ → vectors point opposite ways.

> **ML connection**: Cosine similarity in NLP, dot-product attention in Transformers, logistic regression's $\mathbf{w}^T\mathbf{x}$ — all are dot products.

### Matrix
A 2-D array of numbers.

$$A = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix} \in \mathbb{R}^{m \times n}$$

In ML: a dataset ($m$ samples × $n$ features), a weight matrix in a neural network layer, an image (height × width).

### Tensor
A generalization beyond 2-D. A 3-D tensor is a "stack of matrices."

| Object | Dimensions | ML Example |
|---|---|---|
| Scalar | 0-D | Learning rate |
| Vector | 1-D | Single sample features |
| Matrix | 2-D | Batch of samples, grayscale image |
| 3-D Tensor | 3-D | Color image (H×W×3), batch of sequences |
| 4-D Tensor | 4-D | Batch of color images (B×H×W×C) |

> **Why "TensorFlow"?** Because data flows through computation graphs as tensors.

---

## 1.2 Matrix Operations

### Addition & Scalar Multiplication (element-wise)

$$A + B = [a_{ij} + b_{ij}], \qquad cA = [ca_{ij}]$$

### Matrix Multiplication

Given $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}$:

$$C = AB, \quad c_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}, \quad C \in \mathbb{R}^{m \times p}$$

**Rules to remember:**
- Inner dimensions must match: $(m \times \mathbf{n})(\mathbf{n} \times p)$
- NOT commutative: $AB \neq BA$ in general
- IS associative: $(AB)C = A(BC)$

> **ML connection**: A neural network layer computes $\mathbf{y} = W\mathbf{x} + \mathbf{b}$ — this is matrix-vector multiplication. Stacking layers = multiplying matrices.

### Transpose

Flip rows and columns:

$$(A^T)_{ij} = A_{ji}$$

Properties:
- $(AB)^T = B^T A^T$ (order reverses!)
- $(A^T)^T = A$

### Hadamard (Element-wise) Product

$$A \odot B = [a_{ij} \cdot b_{ij}]$$

Used in: LSTM gates, attention masking, feature-wise scaling.

### Broadcasting (practical concept)

When shapes don't perfectly match, lower-dimensional arrays are "stretched" to match. Example: adding a bias vector $\mathbf{b} \in \mathbb{R}^n$ to every row of $X \in \mathbb{R}^{m \times n}$.

---

## 1.3 Special Matrices & Properties

| Matrix Type | Definition | ML Use |
|---|---|---|
| **Identity** $I$ | 1s on diagonal, 0s elsewhere; $AI = IA = A$ | No-op transformation |
| **Diagonal** | Non-zero only on diagonal | Scaling each feature independently |
| **Symmetric** | $A = A^T$ | Covariance matrices, kernels |
| **Orthogonal** | $Q^TQ = I$, i.e., $Q^{-1} = Q^T$ | PCA rotation, preserves lengths |
| **Positive Definite** | $\mathbf{x}^T A \mathbf{x} > 0$ for all $\mathbf{x} \neq 0$ | Guarantees unique minimum (convex loss) |
| **Sparse** | Mostly zeros | Adjacency matrices in GNNs, bag-of-words |

---

## 1.4 Determinant of a Matrix

The determinant tells you how much a matrix transformation **scales area/volume**.

For a $2 \times 2$ matrix:

$$\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$$

For a $3 \times 3$ matrix (cofactor expansion along the first row):

$$\det(A) = a_{11}(a_{22}a_{33} - a_{23}a_{32}) - a_{12}(a_{21}a_{33} - a_{23}a_{31}) + a_{13}(a_{21}a_{32} - a_{22}a_{31})$$

**Key properties:**

| Property | What it means |
|---|---|
| $\det(A) = 0$ | Matrix is **singular** — no inverse, columns are linearly dependent |
| $\det(A) \neq 0$ | Matrix is **invertible** — transformation is reversible |
| $\det(AB) = \det(A)\det(B)$ | Determinants multiply |
| $\det(A^T) = \det(A)$ | Transpose doesn't change determinant |
| $\det(cA) = c^n \det(A)$ | Scaling each of $n$ dimensions by $c$ |

> **ML connection**: In Gaussian distributions, the determinant of the covariance matrix appears in the normalization constant:
> $$p(\mathbf{x}) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$
> If $\det(\Sigma) = 0$, the distribution is degenerate (collapsed along some dimension).

---

## 1.5 Inverse of a Matrix

$A^{-1}$ is the matrix such that:

$$A A^{-1} = A^{-1} A = I$$

**Exists only when** $\det(A) \neq 0$.

For a $2 \times 2$ matrix:

$$A^{-1} = \frac{1}{ad - bc}\begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

**Properties:**
- $(AB)^{-1} = B^{-1}A^{-1}$
- $(A^T)^{-1} = (A^{-1})^T$

> **ML connection**: The closed-form solution for linear regression is:
> $$\mathbf{w} = (X^TX)^{-1}X^T\mathbf{y}$$
> This requires $X^TX$ to be invertible. If features are linearly dependent (multicollinearity), $\det(X^TX) \approx 0$ and the solution is unstable — this is why we add **regularization** ($X^TX + \lambda I$).

### Pseudo-inverse (Moore-Penrose)

When $A$ is not square or not invertible, we use:

$$A^+ = (A^TA)^{-1}A^T \quad \text{(left pseudo-inverse, when } m > n\text{)}$$

This gives the least-squares solution. NumPy's `np.linalg.pinv()` computes this.

---

## 1.6 Eigenvalues and Eigenvectors

### The Core Idea

An eigenvector of matrix $A$ is a vector whose **direction doesn't change** when $A$ is applied — it only gets scaled:

$$A\mathbf{v} = \lambda\mathbf{v}$$

- $\mathbf{v}$ = eigenvector (the direction)
- $\lambda$ = eigenvalue (the scaling factor)

### How to Find Them

1. Solve the **characteristic equation**: $\det(A - \lambda I) = 0$
2. This gives you the eigenvalues $\lambda_1, \lambda_2, \ldots$
3. For each $\lambda_i$, solve $(A - \lambda_i I)\mathbf{v} = 0$ to get eigenvectors

### Worked Example

$$A = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}$$

**Step 1**: $\det(A - \lambda I) = 0$

$$(4-\lambda)(3-\lambda) - 2 = 0 \implies \lambda^2 - 7\lambda + 10 = 0 \implies \lambda = 5 \text{ or } \lambda = 2$$

**Step 2**: For $\lambda = 5$:

$$(A - 5I)\mathbf{v} = \begin{bmatrix} -1 & 1 \\ 2 & -2 \end{bmatrix}\mathbf{v} = 0 \implies \mathbf{v}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

For $\lambda = 2$:

$$(A - 2I)\mathbf{v} = \begin{bmatrix} 2 & 1 \\ 2 & 1 \end{bmatrix}\mathbf{v} = 0 \implies \mathbf{v}_2 = \begin{bmatrix} 1 \\ -2 \end{bmatrix}$$

### Properties of Eigenvalues

| Property | Formula |
|---|---|
| Sum of eigenvalues | $= \text{trace}(A) = \sum a_{ii}$ |
| Product of eigenvalues | $= \det(A)$ |
| Symmetric matrix | All eigenvalues are **real**; eigenvectors are **orthogonal** |
| Positive definite matrix | All eigenvalues are **positive** |

> **ML connection**: 
> - **PCA**: Eigenvectors of the covariance matrix = principal components (directions of max variance). Eigenvalues = amount of variance along each direction.
> - **Google's PageRank**: The ranking vector is the dominant eigenvector of the web's link matrix.
> - **Spectral clustering**: Uses eigenvectors of the graph Laplacian.

---

## 1.7 Diagonalization

A matrix $A$ is **diagonalizable** if:

$$A = PDP^{-1}$$

Where:
- $P$ = matrix whose columns are eigenvectors of $A$
- $D$ = diagonal matrix of eigenvalues

**Why this matters:**

$$A^k = PD^kP^{-1}$$

Raising a diagonal matrix to a power is trivial — just raise each diagonal entry. This makes computing matrix powers efficient.

**When is $A$ diagonalizable?**
- When $A$ has $n$ linearly independent eigenvectors
- **Always true** for symmetric matrices (Spectral Theorem)

> **ML connection**: Covariance matrices are symmetric → always diagonalizable. PCA diagonalizes the covariance matrix, rotating data into uncorrelated axes.

---

## 1.8 Singular Value Decomposition (SVD)

SVD is the **most important matrix decomposition** in ML. It works for **any** matrix (not just square ones).

### The Decomposition

$$A = U \Sigma V^T$$

Where $A \in \mathbb{R}^{m \times n}$:

| Component | Size | Properties | Meaning |
|---|---|---|---|
| $U$ | $m \times m$ | Orthogonal ($U^TU = I$) | Left singular vectors (row space directions) |
| $\Sigma$ | $m \times n$ | Diagonal, non-negative | Singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$ |
| $V^T$ | $n \times n$ | Orthogonal ($V^TV = I$) | Right singular vectors (column space directions) |

### Intuition: What SVD Does

Any linear transformation = **Rotate → Scale → Rotate**

1. $V^T$: Rotate input space
2. $\Sigma$: Scale along each axis
3. $U$: Rotate to output space

### Connection to Eigenvalues

- Singular values: $\sigma_i = \sqrt{\lambda_i(A^TA)}$
- Columns of $V$ = eigenvectors of $A^TA$
- Columns of $U$ = eigenvectors of $AA^T$

### Low-Rank Approximation (Truncated SVD)

Keep only the top $k$ singular values:

$$A \approx A_k = U_k \Sigma_k V_k^T$$

This is the **best rank-$k$ approximation** of $A$ (Eckart-Young theorem).

Original storage: $m \times n$ values  
Truncated storage: $mk + k + nk = k(m + n + 1)$ values

> **ML connections**:
> - **Dimensionality reduction**: PCA is SVD applied to centered data
> - **Recommender systems**: Netflix Prize used SVD to factorize the user-item rating matrix
> - **NLP**: Latent Semantic Analysis (LSA) = SVD on term-document matrix
> - **Image compression**: Keep top-$k$ singular values to compress images
> - **Noise reduction**: Small singular values often correspond to noise

---

## 1.9 Norms (Measuring Size)

| Norm | Formula | Use in ML |
|---|---|---|
| $L^1$ (Manhattan) | $\|\mathbf{x}\|_1 = \sum |x_i|$ | Lasso regularization (sparsity) |
| $L^2$ (Euclidean) | $\|\mathbf{x}\|_2 = \sqrt{\sum x_i^2}$ | Ridge regularization, distance |
| $L^\infty$ (Max) | $\|\mathbf{x}\|_\infty = \max|x_i|$ | Adversarial robustness bounds |
| Frobenius (matrix) | $\|A\|_F = \sqrt{\sum_{ij} a_{ij}^2}$ | Matrix regularization |

> **Why regularize?** Adding $\lambda\|\mathbf{w}\|^2$ to the loss penalizes large weights, preventing overfitting. $L^1$ drives weights to exactly zero (feature selection). $L^2$ shrinks weights smoothly.

---

## 1.10 Linear Independence, Span, Rank, and Basis

**Linearly independent**: No vector in the set can be written as a combination of others.

**Span**: The set of all vectors reachable by linear combinations.

**Basis**: A linearly independent set that spans the entire space.

**Rank**: The number of linearly independent columns (or rows) of a matrix.
- $\text{rank}(A) = $ number of non-zero singular values
- If $\text{rank}(A) < \min(m,n)$, the matrix is **rank-deficient**

> **ML connection**: If your feature matrix has rank < number of features, you have redundant features (multicollinearity). PCA removes this redundancy.

---

[Next: Calculus →](02_Calculus.md)
