# Part 5 — PUTTING IT ALL TOGETHER

## How Math Connects to ML Algorithms

| ML Algorithm | Linear Algebra | Calculus | Probability | Discrete Math |
|---|---|---|---|---|
| **Linear Regression** | $\mathbf{w} = (X^TX)^{-1}X^T\mathbf{y}$ | Gradient of MSE | Gaussian likelihood | — |
| **Logistic Regression** | $\mathbf{w}^T\mathbf{x}$ | Sigmoid derivative, gradient | Bernoulli, cross-entropy | — |
| **PCA** | Eigendecomposition, SVD | — | Covariance matrix | — |
| **Neural Networks** | Matrix multiplications | Chain rule (backprop) | Cross-entropy loss | Computation graphs (DAG) |
| **SVM** | Dot products, kernels | Subgradients | — | Optimization (convex) |
| **k-Means** | Distance (norms) | — | — | Iterative algorithm |
| **Naive Bayes** | — | — | Bayes theorem, independence | — |
| **Decision Trees** | — | — | Entropy, information gain | Trees, recursion |
| **Random Forests** | — | — | Bootstrapping, averaging | Trees, combinatorics |
| **GNNs** | Adjacency matrix, Laplacian | Backprop on graphs | — | Graph theory |
| **Transformers** | Matrix multiply, softmax | Gradient through attention | Softmax as probability | Sequence operations |
| **VAE** | Reparameterization | Gradient of ELBO | KL divergence, Gaussian | — |
| **GANs** | Matrix operations | Gradient of minimax | Jensen-Shannon divergence | — |

---

## Quick Reference: Essential Formulas

```
LINEAR ALGEBRA
  Matrix multiply:        C = AB,  c_ij = Σ_k a_ik b_kj
  Eigenvalue equation:    Av = λv
  SVD:                    A = UΣVᵀ
  Pseudo-inverse:         A⁺ = (AᵀA)⁻¹Aᵀ

CALCULUS  
  Chain rule:             dL/dw = (dL/dy)(dy/dw)
  Gradient descent:       w ← w - η∇L(w)
  
PROBABILITY
  Bayes:                  P(A|B) = P(B|A)P(A) / P(B)
  Gaussian:               f(x) = (1/σ√2π) exp(-(x-μ)²/2σ²)
  Cross-entropy:          L = -Σ y_c log(ŷ_c)
  KL divergence:          D_KL(p‖q) = Σ p(x) log(p(x)/q(x))
  MLE:                    θ̂ = argmax Σ log P(x_i|θ)
```

---

## Study Roadmap

```
Week 1-2:  Vectors, matrices, operations, norms
            → Practice: NumPy matrix operations
            
Week 3:    Eigenvalues, SVD, PCA
            → Practice: Implement PCA from scratch
            
Week 4:    Derivatives, chain rule, gradients
            → Practice: Compute gradients by hand for simple networks
            
Week 5:    Jacobian, Hessian, optimization
            → Practice: Implement gradient descent from scratch
            
Week 6:    Probability basics, Bayes theorem
            → Practice: Implement Naive Bayes classifier
            
Week 7:    Distributions, MLE, information theory
            → Practice: Fit Gaussians to data, compute cross-entropy
            
Week 8:    Graphs, combinatorics, complexity
            → Practice: Implement decision tree, analyze complexity
```

---

> **Final advice**: Don't memorize formulas — understand what each operation *does geometrically or conceptually*. When you see $\mathbf{w}^T\mathbf{x}$, think "projection." When you see $\nabla L$, think "which direction makes the loss worse." When you see $P(\theta|\text{data})$, think "what do I believe about the parameters *now* that I've seen evidence."

---

## Navigation

| # | Topic | File |
|---|---|---|
| 1 | Linear Algebra | [01_Linear_Algebra.md](01_Linear_Algebra.md) |
| 2 | Calculus | [02_Calculus.md](02_Calculus.md) |
| 3 | Probability & Statistics | [03_Probability_and_Statistics.md](03_Probability_and_Statistics.md) |
| 4 | Discrete Mathematics | [04_Discrete_Mathematics.md](04_Discrete_Mathematics.md) |
| 5 | Summary & Roadmap | [05_Summary_and_Roadmap.md](05_Summary_and_Roadmap.md) |

---

[← Previous: Discrete Mathematics](04_Discrete_Mathematics.md) | [Back to Index](README.md)
