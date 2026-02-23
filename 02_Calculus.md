# Part 2 — CALCULUS

Calculus tells ML models **how to learn**. Derivatives tell you which direction to move parameters to reduce the loss.

---

## 2.1 Derivatives

The derivative of $f(x)$ at point $x$:

$$f'(x) = \frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**Meaning**: The instantaneous rate of change — the slope of the tangent line.

### Essential Derivative Rules

| Rule | Formula |
|---|---|
| Constant | $\frac{d}{dx}(c) = 0$ |
| Power | $\frac{d}{dx}(x^n) = nx^{n-1}$ |
| Sum | $(f+g)' = f' + g'$ |
| Product | $(fg)' = f'g + fg'$ |
| Quotient | $\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}$ |
| Exponential | $\frac{d}{dx}(e^x) = e^x$ |
| Logarithm | $\frac{d}{dx}\ln(x) = \frac{1}{x}$ |
| Sigmoid | $\sigma'(x) = \sigma(x)(1 - \sigma(x))$ |

### Common ML Activation Functions and Their Derivatives

| Function | $f(x)$ | $f'(x)$ |
|---|---|---|
| ReLU | $\max(0, x)$ | $0$ if $x < 0$, $1$ if $x > 0$ |
| Sigmoid | $\frac{1}{1+e^{-x}}$ | $f(x)(1 - f(x))$ |
| Tanh | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $1 - f(x)^2$ |
| Leaky ReLU | $\max(\alpha x, x)$ | $\alpha$ if $x < 0$, $1$ if $x > 0$ |
| Softplus | $\ln(1 + e^x)$ | $\sigma(x)$ |

---

## 2.2 Partial Derivatives

When $f$ depends on multiple variables, a partial derivative measures the rate of change **with respect to one variable**, holding the others fixed.

$$f(x, y) = x^2y + 3y$$

$$\frac{\partial f}{\partial x} = 2xy \qquad \frac{\partial f}{\partial y} = x^2 + 3$$

> **ML connection**: A loss function $L(w_1, w_2, \ldots, w_n)$ depends on many parameters. $\frac{\partial L}{\partial w_i}$ tells you how the loss changes when you tweak weight $w_i$ — this is what gradient descent uses.

---

## 2.3 The Chain Rule

The **engine of backpropagation**.

### Single Variable

If $y = f(g(x))$, then:

$$\frac{dy}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$$

### Multivariate

If $L = L(y)$, $y = f(\mathbf{w})$, $\mathbf{w}$ are parameters:

$$\frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial w_i}$$

### Backpropagation Example

Consider a simple 2-layer network:

$$z_1 = w_1 x \quad \rightarrow \quad a_1 = \sigma(z_1) \quad \rightarrow \quad z_2 = w_2 a_1 \quad \rightarrow \quad L = (z_2 - y)^2$$

To find $\frac{\partial L}{\partial w_1}$, chain through the computation graph:

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial z_2} \cdot \frac{\partial z_2}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial w_1}$$

$$= 2(z_2 - y) \cdot w_2 \cdot \sigma'(z_1) \cdot x$$

> **Key insight**: Backpropagation is just the chain rule applied systematically from the loss backward through every layer. Each layer computes its local derivative and passes it back.

---

## 2.4 Gradient

The gradient collects **all partial derivatives** into a vector:

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

### Critical Properties

1. **Points in the direction of steepest ascent** of $f$
2. **Magnitude** = rate of steepest ascent
3. **Perpendicular to level curves** (contour lines)

### Gradient Descent

To minimize a loss $L(\mathbf{w})$:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla L(\mathbf{w}_t)$$

- $\eta$ = learning rate (step size)
- We move **opposite** to the gradient (steepest descent)

**Variants:**
| Variant | Batch Size | Tradeoff |
|---|---|---|
| Batch GD | All data | Stable but slow |
| Stochastic GD (SGD) | 1 sample | Noisy but fast |
| Mini-batch GD | $k$ samples | Best of both |
| Adam | Mini-batch + adaptive lr | Most commonly used |

---

## 2.5 The Jacobian Matrix

When you have a **vector-valued function** $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian is the matrix of all first-order partial derivatives:

$$J = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix} \in \mathbb{R}^{m \times n}$$

Row $i$ = gradient of the $i$-th output.

> **ML connection**: 
> - In a neural network layer $\mathbf{h} = f(W\mathbf{x} + \mathbf{b})$, backpropagation computes Jacobian-vector products (JVPs) — not the full Jacobian.
> - In normalizing flows (generative models), you need $\det(J)$ to track how probability densities transform.

---

## 2.6 The Hessian Matrix

The Hessian is the matrix of **second-order partial derivatives**:

$$H = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\ \vdots & \vdots & \ddots \end{bmatrix}$$

The Hessian is **always symmetric** (assuming continuous second derivatives).

### What the Hessian Tells You

| Hessian Property | Meaning for the Loss Surface |
|---|---|
| Positive definite ($H \succ 0$) | Local **minimum** — loss curves up in all directions |
| Negative definite ($H \prec 0$) | Local **maximum** — loss curves down |
| Indefinite (mixed eigenvalues) | **Saddle point** — up in some directions, down in others |
| Large eigenvalues | Sharp curvature → need small learning rate |
| Small eigenvalues | Flat region → slow convergence |
| Condition number $\frac{\lambda_{max}}{\lambda_{min}}$ large | Ill-conditioned → gradient descent zigzags |

> **ML connection**:
> - **Newton's method**: $\mathbf{w}_{t+1} = \mathbf{w}_t - H^{-1}\nabla L$ (uses curvature for better steps, but $H$ is $n \times n$ — too big for deep learning).
> - **Adam/AdaGrad**: Approximate diagonal of $H$ cheaply.
> - **Loss landscape research**: Deep networks have many saddle points (not local minima) — the Hessian at critical points is typically indefinite.

---

## 2.7 Taylor Expansion (Why Gradient Descent Works)

Approximating $f$ near point $\mathbf{a}$:

$$f(\mathbf{x}) \approx f(\mathbf{a}) + \nabla f(\mathbf{a})^T(\mathbf{x} - \mathbf{a}) + \frac{1}{2}(\mathbf{x} - \mathbf{a})^T H(\mathbf{a})(\mathbf{x} - \mathbf{a})$$

- 1st-order approximation → justifies gradient descent
- 2nd-order approximation → justifies Newton's method

---

[← Previous: Linear Algebra](01_Linear_Algebra.md) | [Next: Probability & Statistics →](03_Probability_and_Statistics.md)
