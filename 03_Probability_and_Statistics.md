# Part 3 — PROBABILITY AND STATISTICS

ML is fundamentally about making **predictions under uncertainty**. Probability provides the framework.

---

## 3.1 Basics of Probability

### Sample Space, Events, Probability

- **Sample space** $\Omega$: set of all possible outcomes
- **Event** $A \subseteq \Omega$: a subset of outcomes
- **Probability** $P(A) \in [0, 1]$

### Axioms of Probability (Kolmogorov)

1. $P(A) \geq 0$
2. $P(\Omega) = 1$
3. For mutually exclusive events: $P(A \cup B) = P(A) + P(B)$

### Fundamental Rules

| Rule | Formula |
|---|---|
| Complement | $P(A^c) = 1 - P(A)$ |
| Addition | $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ |
| Conditional | $P(A|B) = \frac{P(A \cap B)}{P(B)}$ |
| Multiplication | $P(A \cap B) = P(A|B) \cdot P(B)$ |
| Independence | $A \perp B \iff P(A \cap B) = P(A)P(B)$ |
| Law of Total Prob. | $P(A) = \sum_i P(A|B_i)P(B_i)$ |

### Joint, Marginal, and Conditional

For two random variables $X, Y$:

- **Joint**: $P(X=x, Y=y)$ — probability of both happening
- **Marginal**: $P(X=x) = \sum_y P(X=x, Y=y)$ — "sum out" the other variable
- **Conditional**: $P(Y=y|X=x) = \frac{P(X=x, Y=y)}{P(X=x)}$

---

## 3.2 Bayes' Theorem

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

In ML language:

$$\underbrace{P(\theta | \text{data})}_{\text{posterior}} = \frac{\overbrace{P(\text{data} | \theta)}^{\text{likelihood}} \cdot \overbrace{P(\theta)}^{\text{prior}}}{\underbrace{P(\text{data})}_{\text{evidence}}}$$

### What Each Term Means

| Term | Meaning | Example |
|---|---|---|
| **Prior** $P(\theta)$ | Your belief before seeing data | "Weights are probably small" |
| **Likelihood** $P(\text{data}|\theta)$ | How well the data fits the model | "If $\theta=2$, how likely is this data?" |
| **Posterior** $P(\theta|\text{data})$ | Updated belief after seeing data | "Given data, $\theta$ is probably near 2.3" |
| **Evidence** $P(\text{data})$ | Normalizing constant | Often intractable; approximate |

### Worked Example: Spam Filter

- $P(\text{spam}) = 0.3$ (prior: 30% of emails are spam)
- $P(\text{"free"}|\text{spam}) = 0.8$ (likelihood: 80% of spam contains "free")
- $P(\text{"free"}|\text{not spam}) = 0.1$

$$P(\text{"free"}) = 0.8 \times 0.3 + 0.1 \times 0.7 = 0.31$$

$$P(\text{spam}|\text{"free"}) = \frac{0.8 \times 0.3}{0.31} \approx 0.774$$

> **ML connections**: 
> - **Naive Bayes classifier**: Directly applies Bayes' theorem with independence assumptions.
> - **Bayesian neural networks**: Place priors on weights, compute posterior.
> - **MAP estimation**: $\hat{\theta} = \arg\max P(\theta|\text{data})$ — regularization is equivalent to choosing a prior!
>   - $L^2$ regularization ↔ Gaussian prior on weights
>   - $L^1$ regularization ↔ Laplace prior on weights

---

## 3.3 Random Variables and Probability Distributions

### Random Variable
A function that maps outcomes to numbers: $X: \Omega \to \mathbb{R}$

- **Discrete**: takes countable values (dice roll, word count)
- **Continuous**: takes any real value (temperature, weight)

### Probability Mass Function (PMF) — discrete

$$p(x) = P(X = x), \qquad \sum_x p(x) = 1$$

### Probability Density Function (PDF) — continuous

$$P(a \leq X \leq b) = \int_a^b f(x) \, dx, \qquad \int_{-\infty}^{\infty} f(x) \, dx = 1$$

**Note**: $f(x)$ is NOT a probability — it can be $> 1$. Only the integral gives probability.

### Cumulative Distribution Function (CDF)

$$F(x) = P(X \leq x) = \int_{-\infty}^{x} f(t) \, dt$$

- Always non-decreasing
- $F(-\infty) = 0$, $F(\infty) = 1$
- $f(x) = F'(x)$

---

## 3.4 Expectation (Mean) and Variance

### Expectation — the "average" value

$$E[X] = \begin{cases} \sum_x x \cdot p(x) & \text{discrete} \\ \int x \cdot f(x) \, dx & \text{continuous} \end{cases}$$

**Properties** (linearity):
- $E[aX + b] = aE[X] + b$
- $E[X + Y] = E[X] + E[Y]$ (always, even if dependent!)
- $E[XY] = E[X]E[Y]$ only if $X \perp Y$

### Variance — spread around the mean

$$\text{Var}(X) = E[(X - \mu)^2] = E[X^2] - (E[X])^2$$

**Properties**:
- $\text{Var}(aX + b) = a^2 \text{Var}(X)$
- $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$

### Standard Deviation

$$\sigma = \sqrt{\text{Var}(X)}$$

In the same units as $X$ — more interpretable than variance.

> **ML connection**:
> - **Loss function** = expected prediction error: $L = E[(y - \hat{y})^2]$
> - **Bias-variance tradeoff**: $\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Noise}$
>   - High bias: model too simple (underfitting)
>   - High variance: model too sensitive to training data (overfitting)

---

## 3.5 Covariance and Correlation

### Covariance — do two variables move together?

$$\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)] = E[XY] - E[X]E[Y]$$

| Value | Meaning |
|---|---|
| $\text{Cov} > 0$ | $X$ and $Y$ tend to increase together |
| $\text{Cov} < 0$ | When $X$ increases, $Y$ tends to decrease |
| $\text{Cov} = 0$ | No linear relationship (but could be nonlinearly related!) |

### Covariance Matrix

For a random vector $\mathbf{X} = [X_1, X_2, \ldots, X_n]^T$:

$$\Sigma = \text{Cov}(\mathbf{X}) = E[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^T]$$

$$\Sigma_{ij} = \text{Cov}(X_i, X_j), \qquad \Sigma_{ii} = \text{Var}(X_i)$$

Properties:
- Symmetric: $\Sigma = \Sigma^T$
- Positive semi-definite: $\mathbf{z}^T\Sigma\mathbf{z} \geq 0$ for all $\mathbf{z}$
- Diagonal entries = variances, off-diagonal = covariances

### Correlation — normalized covariance

$$\rho(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} \in [-1, 1]$$

| Value | Meaning |
|---|---|
| $\rho = 1$ | Perfect positive linear relationship |
| $\rho = -1$ | Perfect negative linear relationship |
| $\rho = 0$ | No linear correlation (NOT independence!) |

> **ML connection**:
> - **PCA**: Eigendecomposition of the covariance matrix finds directions of maximum variance.
> - **Feature selection**: Highly correlated features are redundant — remove one.
> - **Multivariate Gaussian**: Fully defined by $\boldsymbol{\mu}$ and $\Sigma$.

---

## 3.6 Important Distributions

### Discrete Distributions

#### Bernoulli Distribution
Single binary trial with success probability $p$.

$$X \sim \text{Bernoulli}(p): \quad P(X=1) = p, \quad P(X=0) = 1-p$$
$$E[X] = p, \quad \text{Var}(X) = p(1-p)$$

ML use: Binary classification output.

#### Binomial Distribution
Number of successes in $n$ independent Bernoulli trials.

$$P(X=k) = \binom{n}{k} p^k(1-p)^{n-k}$$
$$E[X] = np, \quad \text{Var}(X) = np(1-p)$$

#### Categorical / Multinoulli Distribution
Generalization of Bernoulli to $k$ categories.

$$P(X=i) = p_i, \quad \sum_{i=1}^k p_i = 1$$

ML use: Multi-class classification output (softmax).

#### Poisson Distribution
Number of events in a fixed interval, given rate $\lambda$.

$$P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$$
$$E[X] = \lambda, \quad \text{Var}(X) = \lambda$$

ML use: Count data modeling, event rates.

### Continuous Distributions

#### Uniform Distribution

$$f(x) = \frac{1}{b-a}, \quad x \in [a, b]$$
$$E[X] = \frac{a+b}{2}, \quad \text{Var}(X) = \frac{(b-a)^2}{12}$$

ML use: Random weight initialization (some schemes), random sampling.

#### Gaussian (Normal) Distribution ⭐ The Most Important Distribution

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$
$$E[X] = \mu, \quad \text{Var}(X) = \sigma^2$$

**Standard Normal**: $Z \sim \mathcal{N}(0, 1)$, obtained by $Z = \frac{X - \mu}{\sigma}$

**The 68-95-99.7 Rule:**
- 68% of values within $\mu \pm 1\sigma$
- 95% within $\mu \pm 2\sigma$
- 99.7% within $\mu \pm 3\sigma$

**Why it's everywhere in ML:**
1. **Central Limit Theorem**: Sum of many independent random variables → Gaussian
2. **Maximum entropy**: Among all distributions with given mean and variance, the Gaussian has maximum entropy (least assumptions)
3. **Gaussian noise assumption**: Linear regression assumes errors are Gaussian → least squares
4. **Weight initialization**: Xavier/He initialization uses Gaussian distributions
5. **Batch normalization**: Normalizes activations to approximately $\mathcal{N}(0,1)$
6. **VAEs**: Latent space is Gaussian by design

#### Multivariate Gaussian

$$\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$$

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

- $\boldsymbol{\mu}$ = mean vector (center)
- $\Sigma$ = covariance matrix (shape and orientation of the ellipsoid)
- $\Sigma$ diagonal → features are uncorrelated → axis-aligned ellipsoid
- $\Sigma = \sigma^2 I$ → spherical (all features same variance, no correlation)

#### Exponential Distribution

$$f(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$
$$E[X] = \frac{1}{\lambda}, \quad \text{Var}(X) = \frac{1}{\lambda^2}$$

ML use: Modeling time between events, survival analysis.

#### Beta Distribution

$$f(x; \alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}, \quad x \in [0,1]$$

ML use: Prior for Bernoulli parameter (Bayesian learning), Thompson sampling in bandits.

### Summary Table

| Distribution | Type | Parameters | When to Use |
|---|---|---|---|
| Bernoulli | Discrete | $p$ | Binary outcome |
| Binomial | Discrete | $n, p$ | Count of binary successes |
| Categorical | Discrete | $p_1, \ldots, p_k$ | Multi-class output |
| Poisson | Discrete | $\lambda$ | Count of rare events |
| Uniform | Continuous | $a, b$ | Equal probability in range |
| Gaussian | Continuous | $\mu, \sigma^2$ | Default for "natural" data |
| Exponential | Continuous | $\lambda$ | Time until event |
| Beta | Continuous | $\alpha, \beta$ | Probability of probability |

---

## 3.7 Maximum Likelihood Estimation (MLE)

Given data $\mathbf{x}_1, \ldots, \mathbf{x}_N$, find parameters $\theta$ that make the data most probable:

$$\hat{\theta}_{MLE} = \arg\max_\theta \prod_{i=1}^N P(\mathbf{x}_i | \theta) = \arg\max_\theta \sum_{i=1}^N \log P(\mathbf{x}_i | \theta)$$

(We take the log for numerical stability and to turn products into sums.)

### Example: MLE for Gaussian

Given data points $x_1, \ldots, x_N$ from $\mathcal{N}(\mu, \sigma^2)$:

$$\hat{\mu}_{MLE} = \frac{1}{N}\sum_i x_i = \bar{x} \qquad \hat{\sigma}^2_{MLE} = \frac{1}{N}\sum_i (x_i - \bar{x})^2$$

> **ML connection**: 
> - Training a classifier by minimizing cross-entropy loss **is** MLE.
> - Training a regressor by minimizing MSE **is** MLE under Gaussian noise assumption.
> - Adding regularization = MAP (Maximum A Posteriori) estimation.

---

## 3.8 Information Theory Concepts

### Entropy — uncertainty in a distribution

$$H(X) = -\sum_x p(x) \log p(x)$$

- $H = 0$: deterministic (no uncertainty)
- $H$ is maximized for uniform distribution
- Units: bits (log base 2) or nats (natural log)

### Cross-Entropy — comparing two distributions

$$H(p, q) = -\sum_x p(x) \log q(x)$$

> **ML connection**: Cross-entropy loss for classification:
> $$L = -\sum_{c=1}^C y_c \log(\hat{y}_c)$$
> where $y$ is the true label (one-hot) and $\hat{y}$ is the predicted probability.

### KL Divergence — distance between distributions

$$D_{KL}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = H(p, q) - H(p)$$

- $D_{KL} \geq 0$ (Gibbs' inequality)
- $D_{KL} = 0$ iff $p = q$
- **Not symmetric**: $D_{KL}(p\|q) \neq D_{KL}(q\|p)$

> **ML connection**: VAE loss function = reconstruction loss + $D_{KL}(\text{encoder} \| \text{prior})$.

---

[← Previous: Calculus](02_Calculus.md) | [Next: Discrete Mathematics →](04_Discrete_Mathematics.md)
