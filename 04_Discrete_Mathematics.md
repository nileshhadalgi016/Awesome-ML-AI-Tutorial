# Part 4 — DISCRETE MATHEMATICS

Discrete math provides the foundation for **algorithms, data structures, graph-based models, and combinatorics** in ML.

---

## 4.1 Sets and Logic

### Sets
- $A \cup B$ (union), $A \cap B$ (intersection), $A \setminus B$ (difference), $A^c$ (complement)
- $|A|$ = cardinality (number of elements)
- $A \subseteq B$ means every element of $A$ is in $B$

### Logic
| Symbol | Meaning |
|---|---|
| $\land$ | AND |
| $\lor$ | OR |
| $\neg$ | NOT |
| $\Rightarrow$ | Implies (if...then) |
| $\Leftrightarrow$ | If and only if |
| $\forall$ | For all |
| $\exists$ | There exists |

> **ML connection**: Decision trees split data based on logical conditions. Boolean logic underlies feature engineering and data filtering.

---

## 4.2 Combinatorics

### Counting Principles

| Concept | Formula | Meaning |
|---|---|---|
| Permutations | $P(n,r) = \frac{n!}{(n-r)!}$ | Ordered arrangements of $r$ from $n$ |
| Combinations | $\binom{n}{r} = \frac{n!}{r!(n-r)!}$ | Unordered selections of $r$ from $n$ |
| With replacement | $n^r$ | $r$ independent choices from $n$ |

### Why This Matters for ML

- **Feature selection**: Choosing $k$ features from $n$ = $\binom{n}{k}$ possibilities
- **Hyperparameter search**: Grid search explores all combinations
- **Combinatorial explosion**: Why brute-force is infeasible and we need smart algorithms
- **Binomial distribution**: Uses $\binom{n}{k}$

---

## 4.3 Graph Theory

A graph $G = (V, E)$ consists of vertices $V$ and edges $E$.

### Types of Graphs

| Type | Description | ML Example |
|---|---|---|
| Undirected | Edges have no direction | Social networks, molecule bonds |
| Directed (digraph) | Edges have direction | Citation networks, web links |
| Weighted | Edges have values | Distance/similarity graphs |
| Bipartite | Two sets, edges only between sets | User-item interactions |
| Tree | Connected, no cycles | Decision trees |
| DAG | Directed, no cycles | Bayesian networks, computation graphs |

### Key Concepts

**Adjacency Matrix** $A$:

$$A_{ij} = \begin{cases} 1 & \text{if edge from } i \text{ to } j \\ 0 & \text{otherwise} \end{cases}$$

- Symmetric for undirected graphs
- Often sparse → use sparse representations

**Degree**: Number of edges connected to a vertex.

**Degree Matrix** $D$: Diagonal matrix where $D_{ii} = \text{degree}(v_i)$.

**Graph Laplacian**: $L = D - A$
- Eigenvalues of $L$ reveal graph structure (spectral clustering)
- Second-smallest eigenvalue = algebraic connectivity (Fiedler value)

**Paths and Connectivity**:
- **Path**: sequence of vertices connected by edges
- **Connected graph**: path exists between every pair of vertices
- **Shortest path**: BFS (unweighted), Dijkstra (weighted)

### Graph Algorithms in ML

| Algorithm | Purpose | ML Application |
|---|---|---|
| BFS/DFS | Graph traversal | Feature extraction on graphs |
| PageRank | Node importance | Web ranking, node classification |
| Shortest path | Distance computation | Graph kernels |
| Min-cut | Graph partitioning | Image segmentation |
| Message passing | Information propagation | Graph Neural Networks (GNNs) |

> **ML connection — Graph Neural Networks (GNNs)**:
> GNNs learn node representations by aggregating information from neighbors:
> $$\mathbf{h}_v^{(l+1)} = \sigma\left(\sum_{u \in \mathcal{N}(v)} \frac{1}{|\mathcal{N}(v)|} W^{(l)} \mathbf{h}_u^{(l)}\right)$$
> This is essentially matrix multiplication with the normalized adjacency matrix.

---

## 4.4 Trees and Decision Structures

### Binary Trees
- Each node has at most 2 children
- **Depth**: distance from root to node
- **Height**: maximum depth

### Decision Trees (ML perspective)

A decision tree recursively partitions the feature space:

1. At each node, choose feature $j$ and threshold $t$ to split
2. Split criterion: maximize **information gain** or minimize **Gini impurity**

**Information Gain:**
$$IG = H(\text{parent}) - \sum_{\text{child}} \frac{|\text{child}|}{|\text{parent}|} H(\text{child})$$

**Gini Impurity:**
$$G = 1 - \sum_{c=1}^C p_c^2$$

where $p_c$ = proportion of class $c$ in the node.

> **Random Forests**: Ensemble of decision trees, each trained on a random subset of data and features.

---

## 4.5 Recursion, Induction, and Algorithmic Complexity

### Big-O Notation

| Complexity | Name | Example |
|---|---|---|
| $O(1)$ | Constant | Hash table lookup |
| $O(\log n)$ | Logarithmic | Binary search |
| $O(n)$ | Linear | Single pass through data |
| $O(n \log n)$ | Log-linear | Sorting (mergesort) |
| $O(n^2)$ | Quadratic | Pairwise distances |
| $O(n^3)$ | Cubic | Matrix inversion |
| $O(2^n)$ | Exponential | Brute-force subsets |

### ML Complexity Examples

| Operation | Complexity |
|---|---|
| Matrix multiplication $(n \times n)$ | $O(n^3)$ |
| SVD of $m \times n$ matrix | $O(\min(mn^2, m^2n))$ |
| Training linear regression (closed form) | $O(n^2m + n^3)$ |
| k-NN prediction | $O(nm)$ per query |
| Training a decision tree | $O(nm \log m)$ |
| Forward pass, fully connected layer | $O(n_{in} \cdot n_{out})$ |
| Self-attention (Transformer) | $O(n^2 d)$ where $n$ = sequence length |

---

[← Previous: Probability & Statistics](03_Probability_and_Statistics.md) | [Next: Summary & Roadmap →](05_Summary_and_Roadmap.md)
