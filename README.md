# Awesome ML / AI Tutorial

A structured, hands-on reference for learning the mathematical foundations and core supervised learning algorithms of machine learning. Each resource follows a consistent format: formal definitions, worked examples, visualisations, and guided implementation notes.

---

## Repository Structure

```
Awesome-ML-AI-Tutorial/
├── Theory Notes          — Mathematical foundations (Markdown)
├── Algorithm Notebooks   — Step-by-step algorithm walkthroughs (Jupyter)
└── sample_data/          — CSV datasets used by the notebooks
```

---

## Theory Notes

Covers the mathematical prerequisites for understanding ML algorithms. Each file includes definitions, formulas, worked examples, and direct connections to ML applications.

| # | Topic | Key Sections |
|---|-------|-------------|
| 1 | [Linear Algebra](01_Linear_Algebra.md) | Scalars, Vectors, Tensors, Matrix Operations, Determinants, Inverse, Eigenvalues, SVD, Norms, Rank |
| 2 | [Calculus](02_Calculus.md) | Derivatives, Partial Derivatives, Chain Rule, Gradient Descent, Jacobian, Hessian, Taylor Expansion |
| 3 | [Probability & Statistics](03_Probability_and_Statistics.md) | Probability Rules, Bayes' Theorem, Random Variables, PDFs, Expectation, Variance, Covariance, MLE, Information Theory |
| 4 | [Discrete Mathematics](04_Discrete_Mathematics.md) | Sets & Logic, Combinatorics, Graph Theory, Trees, Algorithmic Complexity |
| 5 | [Summary & Roadmap](05_Summary_and_Roadmap.md) | Math ↔ Algorithm map, Formula cheat sheet, 8-week study plan |

---

## Algorithm Notebooks

Each notebook is a self-contained, guided study session. Sections include formal definitions, implementation, visualisation with dark-theme plots, and a reference summary.

| Notebook | Algorithm | What You Will Learn |
|----------|-----------|---------------------|
| [linearregression.ipynb](linearregression.ipynb) | **Linear Regression** | Least squares fitting, slope/intercept interpretation, regression line visualisation, prediction |
| [logisticregression.ipynb](logisticregression.ipynb) | **Logistic Regression** | Binary classification, sigmoid function, train/test split, accuracy evaluation, decision boundary |
| [svm.ipynb](svm.ipynb) | **Support Vector Machine** | Hyperplane, margin maximisation, support vectors, decision function score, effect of C parameter |
| [knn_explainer.ipynb](knn_explainer.ipynb) | **K-Nearest Neighbors** | Distance metrics, Euclidean distance, majority voting, effect of K on decision boundary, lazy learning |

---

## Sample Data

| File | Used In | Description |
|------|---------|-------------|
| [sample_data/student_data.csv](sample_data/student_data.csv) | logisticregression.ipynb | Study hours and pass/fail outcomes |
| [sample_data/iris_binary.csv](sample_data/iris_binary.csv) | — | Binary Iris subset (Setosa vs Versicolor) |

---

## Getting Started

**Prerequisites:** Python 3.8+

```bash
# Clone the repository
git clone https://github.com/nileshhadalgi016/Awesome-ML-AI-Tutorial.git
cd Awesome-ML-AI-Tutorial

# Install dependencies
pip install -r requirements.txt
```

Open any `.ipynb` file in Jupyter Lab, VS Code, or any compatible notebook environment and run the cells top to bottom.

---

## Recommended Study Order

1. Work through Theory Notes 01 → 05 to build mathematical intuition
2. Follow the notebooks in this order: Linear Regression → Logistic Regression → SVM → KNN
3. Each notebook is also usable independently as a standalone reference
