# AI69201: Machine Learning Laboratory

[![Python](https://img.shields.io/badge/Python-3.14+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13.0-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.0+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Package Manager](https://img.shields.io/badge/Managed%20with-uv-DE5FE9?style=flat)](https://github.com/astral-sh/uv)
[![Course](https://img.shields.io/badge/IIT%20Kharagpur-AI69201-blue?style=flat)](https://www.iitkgp.ac.in/)

> **Course:** AI69201 — Machine Learning Laboratory  
> **Institution:** Indian Institute of Technology (IIT) Kharagpur

---

## 📖 Overview

This repository contains the complete curriculum, interactive demonstration notebooks, assignment solutions, source code, and theoretical study materials for **AI69201: Machine Learning Laboratory**.

The coursework spans classical statistical machine learning, algorithms implemented from first principles (NumPy), modern frameworks (`scikit-learn`, `PyTorch`), deep sequence models (1D CNNs, RNNs, LSTMs, GRUs), unsupervised clustering, and dimensionality reduction.

---

## 🗂️ Repository Structure

```text
Machine-Learning-Lab-AI69201/
├── class-01/          # Python, NumPy, Pandas & Matplotlib Fundamentals
├── class-02/          # K-Nearest Neighbors (KNN) - Classification & Regression
├── class-03/          # Linear & Logistic Regression, Multiclass Classification
├── class-04/          # Naive Bayes (Multinomial & Gaussian), NLP Spam Filtering
├── class-05/          # Decision Trees, Cost-Complexity Pruning & Regression Trees
├── class-06/          # Support Vector Machines (SVM) & Support Vector Regression (SVR)
├── class-07/          # Feedforward Neural Networks & Backpropagation (from scratch)
├── class-08/          # Deep Sequence Models: 1D CNN, RNN, GRU, LSTM & Hybrid CNN-LSTM
├── class-09/          # Ensemble Methods: Random Forest & AdaBoost.M1 (from scratch)
├── class-10/          # Unsupervised Clustering: K-Means, Hierarchical & DBSCAN
├── class-11/          # Dimensionality Reduction: PCA & LDA on Handwritten Digits
├── midsem/            # Midsem Preparation: Multiclass SVMs (OvR vs. OvO) & Boundaries
├── theory_mat/        # Lecture Slides, Notes & PDFs on ML Foundations
├── pyproject.toml     # Project dependencies & PyTorch CUDA index configuration
├── uv.lock            # Locked dependency graph (uv)
└── README.md          # Repository documentation
```

---

## 📑 Syllabus & Lab Matrix

| Lab / Class | Topic | Demonstration Notebook | Assignment Solution | Key Datasets & Techniques |
| :--- | :--- | :--- | :--- | :--- |
| **Class 01** | **Python Environment & Foundations** | [`Session_1_Mechine_Learning_Laboratory.ipynb`](class-01/Session_1_Mechine_Learning_Laboratory.ipynb) | *N/A (Orientation)* | Python runtime setup, NumPy matrix operations, Pandas manipulation, Matplotlib visualisations. |
| **Class 02** | **K-Nearest Neighbors (KNN)** | [`Session_2_Machine_Learning_Laboratory.ipynb`](class-02/Session_2_Machine_Learning_Laboratory.ipynb) | [`Assignment 1 (.ipynb)`](class-02/25AI92F02_Assignment1.ipynb) / [`.py`](class-02/25AI92F02_Assignment1.py) | Synthetic regression toy dataset, distance metrics ($L_1, L_2, L_\infty$), $k$-hyperparameter tuning. |
| **Class 03** | **Linear & Logistic Regression** | [`Session_03_ML_Lab(AI69201)_Demo.ipynb`](class-03/Session_03_ML_Lab(AI69201)_Demo.ipynb) | [`Assignment 2 (.ipynb)`](class-03/25AI92F02_Assignment2.ipynb) / [`.py`](class-03/25AI92F02_Assignment2.py) | Wine dataset (`load_wine`), multiclass logistic regression, standardization, one-hot encoding, metric evaluation. |
| **Class 04** | **Naive Bayes Classification** | [`Naive Bayes Lab Demo.ipynb`](class-04/Naive%20Bayes%20Lab%20Demo.ipynb) | [`Assignment 3 (.ipynb)`](class-04/25AI92F02_Assignment3.ipynb) / [`.py`](class-04/25AI92F02_Assignment3.py) | SMS Spam dataset (`sms.tsv`), text preprocessing, Bag-of-Words, Laplace smoothing, Multinomial & Gaussian NB. |
| **Class 05** | **Decision Trees & Pruning** | [`Decision_Trees_Comprehensive_Colab_Executed.ipynb`](class-05/Decision_Trees_Comprehensive_Colab_Executed.ipynb) | [`Assignment 4 (.ipynb)`](class-05/25AI92F02_Assignment4.ipynb) / [`.py`](class-05/25AI92F02_Assignment4.py) | Information Gain, Gini Impurity, Cost-Complexity Pruning ($c_{\alpha}$ / `ccp_alpha`), classification & regression trees. |
| **Class 06** | **Support Vector Machines & SVR** | [`SVM_ML_Lab_Demo_2026.ipynb`](class-06/SVM_ML_Lab_Demo_2026.ipynb) | [`Assignment 5 (.ipynb)`](class-06/25AI92F02_Assignment5.ipynb) / [`.py`](class-06/25AI92F02_Assignment5.py) | Mushroom classification, $\varepsilon$-insensitive tube Support Vector Regression (SVR), kernel trick (Linear, RBF, Poly). |
| **Class 07** | **Neural Networks & Backpropagation** | [`Feedforward_Neural_Network_and_Backpropagation_Laborator.ipynb`](class-07/Feedforward_Neural_Network_and_Backpropagation_Laborator.ipynb) | [`Assignment 6 (.ipynb)`](class-07/25AI92F02_Assignment6.ipynb) / [`.py`](class-07/25AI92F02_Assignment6.py) | Breast Cancer Wisconsin dataset, vectorized backpropagation from scratch with NumPy, gradient descent optimization. |
| **Class 08** | **Deep Sequence Models (PyTorch)** | [`Session_11_Demo_and_Assignment_RNN_CNN.ipynb`](class-08/Session_11_Demo_and_Assignment_RNN_CNN.ipynb) | [`Assignment 7 (.ipynb)`](class-08/25AI92F02_Assignment7.ipynb) / [`.py`](class-08/25AI92F02_Assignment7.py) | UCI HAR (Human Activity Recognition), 1D-CNN, RNN, GRU, LSTM, and Hybrid CNN-LSTM with temporal regularizer $\lambda$. |
| **Class 09** | **Ensemble Methods** | [`ensemble_methods_notebook.ipynb`](class-09/ensemble_methods_notebook.ipynb) | [`Assignment 8 (.ipynb)`](class-09/25AI92F02_Assignment8.ipynb) / [`.py`](class-09/25AI92F02_Assignment8.py) | Random Forest and AdaBoost.M1 from scratch, out-of-bag (OOB) error estimation, feature importance, label noise robustness. |
| **Class 10** | **Clustering Algorithms** | [`CLustering Lab Demo and Assignment.ipynb`](class-10/CLustering%20Lab%20Demo%20and%20Assignment.ipynb) | [`Assignment 9 (.ipynb)`](class-10/25AI92F02_Assignment9.ipynb) / [`.py`](class-10/25AI92F02_Assignment9.py) | UCI Seeds dataset, K-Means (Elbow method, Silhouette score), Agglomerative Hierarchical clustering, DBSCAN. |
| **Class 11** | **Dimensionality Reduction (PCA & LDA)** | [`Detailed_PCA_and_LDA_Machine_Learning_Laboratory.ipynb`](class-11/Detailed_PCA_and_LDA_Machine_Learning_Laboratory.ipynb) | [`Assignment 10 (.ipynb)`](class-11/25AI92F02_Assignment10.ipynb) / [`.py`](class-11/25AI92F02_Assignment10.py) | `load_digits`, unsupervised variance maximization (PCA), Fisher's class separability criterion (LDA), classification on latent spaces. |
| **Midsem** | **Multiclass SVM Deep Dive** | [`midsem/svm_multiclass.ipynb`](midsem/svm_multiclass.ipynb) | *N/A (Comprehensive Guide)* | Mathematical formulations of One-vs-Rest (OvR) and One-vs-One (OvO), 2D decision boundary visualization, support vectors. |

---

## 🔬 Session Details & Highlights

### Class 01 — Python Environment & Foundations
- Environment setup with modern virtual environments and package management.
- Vectorized array operations with NumPy, data structures with Pandas, and exploratory plotting with Matplotlib/Seaborn.

### Class 02 — K-Nearest Neighbors (KNN)
- Core geometric principles of instance-based learning.
- Scikit-learn KNN implementation for continuous targets (Regression).
- Hyperparameter tuning across $k \in [1, 50]$ and distance metrics ($L_1$, $L_2$, Minkowski).

### Class 03 — Linear & Logistic Regression
- Ordinary Least Squares (OLS) closed-form solution and gradient descent formulation.
- Multiclass Logistic Regression using the Scikit-learn `load_wine` dataset.
- Data standardisation, categorical feature encoding, classification reports, and confusion matrices.

### Class 04 — Naive Bayes Classification
- Probabilistic classification using Bayes' Theorem and class conditional independence assumptions.
- Text representation via Bag-of-Words and `CountVectorizer`.
- Multinomial Naive Bayes with additive Laplace smoothing for SMS Spam Classification (`sms.tsv`).

### Class 05 — Decision Trees & Pruning
- Splitting criteria: Information Gain (Entropy), Gini Impurity, and Mean Squared Error.
- Overfitting remediation via Cost-Complexity Pruning ($c_{\alpha}$ / `ccp_alpha`).
- Visualization of decision trees and feature importance analysis.

### Class 06 — Support Vector Machines & SVR
- Maximum margin hyperplane formulation and soft-margin dual optimization.
- Support Vector Regression (SVR): $\varepsilon$-insensitive loss tubes, slack variables, and regularization $C$.
- Kernel functions: Linear, Radial Basis Function (RBF), and Polynomial kernels.

### Class 07 — Neural Networks & Vectorized Backpropagation
- First-principles mathematical implementation of multi-layer perceptrons (MLP) using NumPy.
- Forward pass, non-linear activation functions (Sigmoid, ReLU, Softmax), cross-entropy / binary loss.
- Vectorized backward pass (chain rule gradient derivation) and mini-batch gradient descent on the Breast Cancer Wisconsin Diagnostic dataset.

### Class 08 — Deep Sequence Models in PyTorch
- Sequence modeling on time-series inertial sensor signals from the UCI Human Activity Recognition (HAR) dataset.
- Architecture implementation and comparative analysis:
  - 1D Convolutional Neural Networks (1D-CNN)
  - Recurrent Neural Networks (Vanilla RNN)
  - Gated Recurrent Units (GRU)
  - Long Short-Term Memory (LSTM)
  - Hybrid CNN-LSTM with temporal regularization ($\lambda$).
- Checkpoints saved: `har_cnn1d_best.pth`, `har_gru_best.pth`, `har_cnn_lstm_*.pth`.

### Class 09 — Ensemble Methods
- From-scratch implementation of **Random Forest** (Bootstrap aggregation + random feature subspacing).
- From-scratch implementation of **AdaBoost.M1** (Adaptive sample weight updating and weighted hypothesis voting).
- Performance analysis on corrupted / label-noise datasets and Out-Of-Bag (OOB) error estimation.

### Class 10 — Clustering Algorithms
- Partitional clustering: K-Means algorithm, inertia evaluation, Elbow heuristic, and Silhouette analysis.
- Hierarchical clustering: Agglomerative methods with Ward, Complete, and Average linkage criteria, dendrogram evaluation.
- Density-based spatial clustering (DBSCAN) with core points, border points, and noise detection on the UCI Seeds dataset.

### Class 11 — Dimensionality Reduction (PCA vs. LDA)
- **Principal Component Analysis (PCA):** Unsupervised variance maximization via covariance matrix spectral decomposition / SVD.
- **Linear Discriminant Analysis (LDA):** Supervised projection maximizing between-class scatter ($S_B$) relative to within-class scatter ($S_W$).
- Applied to the 64-dimensional Handwritten Digits dataset (`sklearn.datasets.load_digits`), visualising latent 2D/3D manifolds and benchmarking downstream classifiers.

### Midsem Guide — Multiclass SVM
- In-depth theoretical derivation of multi-class SVM extensions:
  - **One-vs-Rest (OvR / OvA):** $K$ binary classifiers.
  - **One-vs-One (OvO):** $\frac{K(K-1)}{2}$ binary classifiers with tournament voting.
- Visualisation of synthetic 2D multi-class decision surfaces, ambiguity regions, and highlighted support vectors.

---

## 📚 Theory Materials

The [`theory_mat/`](theory_mat/) folder contains official lecture slides and reading material covering fundamental theoretical concepts:
- **Foundations:** `Intro to ML.pdf`, `Intro_to_probability_V2-1.pdf`, `ML_logistics_ons-1.pdf`
- **Supervised Learning:** `KNN.pdf`, `Linear_and_Logisitic_Regression.pdf`, `Naive_bayes_ons-3.pdf`, `Decison Tree_FML_25_ons-3.pdf`, `SVM_ons-5.pdf`
- **Ensemble Learning:** `ensemble-models-class-notes/` (Parts 1, 2, and 3)

---

## ⚙️ Environment Setup & Installation

This project is configured using [`uv`](https://github.com/astral-sh/uv), a fast Python package installer and resolver.

### Prerequisites
- Python $\ge$ 3.14 (or Python 3.10+ compatible environment)
- NVIDIA GPU with CUDA support (optional, for PyTorch acceleration)

### Option 1: Using `uv` (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ShambaC/Machine-Learning-Lab-AI69201.git
   cd Machine-Learning-Lab-AI69201
   ```

2. **Sync the virtual environment:**
   ```bash
   uv sync
   ```

3. **Activate the environment:**
   - **Windows (PowerShell):**
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS:**
     ```bash
     source .venv/bin/activate
     ```

### Option 2: Using standard `pip`

```bash
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate
pip install -r <(python -c "import tomli, sys; ...") # or pip install numpy pandas scikit-learn matplotlib seaborn torch torchvision ucimlrepo ipykernel tqdm
```

---

## 🚀 Running Notebooks & Scripts

Launch Jupyter Lab / Notebook server:
```bash
jupyter lab
# or
jupyter notebook
```

Alternatively, each assignment includes a standalone executable Python script version (`.py`) converted from the notebooks:
```bash
python class-02/25AI92F02_Assignment1.py
python class-07/25AI92F02_Assignment6.py
python class-11/25AI92F02_Assignment10.py
```

---

## 👨‍💻 Author

- **Name:** Shamba Chowdhury
- **Course:** AI69201 — Machine Learning Laboratory
- **Affiliation:** Indian Institute of Technology (IIT) Kharagpur
- **GitHub:** [@ShambaC](https://github.com/ShambaC)

---

## 📜 License & Academic Integrity

This repository is maintained for academic coursework and reference purposes under IIT Kharagpur's academic guidelines.

