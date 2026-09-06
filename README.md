# From-Gradients-to-Generalization-A-NumPy-First-Machine-Learning-Engine
From-scratch Machine Learning implementations using NumPy, covering Linear Regression, Logistic Regression, SGD, Momentum, Adam, KNN, and L1/L2 regularization.

«Machine Learning Foundations Masterclass — Practical Implementation Project
Developed as part of the IEEE Elevate Program»

🚀 Overview

This project is a from-scratch implementation and evaluation of fundamental Machine Learning algorithms, designed to explore what happens under the hood of modern ML models.

Instead of relying on high-level machine learning libraries for the core implementations, the project uses pure, vectorized NumPy to build the mathematical foundations of several
supervised learning algorithms, optimization engines, and regularization techniques.

The implementations are then benchmarked and validated against Scikit-learn baselines.

The project covers the complete workflow:

Data Preparation → Mathematical Formulation → Vectorized Implementation → Optimization → Regularization → Model Evaluation

---

🎯 Project Objectives

The main objectives of this project are to:

- Implement core Machine Learning algorithms from scratch.
- Translate mathematical equations into efficient vectorized NumPy operations.
- Derive and implement analytical gradients.
- Explore optimization techniques such as SGD, Momentum, and Adam.
- Understand the behavior of linear and non-linear decision boundaries.
- Implement instance-based learning using vectorized distance calculations.
- Investigate L1 and L2 regularization and their geometric effects.
- Prevent data leakage through proper dataset partitioning and scaling.
- Compare custom implementations with Scikit-learn models.
- Evaluate models using appropriate regression and classification metrics.

---

🧠 Algorithms & Concepts

1. Data Partitioning & Preprocessing

The project follows a strict three-way dataset partitioning strategy:

- 70% Training
- 15% Validation
- 15% Testing

A fixed random state of 42 is used for reproducibility.

Feature scaling is performed using Z-score standardization, with the mean and standard deviation calculated exclusively from the training set to prevent data leakage.

---

2. Linear Regression From Scratch

A fully vectorized Linear Regression implementation is developed using NumPy.

Core concepts

- Mean Squared Error (MSE)
- Analytical gradients
- Matrix multiplication
- Gradient Descent
- Normal Equations
- Parameter convergence
- Runtime comparison

The implementation uses matrix operations such as:

X.T @ error

instead of slow sample-wise Python loops.

The project also compares iterative Gradient Descent against the closed-form Normal Equation solution.

---

⚡ 3. Optimization Engines

The project implements optimization algorithms directly from their mathematical formulations.

SGD

A baseline gradient-based optimization method.

Momentum

Polyak Momentum is implemented using accumulated velocity:

vₜ = βvₜ₋₁ + (1 − β)∇J(wₜ)

with:

β = 0.9

Adam

Adam is implemented from scratch, including:

- First moment estimation
- Second moment estimation
- Bias correction
- Adaptive learning rates

Using the specified hyperparameters:

α = 0.001
β₁ = 0.9
β₂ = 0.999
ε = 10⁻⁸

Validation loss trajectories are compared across:

- Plain SGD
- SGD + Momentum
- Adam

---

🔬 4. Logistic Regression

A vectorized Logistic Regression classifier is implemented from scratch.

The implementation includes:

- Numerically stable Sigmoid activation
- Binary Cross-Entropy / Log Loss
- L2 regularization
- Analytical gradients
- Vectorized parameter updates

The sigmoid implementation includes clipping to reduce numerical overflow:

σ(z) = 1 / (1 + e⁻ᶜˡⁱᵖ(z,−250,250))

---

📐 5. Non-Linear Decision Boundaries

To demonstrate the limitations of linear classifiers, a non-linearly separable synthetic dataset is generated.

A standard linear Logistic Regression model is first applied to demonstrate its limitations.

The feature space is then expanded using a degree-3 polynomial transformation, allowing Logistic Regression to model a non-linear decision boundary.

This section connects the mathematical representation of features with the geometry of classification.

---

🔎 6. K-Nearest Neighbors From Scratch

A vectorized K-Nearest Neighbors implementation is developed using matrix-based Euclidean distance calculations.

The implementation supports:

Classification

Majority voting / mode.

Regression

Local mean averaging.

Different values of K are evaluated:

K ∈ {1, 3, 5, 11, 21, 51, 101}

The project analyzes the relationship between K and model complexity, including:

- High Variance → K = 1
- High Bias → K = 101

Train and test error curves are used to visualize this behavior.

---

🧩 7. Regularization: L1 vs L2

The project investigates the mathematical and geometric differences between:

L1 Regularization — Lasso

Encourages sparse parameter vectors and can perform feature selection by driving weights toward zero.

L2 Regularization — Ridge

Penalizes large weights while generally maintaining a dense parameter vector.

The project compares:

- Weight magnitudes
- Number of zero-valued coefficients
- Parameter profiles
- Geometric interpretation of regularization

The implementations and experiments demonstrate how the choice of regularization changes model behavior.

---

📊 Datasets

Two benchmark datasets are used.

California Housing

Task: Regression

- 8 continuous numerical features
- Continuous median house-value target

Used for:

- Linear Regression
- Optimization experiments
- Regression evaluation

Breast Cancer Wisconsin Diagnostic

Task: Binary Classification

- 30 continuous geometric features
- Two target classes:
  - Malignant
  - Benign

Used for:

- Logistic Regression
- K-NN
- L1/L2 regularization
- Classification evaluation

---

📈 Model Evaluation

The project uses standard evaluation metrics appropriate for each task.

Classification

- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1 Score

Regression

- MAE
- RMSE
- R²

The project also investigates the mathematical relationship between MAE and RMSE and why:

RMSE ≥ MAE

---

🧮 Mathematical Foundations

Beyond implementation, the project includes mathematical reflections covering:

- Why the bias term is excluded from L1/L2 regularization penalties.
- Derivation of Adam's bias-correction terms.
- Derivation of the Sigmoid derivative:

σ'(z) = σ(z)(1 − σ(z))

- Geometric interpretation of L1 and L2 regularization.
- Relationship between loss contours and regularization constraint shapes.

---

🛠️ Technologies

- Python
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook

---

🏗️ Implementation Philosophy

A major focus of this project is understanding the mathematics behind Machine Learning rather than treating models as black boxes.

The core algorithms are implemented using vectorized NumPy operations, avoiding sample-wise Python loops wherever required.

Scikit-learn is used primarily for:

- Benchmarking
- Verification
- Evaluation metrics
- Regularization experiments

This separation makes it possible to compare a mathematical implementation with production-oriented library implementations

---

💡 Key Takeaways

Through this project, I developed a deeper understanding of:

- How Machine Learning algorithms work internally.
- How mathematical gradients become executable algorithms.
- How vectorization improves computational efficiency.
- How optimizers affect convergence.
- Why non-linear feature transformations matter.
- How regularization controls model complexity.
- The bias-variance trade-off in K-NN.
- How to design reliable training, validation, and testing pipelines.
- How to evaluate and benchmark Machine Learning implementations.

---
