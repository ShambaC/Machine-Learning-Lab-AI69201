# %% [markdown]
# # Assignment - 2 : [20 Marks]

# %% [markdown]
# ## 1. Import required libraries and load the load_Wine Dataset from scikit-learn[MARKS 0]

# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

import warnings
warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8")
sns.set(font_scale=1.1)

from tqdm import tqdm, trange

# %%
# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target.reshape(-1, 1)

df = pd.DataFrame(X, columns=wine.feature_names)
df['Target'] = wine.target

print("Dataset Shape:", df.shape)
print("Feature Names:", df.columns.tolist())
df.head()

# %% [markdown]
# ## 2.Exploratory Data Analysis[2 Marks]
# 
# Perform the following analyses.
# 
# (a) Check
# - Missing values
# - Data types
# 
# (b) Visualize Class distribution
# 
# (c) Draw Correlation Heatmap
# 
# (d) Plot Histograms
# 
# (e) Generate Pair Plot
# 
# (f) Write your observations for each visualization.

# %%
# Missing values and data types
print("=" * 50)
print("Missing Values")
print("=" * 50)
print(df.isnull().sum())

print("\n" + "=" * 50)
print("Dataset Information")
print("=" * 50)
df.info()

# %%
# Class distribution
plt.figure(figsize=(6, 4))
class_counts = df['Target'].value_counts().sort_index()
class_counts.plot(kind='bar')
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Class Distribution of Wine Dataset")
plt.tight_layout()
plt.show()

# %%
# Correlation heatmap
plt.figure(figsize=(12, 10))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# %%
# Histograms
df.drop(columns=['Target']).hist(figsize=(16, 12), bins=20, edgecolor="black")
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.show()

# %%
# Pair plot of selected features
selected_features = ['alcohol', 'flavanoids', 'color_intensity', 'proline', 'Target']
sns.pairplot(df[selected_features], hue='Target')
plt.suptitle("Pair Plot of Selected Features", y=1.02)
plt.show()

# %% [markdown]
# ### Observations:
# - Missing values: Dataset has zero missing values across all features.
# - Data types: All 13 input features are continuous numeric (float64); target is categorical integer.
# - Class distribution: Classes are reasonably balanced (Class 0: 59, Class 1: 71, Class 2: 48).
# - Correlation: Flavanoids and total_phenols exhibit strong positive correlation (0.86); flavanoids and target show strong negative correlation (-0.85).
# - Histograms: Several features follow near-normal distributions (e.g., alcohol, hue), while others like proline and color_intensity exhibit skewness.
# - Pair plot: Features like alcohol, flavanoids, and proline provide clear separation between the three wine classes.

# %% [markdown]
# ## 3. Perform Feature Scaling and One-Hot Encoding[1 Marks]
# 
# - Use StandardScaler
# 
# - Compare Original features with Scaled features
# 
# - Display Mean , Standard deviation
# 
# - Convert the target labels into one-hot vectors.
# 
# - Display the encoded labels.

# %%
# Feature scaling using StandardScaler
X_features = df.drop(columns=['Target'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)
X_scaled_df = pd.DataFrame(X_scaled, columns=wine.feature_names)

# %%
# Compare original and scaled features
comparison = pd.concat([X_features.head(), X_scaled_df.head()], axis=1, keys=["Original Features", "Scaled Features"])
comparison

# %%
# Mean and standard deviation after scaling
scaled_stats = pd.DataFrame({
    "Mean": X_scaled_df.mean(),
    "Standard Deviation": X_scaled_df.std()
})
scaled_stats

# %%
# One-hot encoding of target labels
ohe = OneHotEncoder(sparse_output=False)
y_encoded = ohe.fit_transform(y)
y_encoded_df = pd.DataFrame(y_encoded, columns=ohe.get_feature_names_out(['Target']))
y_encoded_df.head()

# %% [markdown]
# ## 5. Split the data into training, validation, and test sets (70%-15%-15%). [Marks 1]

# %%
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y_encoded, test_size=0.30, random_state=42, stratify=wine.target
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=np.argmax(y_temp, axis=1)
)

print("=" * 50)
print("Training, Validation, and Testing Shapes")
print("=" * 50)
print(f"X_train : {X_train.shape}")
print(f"X_val   : {X_val.shape}")
print(f"X_test  : {X_test.shape}")
print()
print(f"y_train : {y_train.shape}")
print(f"y_val   : {y_val.shape}")
print(f"y_test  : {y_test.shape}")

# %% [markdown]
# # 6. Implement Elastic-Net Softmax Regression from Scratch [6 Marks]
# 
# In this step, you will implement an **Elastic-Net Regularized Softmax Regression** model from scratch using **Gradient Descent**. **Do not use any machine learning library** (e.g., Scikit-learn, TensorFlow, or PyTorch) for model training.
# 
# Your implementation must include the following components.
# 
# ---
# 
# ## (a) Softmax Function
# 
# Implement the **Softmax activation function** to convert the raw logits into class probabilities.
# 
# The Softmax function should satisfy the following properties:
# 
# - The probability of each class lies between **0 and 1**.
# - The sum of probabilities for each sample should be **equal to 1**.
# 
# ---
# 
# ## (b) Cross-Entropy Loss
# 
# Implement the **Multiclass Cross-Entropy Loss** to measure the difference between the predicted probabilities and the true class labels.
# 
# Your implementation should compute the average loss over all training samples.
# 
# ---
# 
# ## (c) L1 Regularization (Lasso)
# 
# Extend the loss function by adding an **L1 Regularization** term.
# 
# The L1 penalty is defined as
# 
# $$
# L_{L1}=\lambda_1\sum_{i,j}|W_{ij}|
# $$
# 
# where
# 
# - $W$ is the weight matrix.
# - $\lambda_1$ is the L1 regularization parameter.
# 
# ---
# 
# ## (d) L2 Regularization (Ridge)
# 
# Further extend the loss function by adding an **L2 Regularization** term.
# 
# The L2 penalty is defined as
# 
# $$
# L_{L2}=\frac{\lambda_2}{2}\sum_{i,j}W_{ij}^{2}
# $$
# 
# where
# 
# - $W$ is the weight matrix.
# - $\lambda_2$ is the L2 regularization parameter.
# 
# ---
# 
# ## (e) Gradient of L1 Regularization
# 
# Implement the gradient of the L1 regularization term.
# 
# The gradient is given by
# 
# $$
# \frac{\partial |W|}{\partial W}=\operatorname{sign}(W)
# $$
# 
# ---
# 
# ## (f) Gradient of L2 Regularization
# 
# Implement the gradient of the L2 regularization term.
# 
# The gradient is given by
# 
# $$
# \frac{\partial}{\partial W}\left(\frac{\lambda_2}{2}\|W\|_2^2\right)=\lambda_2W
# $$
# 
# ---
# 
# ## (g) Gradient Descent Optimization
# 
# Implement the complete **Gradient Descent** algorithm for training the model.
# 
# During each training epoch, your implementation must perform the following steps:
# 
# 1. Compute the logits.
# 2. Apply the Softmax function.
# 3. Compute the Cross-Entropy Loss.
# 4. Add the L1 and L2 regularization terms.
# 5. Compute the gradients of the weights and bias.
# 6. Update the model parameters using Gradient Descent.
# 7. Store the **Training Loss**.
# 8. Compute and store the **Validation Loss**.
# 
# ---
# 
# ## Expected Outputs
# 
# After completing this step, your implementation should:
# 
# - Successfully train an Elastic-Net Softmax Regression model.
# - Store the **training loss** after every epoch.
# - Store the **validation loss** after every epoch.
# - Learn the optimal weight matrix and bias vector.
# 
# > **Note**
# >
# > - You must implement the complete algorithm **from scratch**.
# > - Do **not** use any built-in machine learning library for model training.
# > - Only **NumPy** may be used for numerical computations.

# %%
class ElasticNetSoftmaxRegression:

    def __init__(self, learning_rate=0.01, n_epochs=1000, l1=0.01, l2=0.01):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.l1 = l1
        self.l2 = l2

        self.W = None
        self.b = None

        self.train_loss = []
        self.val_loss = []

    def softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def compute_loss(self, y_true, y_pred, m):
        loss = -np.mean(np.sum(y_true * np.log(y_pred + 1e-15), axis=1))

        if self.W is not None:
            loss += (self.l1 / m) * np.sum(np.abs(self.W))
            loss += (self.l2 / (2 * m)) * np.sum(self.W ** 2)

        return loss

    def fit(self, X_train, y_train, X_val, y_val):
        m, n = X_train.shape
        k = y_train.shape[1]

        self.W = np.zeros((n, k))
        self.b = np.zeros((1, k))

        for epoch in trange(self.n_epochs):
            # Forward pass
            logits = X_train @ self.W + self.b
            probabilities = self.softmax(logits)

            # Training loss
            train_loss = self.compute_loss(y_train, probabilities, m)
            self.train_loss.append(train_loss)

            # Compute gradients
            dW = (X_train.T @ (probabilities - y_train)) / m
            db = np.sum(probabilities - y_train, axis=0, keepdims=True) / m

            # Elastic-Net penalty gradients
            dW += (self.l1 / m) * np.sign(self.W)
            dW += (self.l2 / m) * self.W

            # Update parameters
            self.W -= self.lr * dW
            self.b -= self.lr * db

            # Validation loss
            val_logits = X_val @ self.W + self.b
            val_prob = self.softmax(val_logits)
            val_loss = self.compute_loss(y_val, val_prob, X_val.shape[0])
            self.val_loss.append(val_loss)

    def predict_proba(self, X):
        logits = X @ self.W + self.b
        return self.softmax(logits)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

# %% [markdown]
# # 7. Train the Model [1 Mark]
# 
# Train the **Elastic-Net Softmax Regression** model using the **Gradient Descent** algorithm.
# 
# ---
# 
# ## Training Configuration
# 
# Train the model for **1000 epochs**.
# 
# Select appropriate values for the following hyperparameters:
# 
# - **Learning Rate ($\alpha$)**
# - **L1 Regularization Parameter ($\lambda_1$)**
# - **L2 Regularization Parameter ($\lambda_2$)**
# 
# You may experiment with different values of these hyperparameters to achieve better model performance.

# %%
# Train Elastic-Net Softmax Regression model
model = ElasticNetSoftmaxRegression(learning_rate=0.05, n_epochs=1000, l1=0.01, l2=0.01)
model.fit(X_train, y_train, X_val, y_val)

# %% [markdown]
# 
# ## 8. Plot Learning Curve[1 Mark]
# 
# - Plot Training Loss and Validation Loss
# 
# - Discuss Underfitting and Overfitting

# %%
def plot_loss(train_loss, val_loss, ylabel, title):
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, linewidth=2, label="Training Loss")
    plt.plot(val_loss, '--', linewidth=2, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_loss(
    model.train_loss,
    model.val_loss,
    ylabel="Cross-Entropy Loss",
    title="Elastic-Net Softmax Regression Learning Curve"
)

# %% [markdown]
# ### Discussion on Underfitting and Overfitting:
# - Both training and validation losses decrease steadily and stabilize around 300 epochs.
# - Minimal gap between training and validation loss indicates strong generalization with no overfitting although training loss is lower than validation loss.
# - Convergence to a low loss value shows the model has captured data representations without underfitting.

# %% [markdown]
# ## 9. Predict on test set[1 Mark]
#  - display actual and predicted class

# %%
# Predict on test set
y_pred = model.predict(X_test)
y_true = np.argmax(y_test, axis=1)

results = pd.DataFrame({
    "Actual Class": y_true,
    "Predicted Class": y_pred
})

results.head(15)

# %% [markdown]
# ## 10. Evaluate the Model[1 Mark]
# 
# - compute Accuracy, precision, recall, f1_score

# %%
def classification_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

def print_metrics(title, metrics):
    print("=" * 55)
    print(title)
    print("=" * 55)
    for key, value in metrics.items():
        print(f"{key:<12}: {value:.4f}")

# %%
scratch_metrics = classification_metrics(y_true, y_pred)
print_metrics("Scratch Elastic-Net Softmax Regression Metrics", scratch_metrics)

# %% [markdown]
# ## 11. Plot the confusion matrix, Classification report and interpret the results.[1 Marks]

# %%
def plot_confusion_matrix(y_true, y_pred, title, cmap="Blues"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=cmap, values_format="d")
    plt.title(title)
    plt.grid(False)
    plt.show()

plot_confusion_matrix(y_true, y_pred, "Confusion Matrix (Scratch Implementation)")

print("Classification Report:\n")
print(classification_report(y_true, y_pred))

# %% [markdown]
# ### Interpretation:
# - The confusion matrix confirms high classification accuracy with no off-diagonal misclassifications.
# - Completely accurate precision, recall, and F1-scores across all three classes demonstrate strong discriminative ability.

# %% [markdown]
# ## 12. Perform the Sklearn Implementation and Compare the both model.[1 Marks]

# %%
# Scikit-learn Logistic Regression
sklearn_model = LogisticRegression(
    solver="saga",
    penalty="elasticnet",
    l1_ratio=0.5,
    C=1.0,
    max_iter=1000,
    random_state=42
)

sklearn_model.fit(X_train, np.argmax(y_train, axis=1))

y_pred_sklearn = sklearn_model.predict(X_test)
sklearn_metrics = classification_metrics(y_true, y_pred_sklearn)


# %%
print_metrics("Scikit-learn Logistic Regression Metrics", sklearn_metrics)
plot_confusion_matrix(y_true, y_pred_sklearn, "Confusion Matrix (Scikit-Learn)")

def compare_models(scratch_metrics, sklearn_metrics):
    comparison = pd.DataFrame({
        "Metric": list(scratch_metrics.keys()),
        "Scratch": list(scratch_metrics.values()),
        "Scikit-learn": list(sklearn_metrics.values())
    })
    return comparison.round(4)

comparison_df = compare_models(scratch_metrics, sklearn_metrics)
comparison_df

# %% [markdown]
# # 13: Effect of Regularization[3 Marks]
# 
# In this section, analyze the effect of different regularization techniques on the performance of the **Softmax Regression** model.
# 
# Train the model using the following three configurations and compare the results.
# 
# ---
# 
# ## Case 1: No Regularization
# 
# Train the model without any regularization.
# 
# **Hyperparameters**
# 
# - L1 Regularization ($\lambda_1$) = **0**
# - L2 Regularization ($\lambda_2$) = **0**
# 
# This serves as the baseline model.
# 
# ---
# 
# ## Case 2: L2 Regularization (Ridge)
# 
# Train the model using **only L2 Regularization**.
# 
# **Hyperparameters**
# 
# - L1 Regularization ($\lambda_1$) = **0**
# - L2 Regularization ($\lambda_2$) = **0.01**
# 
# Observe how L2 regularization affects the model performance and the learning curves.
# 
# ---
# 
# ## Case 3: Elastic-Net Regularization
# 
# Train the model using **both L1 and L2 Regularization**.
# 
# **Hyperparameters**
# 
# - L1 Regularization ($\lambda_1$) = **0.01**
# - L2 Regularization ($\lambda_2$) = **0.01**
# 
# Observe the combined effect of L1 and L2 regularization on the model.
# 
# ---
# 
# ## Compare the Following Performance Metrics
# 
# For each configuration, compute and compare:
# 
# - Accuracy
# - Precision
# - Recall
# - F1-Score
# - Training Loss
# - Validation Loss
# 
# Present your results in the following table.
# 
# | Model | Accuracy | Precision | Recall | F1-Score | Training Loss | Validation Loss |
# |--------|---------:|----------:|--------:|---------:|--------------:|----------------:|
# | No Regularization | | | | | | |
# | L2 Regularization | | | | | | |
# | Elastic-Net Regularization | | | | | | |
# 
# ---
# 
# 

# %%
# Case 1: No Regularization
model_none = ElasticNetSoftmaxRegression(learning_rate=0.05, n_epochs=1000, l1=0.0, l2=0.0)
model_none.fit(X_train, y_train, X_val, y_val)
y_pred_none = model_none.predict(X_test)
metrics_none = classification_metrics(y_true, y_pred_none)

# %%
# Case 2: L2 Regularization
model_l2 = ElasticNetSoftmaxRegression(learning_rate=0.05, n_epochs=1000, l1=0.0, l2=0.01)
model_l2.fit(X_train, y_train, X_val, y_val)
y_pred_l2 = model_l2.predict(X_test)
metrics_l2 = classification_metrics(y_true, y_pred_l2)

# %%
# Case 3: Elastic-Net Regularization
model_enet = ElasticNetSoftmaxRegression(learning_rate=0.05, n_epochs=1000, l1=0.01, l2=0.01)
model_enet.fit(X_train, y_train, X_val, y_val)
y_pred_enet = model_enet.predict(X_test)
metrics_enet = classification_metrics(y_true, y_pred_enet)

# %%
# Comparison plot of loss curves
plt.figure(figsize=(10, 6))
plt.plot(model_none.train_loss, label="No Reg (Train)", linestyle=":")
plt.plot(model_none.val_loss, label="No Reg (Val)", linestyle="-")
plt.plot(model_l2.train_loss, label="L2 Reg (Train)", linestyle=":")
plt.plot(model_l2.val_loss, label="L2 Reg (Val)", linestyle="-")
plt.plot(model_enet.train_loss, label="Elastic-Net (Train)", linestyle=":")
plt.plot(model_enet.val_loss, label="Elastic-Net (Val)", linestyle="-")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Regularization Effect on Learning Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Performance comparison table
reg_comparison = pd.DataFrame([
    {
        "Model": "No Regularization",
        "Accuracy": metrics_none["Accuracy"],
        "Precision": metrics_none["Precision"],
        "Recall": metrics_none["Recall"],
        "F1-Score": metrics_none["F1 Score"],
        "Training Loss": model_none.train_loss[-1],
        "Validation Loss": model_none.val_loss[-1]
    },
    {
        "Model": "L2 Regularization",
        "Accuracy": metrics_l2["Accuracy"],
        "Precision": metrics_l2["Precision"],
        "Recall": metrics_l2["Recall"],
        "F1-Score": metrics_l2["F1 Score"],
        "Training Loss": model_l2.train_loss[-1],
        "Validation Loss": model_l2.val_loss[-1]
    },
    {
        "Model": "Elastic-Net Regularization",
        "Accuracy": metrics_enet["Accuracy"],
        "Precision": metrics_enet["Precision"],
        "Recall": metrics_enet["Recall"],
        "F1-Score": metrics_enet["F1 Score"],
        "Training Loss": model_enet.train_loss[-1],
        "Validation Loss": model_enet.val_loss[-1]
    }
])

reg_comparison.round(4)

# %% [markdown]
# ## 14. Analysis[1 Marks]
# 
# Based on the experimental results, answer the following questions.
# 
# 1. Which regularization technique achieved the highest classification accuracy?
# 
# 2. How did L2 regularization affect the training and validation losses compared to the model without regularization?
# 
# 3. What impact did adding L1 regularization have on the overall model performance?
# 
# 4. Which model showed the best generalization performance on the validation and test datasets?
# 
# 5. Which regularization technique would you recommend for this dataset? Justify your answer using the obtained results.

# %% [markdown]
# 1. Highest Classification Accuracy:
#    All three configurations achieved identical perfect accuracy (1.0), precision (1.0), recall (1.0), and F1-score (1.0) on the test set.
# 
# 2. Effect of L2 Regularization on Losses:
#    L2 regularization increased training loss slightly (0.0283 -> 0.0289) and validation loss (0.0757 -> 0.0781) because the penalty term penalizes large weight magnitudes.
# 
# 3. Impact of Adding L1 Regularization:
#    Adding L1 regularization (Elastic-Net) increased total loss further (train: 0.0305, val: 0.0844) due to the sparsity penalty, while maintaining 100% classification metrics.
# 
# 4. Best Generalization Performance:
#    All models showed strong generalization with 100% test accuracy. The unregularized model achieved lowest objective loss, while L2 and Elastic-Net maintained smaller weight norms to guard against overfitting.
# 
# 5. Recommended Regularization:
#    L2 or Elastic-Net Regularization is recommended. Even though the clean Wine dataset is linearly separable without penalty, regularization constrains weight magnitudes and improves robustness on noisy real-world data.
# 


