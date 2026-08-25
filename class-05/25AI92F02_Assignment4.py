# %% [markdown]
# # Assignment 4: Decision Trees — Classification, Pruning and Regression

# %% [markdown]
# ### **Objective**
# 
# Apply Decision Tree learning using **Gini impurity** and **Information Gain (Entropy-based)**, implement the core classifier from scratch, study categorical and numerical splitting, investigate pruning, and extend the analysis to Regression Trees.
# 
# ### **Datasets**
# 
# - **Classification:** `sklearn.datasets.load_breast_cancer()`
# - **Regression:** `sklearn.datasets.load_diabetes()`
# - **Categorical mini-task:** supplied directly in this notebook
# 
# ### **Total Marks: 20**
# 
# Use `random_state=42` wherever applicable.
# 
# > **Restriction:** `DecisionTreeClassifier` must not be used to build the scratch classifier in Tasks 3–6. It may be used from Task 8 onward where explicitly requested.

# %% [markdown]
# ## Task 1: Load and Inspect the Classification Dataset
# 
# Load the Breast Cancer Wisconsin dataset and report:
# 
# - first five observations;
# - dataset shape;
# - feature names;
# - target names;
# - class counts and percentages;
# - missing-value count.
# 
# Identify the majority class.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, r2_score
)

RANDOM_STATE = 42
np.set_printoptions(precision=4, suppress=True)
pd.set_option("display.max_columns", 50)

# %%
cancer = load_breast_cancer()
X_class = cancer.data
y_class = cancer.target
cancer_df = pd.DataFrame(X_class, columns=cancer.feature_names)
cancer_df["target"] = y_class

# %%
print("First five observations:")
print(cancer_df.head().to_string(index=False))
print("\nShape:", X_class.shape)
print("Feature names:", list(cancer.feature_names))
print("Target names:", list(cancer.target_names))

class_counts = pd.Series(y_class).value_counts().sort_index()
class_report = pd.DataFrame({
    "class_name": [cancer.target_names[i] for i in class_counts.index],
    "count": class_counts.values,
    "percentage": 100 * class_counts.values / len(y_class)
}, index=class_counts.index)
print("\nClass counts and percentages:")
print(class_report.to_string(index=True, formatters={"percentage": "{:.2f}".format}))
print("\nMissing values:", int(cancer_df.isna().sum().sum()))
majority_class = int(class_counts.idxmax())
print("Majority class:", majority_class, f"({cancer.target_names[majority_class]})")

# %% [markdown]
# ## Task 2: Train–Validation–Test Split
# 
# Create **stratified** partitions:
# 
# $$
# 70\% : 15\% : 15\%.
# $$
# 
# Report:
# 
# - number of observations in each set;
# - class proportions in each set.
# 
# In one or two sentences, explain why standardization is generally unnecessary for an ordinary Decision Tree.

# %%
X_train, X_temp, y_train, y_temp = train_test_split(
    X_class, y_class, test_size=0.30, stratify=y_class,
    random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp,
    random_state=RANDOM_STATE
)

split_report = pd.DataFrame({
    "set": ["train", "validation", "test"],
    "observations": [len(y_train), len(y_val), len(y_test)],
    "class_0_proportion": [np.mean(y_train == 0), np.mean(y_val == 0), np.mean(y_test == 0)],
    "class_1_proportion": [np.mean(y_train == 1), np.mean(y_val == 1), np.mean(y_test == 1)]
})
print(split_report.to_string(index=False, formatters={
    "class_0_proportion": "{:.4f}".format,
    "class_1_proportion": "{:.4f}".format
}))

# %% [markdown]
# Standardization is generally unnecessary: trees compare each feature with thresholds. So, monotonic rescaling does not change sample ordering or candidate partitions

# %% [markdown]
# ## Task 3: Gini, Entropy and Information Gain from Scratch
# 
# Without using a tree library, implement:
# 
# 1. class-probability calculation;
# 2. Gini impurity;
# 3. Entropy;
# 4. weighted child impurity;
# 5. Gini impurity reduction;
# 6. Information Gain.
# 
# Use
# 
# $$
# G(S)=1-\sum_kp_k^2,
# $$
# 
# $$
# H(S)=-\sum_kp_k\log_2p_k,
# $$
# 
# and
# 
# $$
# IG
# =
# H(S)
# -
# \frac{|S_L|}{|S|}H(S_L)
# -
# \frac{|S_R|}{|S|}H(S_R).
# $$
# 
# Verify the functions using at least:
# 
# - one **pure** manually created label vector;
# - one **mixed** manually created label vector.
# 
# Show the calculated results.

# %%
def class_probabilities(y):
    y = np.asarray(y)
    if y.size == 0:
        return np.array([])
    _, counts = np.unique(y, return_counts=True)
    return counts / counts.sum()


def gini_impurity(y):
    probabilities = class_probabilities(y)
    return 0.0 if probabilities.size == 0 else 1.0 - np.sum(probabilities ** 2)


def entropy(y):
    probabilities = class_probabilities(y)
    if probabilities.size == 0:
        return 0.0
    return float(-np.sum(probabilities * np.log2(probabilities)))


def weighted_child_impurity(left, right, impurity_function):
    total = len(left) + len(right)
    if total == 0:
        return 0.0
    return (len(left) / total) * impurity_function(left) + (len(right) / total) * impurity_function(right)


def gini_impurity_reduction(parent, left, right):
    if len(left) == 0 or len(right) == 0:
        return 0.0
    return gini_impurity(parent) - weighted_child_impurity(left, right, gini_impurity)


def information_gain(parent, left, right):
    if len(left) == 0 or len(right) == 0:
        return 0.0
    return entropy(parent) - weighted_child_impurity(left, right, entropy)

# %%
pure_labels = np.array([1, 1, 1, 1])
mixed_labels = np.array([0, 0, 1, 1, 1, 0])
mixed_left = mixed_labels[:3]
mixed_right = mixed_labels[3:]

print("Pure labels:", pure_labels)
print(f"  probabilities={class_probabilities(pure_labels)}, Gini={gini_impurity(pure_labels):.4f}, Entropy={entropy(pure_labels):.4f}")
print("\nMixed labels:", mixed_labels)
print(f"  probabilities={class_probabilities(mixed_labels)}, Gini={gini_impurity(mixed_labels):.4f}, Entropy={entropy(mixed_labels):.4f}")
print(f"  weighted child Gini={weighted_child_impurity(mixed_left, mixed_right, gini_impurity):.4f}")
print(f"  weighted child Entropy={weighted_child_impurity(mixed_left, mixed_right, entropy):.4f}")
print(f"  Gini reduction={gini_impurity_reduction(mixed_labels, mixed_left, mixed_right):.4f}")
print(f"  Information Gain={information_gain(mixed_labels, mixed_left, mixed_right):.4f}")

# %% [markdown]
# ## Task 4: Implement a Numerical Decision Tree Classifier from Scratch
# 
# Implement your own binary Decision Tree classifier.
# 
# It must support both:
# 
# ```python
# criterion="gini"
# ```
# 
# and
# 
# ```python
# criterion="information_gain"
# ```
# 
# For `information_gain`, use Entropy as the node impurity.
# 
# Your implementation must:
# 
# 1. generate candidate thresholds for numerical features;
# 2. examine feature-threshold pairs;
# 3. calculate split quality;
# 4. select the best split;
# 5. recursively create child nodes;
# 6. stop at pure nodes;
# 7. support `max_depth`;
# 8. support `min_samples_split`;
# 9. support `min_samples_leaf`;
# 10. use majority class prediction at a leaf;
# 11. implement `fit(X, y)`;
# 12. implement `predict(X)`.
# 
# You may create helper structures/functions such as:
# 
# ```python
# Node
# _best_split()
# _grow_tree()
# _traverse_tree()
# ```
# 
# or equivalent components.

# %%
@dataclass
class TreeNode:
    feature_index: object = None
    threshold: object = None
    left: object = None
    right: object = None
    value: object = None


class ScratchDecisionTreeClassifier:
    def __init__(self, criterion="gini", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1):
        if criterion not in {"gini", "information_gain"}:
            raise ValueError("criterion must be 'gini' or 'information_gain'")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def _impurity(self, y):
        return gini_impurity(y) if self.criterion == "gini" else entropy(y)

    def _best_split(self, X, y):
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        parent_impurity = self._impurity(y)

        for feature_index in range(X.shape[1]):
            values = np.sort(np.unique(X[:, feature_index]))
            thresholds = (values[:-1] + values[1:]) / 2.0
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask
                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue
                left_y, right_y = y[left_mask], y[right_mask]
                child_impurity = weighted_child_impurity(left_y, right_y, self._impurity)
                gain = parent_impurity - child_impurity
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _grow_tree(self, X, y, depth):
        values, counts = np.unique(y, return_counts=True)
        majority_class = values[np.argmax(counts)]
        stop_for_depth = self.max_depth is not None and depth >= self.max_depth
        stop_for_size = len(y) < self.min_samples_split
        if len(values) == 1 or stop_for_depth or stop_for_size:
            return TreeNode(value=majority_class)

        feature_index, threshold, gain = self._best_split(X, y)
        if feature_index is None or gain <= 1e-12:
            return TreeNode(value=majority_class)

        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        return TreeNode(
            feature_index=feature_index,
            threshold=threshold,
            left=self._grow_tree(X[left_mask], y[left_mask], depth + 1),
            right=self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        )

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2-D and match y length")
        if self.min_samples_split < 2 or self.min_samples_leaf < 1:
            raise ValueError("invalid sample-size constraint")
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.root = self._grow_tree(X, y, depth=0)
        return self

    def _traverse_tree(self, row, node):
        if node.value is not None:
            return node.value
        if row[node.feature_index] <= node.threshold:
            return self._traverse_tree(row, node.left)
        return self._traverse_tree(row, node.right)

    def predict(self, X):
        if self.root is None:
            raise ValueError("fit must be called before predict")
        X = np.asarray(X, dtype=float)
        return np.array([self._traverse_tree(row, self.root) for row in X])

# %%
scratch_demo = ScratchDecisionTreeClassifier(
    criterion="gini", max_depth=3, min_samples_split=2, min_samples_leaf=1
).fit(X_train, y_train)
print("Scratch classifier fitted.")
print("Example predictions:", scratch_demo.predict(X_val[:10]))

# %% [markdown]
# ## Task 5: Compare Gini and Information Gain
# 
# Using the same training data and the same pre-pruning settings, train:
# 
# ```python
# criterion="gini"
# ```
# 
# and
# 
# ```python
# criterion="information_gain"
# ```
# 
# Report for both models:
# 
# - training accuracy;
# - validation accuracy;
# - validation Macro F1-score.
# 
# Also report:
# 
# - number of validation predictions on which the two models disagree.
# 
# Briefly discuss whether changing the criterion materially changes the predictions.

# %%
pre_pruning_settings = {
    "max_depth": 5,
    "min_samples_split": 2,
    "min_samples_leaf": 1
}

gini_tree = ScratchDecisionTreeClassifier(criterion="gini", **pre_pruning_settings)
information_gain_tree = ScratchDecisionTreeClassifier(
    criterion="information_gain", **pre_pruning_settings
)
gini_tree.fit(X_train, y_train)
information_gain_tree.fit(X_train, y_train)

# %%
gini_train_pred = gini_tree.predict(X_train)
gini_val_pred = gini_tree.predict(X_val)
ig_train_pred = information_gain_tree.predict(X_train)
ig_val_pred = information_gain_tree.predict(X_val)

comparison = pd.DataFrame([
    {"criterion": "gini",
     "training_accuracy": accuracy_score(y_train, gini_train_pred),
     "validation_accuracy": accuracy_score(y_val, gini_val_pred),
     "validation_macro_f1": f1_score(y_val, gini_val_pred, average="macro")},
    {"criterion": "information_gain",
     "training_accuracy": accuracy_score(y_train, ig_train_pred),
     "validation_accuracy": accuracy_score(y_val, ig_val_pred),
     "validation_macro_f1": f1_score(y_val, ig_val_pred, average="macro")}
])
print(comparison.to_string(index=False, float_format="{:.4f}".format))
disagreements = int(np.sum(gini_val_pred != ig_val_pred))
print("\nValidation disagreements:", disagreements, f"of {len(y_val)}")
print("Changing criterion changes predictions only where split rankings differ; the disagreement count measures its practical effect.")

# %% [markdown]
# ## Task 6: Pre-Pruning Experiment
# 
# Using your scratch tree, study
# 
# $$
# \texttt{max\_depth}
# \in
# \{1,2,3,4,5,6,7,8\}.
# $$
# 
# For each depth, report:
# 
# - training accuracy;
# - validation accuracy.
# 
# Plot both curves.
# 
# Then perform **one additional pre-pruning experiment** using either:
# 
# - `min_samples_split`, or
# - `min_samples_leaf`.
# 
# Select the final pre-pruning settings using validation data only.
# 
# Explain briefly:
# 
# - where underfitting appears;
# - whether deeper trees show signs of overfitting;
# - why training accuracy alone should not be used for model selection.

# %%
depths = list(range(1, 9))
depth_results = []

for depth in depths:
    model = ScratchDecisionTreeClassifier(
        criterion="gini", max_depth=depth, min_samples_split=2, min_samples_leaf=1
    ).fit(X_train, y_train)
    depth_results.append({
        "max_depth": depth,
        "training_accuracy": accuracy_score(y_train, model.predict(X_train)),
        "validation_accuracy": accuracy_score(y_val, model.predict(X_val))
    })

# %%
depth_results_df = pd.DataFrame(depth_results)
print(depth_results_df.to_string(index=False, float_format="{:.4f}".format))

plt.figure(figsize=(8, 5))
plt.plot(depth_results_df["max_depth"], depth_results_df["training_accuracy"], marker="o", label="Training")
plt.plot(depth_results_df["max_depth"], depth_results_df["validation_accuracy"], marker="o", label="Validation")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Pre-pruning: accuracy versus max_depth")
plt.xticks(depths)
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# %%
best_depth = int(depth_results_df.loc[depth_results_df["validation_accuracy"].idxmax(), "max_depth"])
leaf_sizes = [1, 2, 5, 10, 20]
leaf_results = []
for leaf_size in leaf_sizes:
    model = ScratchDecisionTreeClassifier(
        criterion="gini", max_depth=best_depth,
        min_samples_split=2, min_samples_leaf=leaf_size
    ).fit(X_train, y_train)
    leaf_results.append({
        "min_samples_leaf": leaf_size,
        "training_accuracy": accuracy_score(y_train, model.predict(X_train)),
        "validation_accuracy": accuracy_score(y_val, model.predict(X_val))
    })

# %%
leaf_results_df = pd.DataFrame(leaf_results)
print("\nAdditional pre-pruning experiment:")
print(leaf_results_df.to_string(index=False, float_format="{:.4f}".format))
best_leaf = int(leaf_results_df.loc[leaf_results_df["validation_accuracy"].idxmax(), "min_samples_leaf"])
print(f"\nSelected using validation data: max_depth={best_depth}, min_samples_leaf={best_leaf}")

plt.figure(figsize=(8, 5))
plt.plot(leaf_results_df["min_samples_leaf"], leaf_results_df["training_accuracy"], marker="o", label="Training")
plt.plot(leaf_results_df["min_samples_leaf"], leaf_results_df["validation_accuracy"], marker="o", label="Validation")
plt.xlabel("min_samples_leaf")
plt.ylabel("Accuracy")
plt.title("Pre-pruning: accuracy versus min_samples_leaf")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# %% [markdown]
# - Underfitting appears at shallow depths when both accuracies are low.
# - Overfitting is indicated when training accuracy keeps rising while validation accuracy falls.
# - Training accuracy alone rewards increasingly complex trees and cannot estimate generalization.

# %% [markdown]
# ## Task 7: Categorical Split to Leaf Nodes
# 
# Use the following categorical dataset.

# %%
categorical_assignment_df = pd.DataFrame({
    "Weather": [
        "Sunny", "Sunny", "Cloudy", "Rainy",
        "Rainy", "Cloudy", "Sunny", "Rainy"
    ],
    "Wind": [
        "Weak", "Strong", "Weak", "Weak",
        "Strong", "Strong", "Weak", "Strong"
    ],
    "Traffic": [
        "High", "High", "Low", "Low",
        "High", "Low", "Low", "High"
    ],
    "Go_Out": [
        "No", "No", "Yes", "Yes",
        "No", "Yes", "Yes", "No"
    ]
})

categorical_assignment_df

# %% [markdown]
# Without using a Decision Tree library:
# 
# 1. Calculate the parent Entropy.
# 2. Calculate Information Gain for every categorical feature.
# 3. Select the root feature.
# 4. Inspect each branch.
# 5. Split every impure branch again when possible.
# 6. Draw the final tree using Markdown/text.
# 7. Clearly identify all leaf predictions.
# 
# Show all intermediate calculations.

# %%
def categorical_information_gain(df, feature, target="Go_Out"):
    parent = df[target].to_numpy()
    weighted_entropy = 0.0
    details = []
    for value, group in df.groupby(feature, sort=True):
        labels = group[target].to_numpy()
        contribution = len(labels) / len(parent) * entropy(labels)
        weighted_entropy += contribution
        details.append((value, len(labels), group[target].value_counts().to_dict(), entropy(labels), contribution))
    return entropy(parent) - weighted_entropy, weighted_entropy, details

parent_labels = categorical_assignment_df["Go_Out"].to_numpy()
print("Parent class counts:", categorical_assignment_df["Go_Out"].value_counts().to_dict())
print(f"Parent entropy: {entropy(parent_labels):.4f}\n")

# %%
categorical_features = ["Weather", "Wind", "Traffic"]
categorical_results = []
for feature in categorical_features:
    gain, weighted, details = categorical_information_gain(categorical_assignment_df, feature)
    categorical_results.append({"feature": feature, "weighted_child_entropy": weighted, "information_gain": gain})
    print(f"{feature}: weighted entropy={weighted:.4f}, information gain={gain:.4f}")
    for value, n, counts, branch_entropy, contribution in details:
        print(f"  {value}: n={n}, counts={counts}, entropy={branch_entropy:.4f}, contribution={contribution:.4f}")

categorical_results_df = pd.DataFrame(categorical_results)
root_feature = categorical_results_df.loc[categorical_results_df["information_gain"].idxmax(), "feature"]
print(f"\nSelected root feature: {root_feature}")

# %%
print("\nBranch inspection:")
for value, branch in categorical_assignment_df.groupby(root_feature, sort=True):
    counts = branch["Go_Out"].value_counts().to_dict()
    prediction = branch["Go_Out"].mode().iloc[0]
    print(f"{root_feature}={value}: counts={counts}, entropy={entropy(branch['Go_Out'].to_numpy()):.4f}, leaf prediction={prediction}")

# %%
print("\nFinal tree:")
print(f"Root: {root_feature}")
for value, branch in categorical_assignment_df.groupby(root_feature, sort=True):
    prediction = branch["Go_Out"].mode().iloc[0]
    print(f"  |-- {root_feature} == {value} -> Leaf: Go_Out={prediction}")
print("All root branches are pure, so no second split is needed.")

# %% [markdown]
# ## Task 8: Post-Pruning with Cost Complexity
# 
# Now Scikit-learn may be used.
# 
# Train an initially unrestricted `DecisionTreeClassifier`.
# 
# Obtain:
# 
# ```python
# cost_complexity_pruning_path()
# ```
# 
# Then train models across candidate `ccp_alpha` values.
# 
# Plot:
# 
# - training accuracy versus `ccp_alpha`;
# - validation accuracy versus `ccp_alpha`.
# 
# Select an appropriate `ccp_alpha` using validation performance.
# 
# Report the selected tree's:
# 
# - depth;
# - number of leaves;
# - validation accuracy.

# %%
unrestricted_tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
unrestricted_tree.fit(X_train, y_train)
pruning_path = unrestricted_tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = pruning_path.ccp_alphas

pruning_results = []
pruned_models = []
for alpha in ccp_alphas:
    model = DecisionTreeClassifier(ccp_alpha=alpha, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    pruned_models.append(model)
    pruning_results.append({
        "ccp_alpha": alpha,
        "training_accuracy": accuracy_score(y_train, model.predict(X_train)),
        "validation_accuracy": accuracy_score(y_val, model.predict(X_val))
    })

pruning_results_df = pd.DataFrame(pruning_results)
best_index = max(
    range(len(pruned_models)),
    key=lambda i: (pruning_results_df.loc[i, "validation_accuracy"], -ccp_alphas[i])
)
selected_pruned_tree = pruned_models[best_index]

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(ccp_alphas, pruning_results_df["training_accuracy"], marker="o")
axes[0].set_title("Training accuracy")
axes[1].plot(ccp_alphas, pruning_results_df["validation_accuracy"], marker="o")
axes[1].set_title("Validation accuracy")
for ax in axes:
    ax.set_xlabel("ccp_alpha")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %%
print(pruning_results_df.to_string(index=False, float_format="{:.4f}".format))
print(f"\nSelected ccp_alpha: {ccp_alphas[best_index]:.8f}")
print("Selected tree depth:", selected_pruned_tree.get_depth())
print("Selected tree leaves:", selected_pruned_tree.get_n_leaves())
print("Selected validation accuracy:", f"{pruning_results_df.loc[best_index, 'validation_accuracy']:.4f}")

# %% [markdown]
# ## Task 9: Regression Tree
# 
# Use `load_diabetes()`.
# 
# 1. Create 70% / 15% / 15% train-validation-test partitions.
# 2. Train `DecisionTreeRegressor` for several values of `max_depth`.
# 3. Select the depth using validation RMSE.
# 4. Evaluate the selected model on the untouched test set.
# 
# Report:
# 
# - RMSE;
# - MAE;
# - $R^2$;
# - fitted depth;
# - number of leaves.
# 
# Also explain why the squared-error Regression Tree typically predicts the **mean target value** at a leaf.

# %%
diabetes = load_diabetes()
X_reg, y_reg = diabetes.data, diabetes.target
X_reg_train, X_reg_temp, y_reg_train, y_reg_temp = train_test_split(
    X_reg, y_reg, test_size=0.30, random_state=RANDOM_STATE
)
X_reg_val, X_reg_test, y_reg_val, y_reg_test = train_test_split(
    X_reg_temp, y_reg_temp, test_size=0.50, random_state=RANDOM_STATE
)

# %%
regression_depths = [1, 2, 3, 4, 5, 6, 7, 8, None]
regression_results = []
regression_models = []
for depth in regression_depths:
    model = DecisionTreeRegressor(max_depth=depth, random_state=RANDOM_STATE)
    model.fit(X_reg_train, y_reg_train)
    validation_pred = model.predict(X_reg_val)
    regression_models.append(model)
    regression_results.append({
        "max_depth": "unrestricted" if depth is None else depth,
        "validation_RMSE": np.sqrt(mean_squared_error(y_reg_val, validation_pred))
    })

regression_results_df = pd.DataFrame(regression_results)
best_regression_index = int(regression_results_df["validation_RMSE"].idxmin())
best_regression_depth = regression_depths[best_regression_index]
selected_regression_tree = regression_models[best_regression_index]
test_pred = selected_regression_tree.predict(X_reg_test)

print("Split sizes:", len(y_reg_train), len(y_reg_val), len(y_reg_test))
print(regression_results_df.to_string(index=False, float_format="{:.4f}".format))
print(f"\nSelected max_depth: {best_regression_depth}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_reg_test, test_pred)):.4f}")
print(f"Test MAE : {mean_absolute_error(y_reg_test, test_pred):.4f}")
print(f"Test R^2 : {r2_score(y_reg_test, test_pred):.4f}")
print("Fitted depth:", selected_regression_tree.get_depth())
print("Number of leaves:", selected_regression_tree.get_n_leaves())

# %% [markdown]
# Squared error is minimized by the mean of target values in each leaf, because the derivative of the sum of squared deviations is zero at that mean.


