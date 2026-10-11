# %% [markdown]
# # **Assignment 8**
# 
# **Dataset: UCI Human Activity Recognition (HAR) using Smartphones**
# Link: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
# 
# Each sample is a **2.56 s window of smartphone sensor data**: 128 time steps x 9 channels (body acc x/y/z, body gyro x/y/z, total acc x/y/z). The task is to classify the activity into 6 classes: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING.
# 
# Because this is **time-series** data, a 1D CNN (local temporal patterns), an RNN (long-range order) and a hybrid of the two are all natural choices, and you will compare them.
# 
# **Total: 20 marks**
# 
# 1. a. Load the UCI HAR dataset using the provided helper cell below, and **combine** the official train and test splits into one dataset. Print the shapes of `X` and `y`. [1 mark]
# 
#    b. Standardize each of the 9 channels, then divide the dataset into training, validation and test sets (70%-15%-15%, stratified by activity label). Compute the mean/std on the **training set only**. [1 mark]
# 
#    c. Plot the distribution of the activity labels on the training set (bar plot with class names). Plot one example window per activity class (9 channels, 6 subplots). [1 mark]
# 
# 2. a. Using PyTorch, design a **1D CNN** (`nn.Conv1d`) for activity classification. Use ReLU activation and `MaxPool1d` after every 2 conv layers, following the `SimpleCNN` structure from the demo. Note that `Conv1d` expects input of shape `(batch, channels, time)`. [2 marks]
# 
#    b. Design a **recurrent model** using `nn.GRU` or `nn.LSTM` (bidirectional optional) that classifies from the final hidden state. [2 marks]
# 
#    c. Design a **hybrid CNN-LSTM**: Conv1d layers extract local features and downsample the sequence, and the LSTM then models the resulting shorter sequence. Experiment with varying depth, number of filters and hidden size to achieve better performance. [2 marks]
# 
#    d. Design a **joint (multi-task) loss** for the CNN-LSTM. Add a second output head that predicts whether the activity is **static** (SITTING, STANDING, LAYING) or **dynamic** (the three walking classes). Combine the two losses as $\mathcal{L} = \mathcal{L}_{CE} + \lambda\,\mathcal{L}_{BCE}$. Justify your choice of $\lambda$ using at least 3 values (e.g. 0, 0.5, 1). [2 marks]
# 
# 3. a. Train all models using SGD with an appropriate minibatch size and learning rate (momentum and gradient clipping are allowed). Record the training and validation loss after every epoch. Save the best checkpoint of each model (lowest validation loss). [3 marks]
# 
#    b. Plot the training and validation losses for each model. [1 mark]
# 
# 4. a. Load the saved models and evaluate them on the test set. For each model report: classification report (accuracy, precision, recall, F1) and the confusion matrix. For the joint-loss model also report the accuracy of the static/dynamic head. [3 marks]
# 
#    b. Prepare a comparison table (parameters, test accuracy, macro-F1, training time) for the three models and write a short discussion (4-5 lines): which model is best, which classes are most confused (e.g. SITTING vs STANDING) and why, and what the recurrent component adds over a pure CNN. [2 marks]
# 
# **Submission:** a single `.ipynb` with outputs visible. Submit the model files (`.pth`) along with the notebook.

# %%
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# %% [markdown]
# ### Provided helper: dataset download and loading (run as is)

# %%
import os, zipfile, urllib.request                                      # filesystem, zip extraction, URL download

HAR_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"  # dataset download URL
SIGNALS = ['body_acc_x', 'body_acc_y', 'body_acc_z',                     # the 9 sensor channel name prefixes
           'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
           'total_acc_x', 'total_acc_y', 'total_acc_z']
ACTIVITIES = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']  # the 6 class names

def download_har(root='./har'):                                         # download + unzip the HAR dataset if needed
    os.makedirs(root, exist_ok=True)                                    # create the target directory
    zpath = os.path.join(root, 'har.zip')                                # path to the downloaded zip
    if not os.path.exists(zpath):                                       # skip re-downloading if already present
        urllib.request.urlretrieve(HAR_URL, zpath)                      # fetch the zip from the UCI archive
    with zipfile.ZipFile(zpath) as z:                                    # open the outer zip
        z.extractall(root)                                               # extract its contents
    inner = os.path.join(root, 'UCI HAR Dataset.zip')       # the archive contains a nested zip  # path to the nested zip
    if os.path.exists(inner):                                            # only if the nested zip exists
        with zipfile.ZipFile(inner) as z:                                # open the nested zip
            z.extractall(root)                                           # extract it too
    return os.path.join(root, 'UCI HAR Dataset')                        # path to the usable dataset folder

def load_split(base, split):                                            # load one split ('train' or 'test') as arrays
    chans = [
        np.loadtxt(f"{base}/{split}/Inertial Signals/{s}_{split}.txt")
        for s in tqdm(SIGNALS, desc=f"Loading {split} channels")
    ]  # load each of the 9 channel files
    X = np.stack(chans, axis=-1).astype(np.float32)          # (N, 128, 9)  # stack channels into (samples, timesteps, channels)
    y = np.loadtxt(f"{base}/{split}/y_{split}.txt", dtype=int) - 1   # labels 0..5  # load labels, shift from 1-indexed to 0-indexed
    return X, y                                                         # return features and labels

base = download_har()                                                    # ensure dataset is downloaded, get its folder path
X_tr, y_tr = load_split(base, 'train')                                  # load the official training split
X_te, y_te = load_split(base, 'test')                                   # load the official test split
print(X_tr.shape, y_tr.shape, X_te.shape, y_te.shape)       # expect (7352,128,9) (7352,) (2947,128,9) (2947,)  # sanity-check shapes

# %% [markdown]
# # Q1 — Data loading, splitting and visualization

# %%
# Combine the official splits before the 70/15/15 split.
X = np.concatenate((X_tr, X_te), axis=0)
y = np.concatenate((y_tr, y_te), axis=0)
print("X:", X.shape, "| y:", y.shape)


# %%
X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)
X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp_raw, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)

channel_mean = X_train_raw.mean(axis=(0, 1), keepdims=True)  # one mean per sensor channel
channel_std = X_train_raw.std(axis=(0, 1), keepdims=True)    # one standard deviation per channel
channel_std = np.where(channel_std == 0, 1.0, channel_std)   # avoid division by zero for a constant channel
X_train = ((X_train_raw - channel_mean) / channel_std).astype(np.float32)
X_val = ((X_val_raw - channel_mean) / channel_std).astype(np.float32)
X_test = ((X_test_raw - channel_mean) / channel_std).astype(np.float32)

def make_dataset(features, labels):
    x = torch.from_numpy(features.transpose(0, 2, 1).copy())  # (N, 128, 9) -> (N, 9, 128)
    y_class = torch.as_tensor(labels, dtype=torch.long)
    y_static = torch.as_tensor(labels >= 3, dtype=torch.float32)  # walking=0, stationary=1
    return TensorDataset(x, y_class, y_static)

train_dataset = make_dataset(X_train, y_train)
val_dataset = make_dataset(X_val, y_val)
test_dataset = make_dataset(X_test, y_test)
BATCH_SIZE = 128
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    generator=torch.Generator().manual_seed(SEED)
)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
print("Model input shape:", next(iter(val_loader))[0].shape)  # (batch, channels, time)


# %%
# Show the training split's class balance.
train_counts = np.bincount(y_train, minlength=len(ACTIVITIES))
plt.figure(figsize=(9, 4))
plt.bar(ACTIVITIES, train_counts, color="steelblue")
plt.ylabel("Training windows")
plt.title("HAR activity distribution in the training set")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()

# Plot one training window per class
rng = np.random.default_rng(SEED)
fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True)
for class_id, ax in enumerate(tqdm(axes.flat, total=len(ACTIVITIES), desc="Activity examples")):
    candidates = np.flatnonzero(y_train == class_id)
    sample = X_train[rng.choice(candidates)]
    for channel, signal_name in enumerate(tqdm(SIGNALS, desc=f"{ACTIVITIES[class_id]} channels", leave=False)):
        ax.plot(sample[:, channel], linewidth=0.9, label=signal_name)
    ax.set_title(ACTIVITIES[class_id])
    ax.set_xlabel("Time step")
    ax.set_ylabel("Standardized value")
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
fig.suptitle("Example sensor windows by activity", y=1.03)
fig.tight_layout(rect=(0, 0, 1, 0.91))
plt.show()


# %% [markdown]
# # Q2 — Model design

# %%
class CNN1D(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.conv1 = nn.Conv1d(9, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2, 2)  # halve the sequence after each pair of convolutions
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 32, 128)  # two pools reduce 128 time steps to 32
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):  # x: (batch, channels, time)
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)  # raw class logits for CrossEntropyLoss

cnn = CNN1D()
print("CNN1D parameters:", sum(p.numel() for p in cnn.parameters() if p.requires_grad))


# %%
class GRUClassifier(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, num_layers=2, num_classes=6):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers=num_layers,
            batch_first=True, dropout=0.2
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, 9, 128) -> (batch, time, features)
        _, hidden = self.gru(x)
        return self.fc(self.dropout(hidden[-1]))  # classify from the final layer's hidden state

gru = GRUClassifier()
print("GRU parameters:", sum(p.numel() for p in gru.parameters() if p.requires_grad))


# %%
class CNNLSTM(nn.Module):
    def __init__(self, filters=(32, 64, 96, 128), hidden_size=64, num_classes=6):
        super().__init__()
        self.convs = nn.ModuleList()
        in_channels = 9
        for out_channels in tqdm(filters, desc="CNN-LSTM layers", leave=False):
            self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2))
            in_channels = out_channels
        self.pool = nn.MaxPool1d(2, 2)
        self.lstm = nn.LSTM(in_channels, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.activity_head = nn.Linear(hidden_size, num_classes)

    def encode(self, x):
        for layer_index, conv in enumerate(self.convs):
            x = F.relu(conv(x))
            if (layer_index + 1) % 2 == 0:
                x = self.pool(x)  # downsample after every two convolutional layers
        _, (hidden, _) = self.lstm(x.transpose(1, 2))  # recurrent input is (batch, time, features)
        return self.dropout(hidden[-1])

    def forward(self, x):
        return self.activity_head(self.encode(x))


class JointCNNLSTM(CNNLSTM):
    def __init__(self, filters=(32, 64, 96, 128), hidden_size=64, num_classes=6):
        super().__init__(filters=filters, hidden_size=hidden_size, num_classes=num_classes)
        self.static_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        features = self.encode(x)
        return self.activity_head(features), self.static_head(features).squeeze(1)

hybrid = CNNLSTM()
print("CNN-LSTM parameters:", sum(p.numel() for p in hybrid.parameters() if p.requires_grad))


# %%
joint_model = JointCNNLSTM()
class_loss = nn.CrossEntropyLoss()
static_dynamic_loss = nn.BCEWithLogitsLoss()

print("Joint CNN-LSTM parameters:", sum(p.numel() for p in joint_model.parameters() if p.requires_grad))
print("Joint objective: CrossEntropy(activity) + lambda * BCEWithLogits(static/dynamic)")


# %% [markdown]
# # Q3 — Training

# %%
class_loss = nn.CrossEntropyLoss()
static_dynamic_loss = nn.BCEWithLogitsLoss()

def run_epoch(model, loader, optimizer=None, lambda_aux=0.0):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    samples = 0
    grad_context = torch.enable_grad() if training else torch.no_grad()
    with grad_context:
        split_name = "Training" if training else "Validation"
        for x, y_class, y_static in tqdm(loader, desc=f"{split_name} batches", leave=False):
            x = x.to(device)
            y_class = y_class.to(device)
            y_static = y_static.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            output = model(x)
            if isinstance(output, tuple):
                logits, static_logits = output
                loss = class_loss(logits, y_class) + lambda_aux * static_dynamic_loss(static_logits, y_static)
            else:
                logits = output
                loss = class_loss(logits, y_class)

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == y_class).sum().item()
            samples += batch_size
    return total_loss / samples, correct / samples


def train_model(model, name, epochs, lambda_aux=0.0, checkpoint_path=None, verbose=True):
    torch.manual_seed(SEED)  # reset dropout and data-order randomness for each run
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    train_loader.generator.manual_seed(SEED)  # use the same shuffled mini-batches for each run
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
    train_losses, val_losses, val_accuracies = [], [], []
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_state = None
    start_time = time.time()

    for epoch in tqdm(range(epochs), desc=f"{name} epochs"):
        train_loss, _ = run_epoch(model, train_loader, optimizer, lambda_aux)
        val_loss, val_accuracy = run_epoch(model, val_loader, lambda_aux=lambda_aux)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if verbose:
            print(f"[{name}] Epoch {epoch + 1:2d}/{epochs} | train {train_loss:.4f} | val {val_loss:.4f} | val acc {val_accuracy:.4f}")

    model.load_state_dict(best_state)
    if checkpoint_path is not None:
        torch.save(best_state, checkpoint_path)
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "time": time.time() - start_time,
    }


CHECKPOINT_DIR = Path("class-08") if Path("class-08").is_dir() else Path(".")
EPOCHS = 12
TUNING_EPOCHS = 3
LAMBDA_VALUES = (0.0, 0.5, 1.0)

# Compare a shallow model and two deeper/filter-width/hidden-size choices on validation data.
architecture_candidates = [
    {"filters": (32, 64), "hidden_size": 48},
    {"filters": (32, 64, 96, 128), "hidden_size": 64},
    {"filters": (32, 48, 64, 96), "hidden_size": 96},
]
tuning_results = []
for candidate in tqdm(architecture_candidates, desc="Architecture tuning"):
    candidate_name = f"tune filters={candidate['filters']} hidden={candidate['hidden_size']}"
    torch.manual_seed(SEED)  # initialize each architecture from the same seed
    candidate_result = train_model(
        CNNLSTM(**candidate), candidate_name, epochs=TUNING_EPOCHS,
        verbose=False
    )
    tuning_results.append((candidate, candidate_result))
    print(f"{candidate_name}: best validation accuracy {candidate_result['best_val_accuracy']:.4f}")

best_architecture, _ = max(tuning_results, key=lambda item: item[1]["best_val_accuracy"])
print("Selected CNN-LSTM architecture:", best_architecture)

history = {}
checkpoint_paths = {}
torch.manual_seed(SEED)
history["CNN1D"] = train_model(
    CNN1D(), "CNN1D", epochs=EPOCHS, checkpoint_path=CHECKPOINT_DIR / "har_cnn1d_best.pth"
)
checkpoint_paths["CNN1D"] = CHECKPOINT_DIR / "har_cnn1d_best.pth"
torch.manual_seed(SEED)
history["GRU"] = train_model(
    GRUClassifier(), "GRU", epochs=EPOCHS, checkpoint_path=CHECKPOINT_DIR / "har_gru_best.pth"
)
checkpoint_paths["GRU"] = CHECKPOINT_DIR / "har_gru_best.pth"

# Train the joint CNN-LSTM with three auxiliary-loss weights; select by validation activity accuracy.
lambda_histories = {}
for lambda_value in tqdm(LAMBDA_VALUES, desc="Lambda sweep"):
    lambda_tag = str(lambda_value).replace(".", "p")
    checkpoint = CHECKPOINT_DIR / f"har_cnn_lstm_lambda_{lambda_tag}_best.pth"
    torch.manual_seed(SEED)  # same initialization for a fair lambda comparison
    lambda_histories[lambda_value] = train_model(
        JointCNNLSTM(**best_architecture),
        f"CNN-LSTM lambda={lambda_value:g}",
        epochs=EPOCHS,
        lambda_aux=lambda_value,
        checkpoint_path=checkpoint,
    )

# Lambda zero is the single-task baseline; select the best positive weight for the joint model.
best_lambda = max(LAMBDA_VALUES[1:], key=lambda value: lambda_histories[value]["best_val_accuracy"])
history["CNN-LSTM"] = lambda_histories[best_lambda]
checkpoint_paths["CNN-LSTM"] = CHECKPOINT_DIR / f"har_cnn_lstm_lambda_{str(best_lambda).replace('.', 'p')}_best.pth"
print("Validation activity accuracy by lambda:")
for lambda_value in tqdm(LAMBDA_VALUES, desc="Lambda results"):
    print(f"  lambda={lambda_value:g}: {lambda_histories[lambda_value]['best_val_accuracy']:.4f}")
print(f"Selected positive lambda={best_lambda:g} by validation activity accuracy; lambda=0 remains the single-task baseline.")


# %%
# Compare the two single-task models and all three joint-loss trials.
loss_histories = {"CNN1D": history["CNN1D"], "GRU": history["GRU"]}
loss_histories.update({f"CNN-LSTM (lambda={value:g})": lambda_histories[value] for value in LAMBDA_VALUES})
fig, axes = plt.subplots(1, len(loss_histories), figsize=(4.2 * len(loss_histories), 4), squeeze=False)
for ax, (name, run) in tqdm(zip(axes[0], loss_histories.items()), total=len(loss_histories), desc="Loss curves"):
    epochs = range(1, len(run["train_losses"]) + 1)
    ax.plot(epochs, run["train_losses"], label="Training")
    ax.plot(epochs, run["val_losses"], label="Validation")
    ax.set_title(name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
plt.tight_layout()
plt.show()

# The lambda trials are compared on the same validation activity-accuracy metric.
plt.figure(figsize=(6, 4))
plt.bar([str(value) for value in LAMBDA_VALUES], [lambda_histories[value]["best_val_accuracy"] for value in LAMBDA_VALUES], color="teal")
plt.xlabel("Auxiliary loss weight (lambda)")
plt.ylabel("Best validation activity accuracy")
plt.title("CNN-LSTM lambda selection")
plt.show()


# %% [markdown]
# # Q4 — Evaluation

# %%
@torch.no_grad()
def predict(model, loader):
    model.eval()
    predictions, labels, static_predictions, static_labels = [], [], [], []
    for x, y_class, y_static in tqdm(loader, desc="Test batches", leave=False):
        output = model(x.to(device))
        if isinstance(output, tuple):
            logits, static_logits = output
            static_predictions.extend((static_logits >= 0).cpu().numpy())
        else:
            logits = output
        predictions.extend(logits.argmax(dim=1).cpu().numpy())
        labels.extend(y_class.numpy())
        static_labels.extend(y_static.numpy().astype(int))
    return np.asarray(predictions), np.asarray(labels), np.asarray(static_predictions), np.asarray(static_labels)


model_factories = {
    "CNN1D": lambda: CNN1D(),
    "GRU": lambda: GRUClassifier(),
    "CNN-LSTM": lambda: JointCNNLSTM(**best_architecture),
}
results = {}
for name, make_model in tqdm(model_factories.items(), desc="Evaluating models"):
    model = make_model().to(device)
    model.load_state_dict(torch.load(checkpoint_paths[name], map_location=device, weights_only=True))
    preds, labels, static_preds, static_labels = predict(model, test_loader)
    results[name] = {
        "preds": preds,
        "labels": labels,
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "confusion_matrix": confusion_matrix(labels, preds, labels=np.arange(len(ACTIVITIES))),
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "time": history[name]["time"],
    }
    print(f"\n===== {name} =====")
    print(classification_report(
        labels, preds, labels=np.arange(len(ACTIVITIES)),
        target_names=ACTIVITIES, digits=3, zero_division=0
    ))
    cm = results[name]["confusion_matrix"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=ACTIVITIES, yticklabels=ACTIVITIES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{name}: confusion matrix")
    plt.tight_layout()
    plt.show()

    if name == "CNN-LSTM":
        results[name]["static_accuracy"] = accuracy_score(static_labels, static_preds)
        print(f"Static/dynamic head accuracy (lambda={best_lambda:g}): {results[name]['static_accuracy']:.4f}")


# %%
print("Training time is for one final fit; architecture and lambda sweep time is excluded.")
print(f"{'Model':<12} {'Parameters':>12} {'Test accuracy':>14} {'Macro-F1':>10} {'Train time (s)':>16}")
for name, result in tqdm(results.items(), desc="Comparison table"):
    print(f"{name:<12} {result['params']:>12,d} {result['accuracy']:>14.4f} {result['macro_f1']:>10.4f} {result['time']:>16.1f}")


# %% [markdown]
# ### Discussion (Q4b)
# 
# - CNN1D performs best by test accuracy (0.9515; macro-F1 0.9545).
# - SITTING and STANDING are the most-confused pair for CNN1D, with 77 combined errors across both directions.
# - Both stationary activities have similarly low motion, while the walking activities share periodic sensor patterns.
# - The validation sweep selected lambda=0.5; its auxiliary static/dynamic head reached 1.000 test accuracy.
# - The best recurrent model reached 0.943 test accuracy; temporal context helped, but did not beat CNN1D in this run.


