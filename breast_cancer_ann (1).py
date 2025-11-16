
# Advanced ANN-style workflow using scikit-learn's MLPClassifier
# Compatible with local Jupyter / VS Code (no caas_jupyter_tools)

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay, log_loss
)
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ---------------------------- a) Data Loading & Exploration ----------------------------
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

print("Shape of X (features):", X.shape)
print("Shape of y (labels):", y.shape)

df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print("\nFirst 5 rows:")
print(df.head())

print("\nSummary statistics:")
print(df.describe())

print("\nClass distribution:")
print(df['target'].value_counts())

# ---------------------------- b) Stratified Split & Preprocessing ----------------------------
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.20, stratify=y_train_full, random_state=42
)

print("\nShapes after splitting:")
print("Train:", X_tr.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ---------------------------- c) ANN Model (MLPClassifier) ----------------------------
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    batch_size=32,
    learning_rate_init=0.001,
    alpha=0.001,           
    max_iter=1,            
    warm_start=True,
    random_state=42
)

epochs = 30
train_loss_curve = []
val_loss_curve = []
train_acc_curve = []
val_acc_curve = []

mlp.fit(X_tr_scaled, y_tr)

train_loss_curve.append(mlp.loss_)
train_acc_curve.append(accuracy_score(y_tr, mlp.predict(X_tr_scaled)))

val_pred_proba = mlp.predict_proba(X_val_scaled)[:, 1]
val_loss_curve.append(log_loss(y_val, val_pred_proba))
val_acc_curve.append(accuracy_score(y_val, mlp.predict(X_val_scaled)))

for epoch in range(1, epochs):
    mlp.fit(X_tr_scaled, y_tr)

    train_loss_curve.append(mlp.loss_)
    train_acc_curve.append(accuracy_score(y_tr, mlp.predict(X_tr_scaled)))

    val_pred_proba = mlp.predict_proba(X_val_scaled)[:, 1]
    val_loss_curve.append(log_loss(y_val, val_pred_proba))
    val_acc_curve.append(accuracy_score(y_val, mlp.predict(X_val_scaled)))

print("\nMLP Model Summary:")
print(f"Hidden layers: {mlp.hidden_layer_sizes}")
print(f"Activation: {mlp.activation}")
print(f"L2 alpha: {mlp.alpha}")
print(f"Total parameters (approx): {sum(p.size for p in mlp.coefs_ + mlp.intercepts_)}")

# ---------------------------- d) Test Evaluation ----------------------------
test_pred_proba = mlp.predict_proba(X_test_scaled)[:, 1]
test_pred = (test_pred_proba >= 0.5).astype(int)

test_accuracy = accuracy_score(y_test, test_pred)
test_precision = precision_score(y_test, test_pred)
test_recall = recall_score(y_test, test_pred)

print("\nTest Results:")
print("Accuracy:", test_accuracy)
print("Precision:", test_precision)
print("Recall:", test_recall)

cm = confusion_matrix(y_test, test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

plt.plot(train_loss_curve, label="Train Loss")
plt.plot(val_loss_curve, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()

plt.plot(train_acc_curve, label="Train Accuracy")
plt.plot(val_acc_curve, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.show()

correct = (test_pred == y_test).sum()
incorrect = (test_pred != y_test).sum()

plt.bar(["Correct", "Incorrect"], [correct, incorrect])
plt.title("Correct vs Incorrect Predictions")
plt.ylabel("Count")

for i, v in enumerate([correct, incorrect]):
    plt.text(i, v + 1, str(v), ha="center")

plt.show()

print("\nCorrect predictions:", correct)
print("Incorrect predictions:", incorrect)
