import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, log_loss
)
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Breast Cancer Classification using ANN (MLPClassifier)")

# Dataset options
st.sidebar.header("Dataset Options")
use_default = st.sidebar.checkbox("Use default Breast Cancer dataset", True)
uploaded_file = None
if not use_default:
    uploaded_file = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])

if use_default:
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
else:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        X = df.drop("target", axis=1).values
        y = df["target"].values
        feature_names = df.drop("target", axis=1).columns
        target_names = ["benign", "malignant"]
    else:
        st.warning("Upload a dataset or use the default dataset.")
        st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head())
st.subheader("Class Distribution")
st.bar_chart(df['target'].value_counts())

# Train/Validation/Test split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
)

scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# MLP Hyperparameters
st.sidebar.header("MLP Hyperparameters")
hidden_layers = st.sidebar.text_input("Hidden Layers (comma-separated)", "64,32")
activation = st.sidebar.selectbox("Activation Function", ["relu", "tanh", "logistic"])
alpha = st.sidebar.number_input("L2 Regularization (alpha)", 0.0, 0.01, 0.001, step=0.001)
learning_rate = st.sidebar.number_input("Learning Rate", 0.0001, 0.01, 0.001, step=0.0001)
epochs = st.sidebar.slider("Epochs", 5, 100, 30)
hidden_layers_tuple = tuple(int(x.strip()) for x in hidden_layers.split(","))

mlp = MLPClassifier(
    hidden_layer_sizes=hidden_layers_tuple,
    activation=activation,
    solver='adam',
    alpha=alpha,
    learning_rate_init=learning_rate,
    max_iter=1,
    warm_start=True,
    random_state=42
)

train_loss_curve = []
val_loss_curve = []
train_acc_curve = []
val_acc_curve = []

for epoch in range(epochs):
    mlp.fit(X_tr_scaled, y_tr)
    train_loss_curve.append(mlp.loss_)
    train_acc_curve.append(accuracy_score(y_tr, mlp.predict(X_tr_scaled)))
    val_pred_proba = mlp.predict_proba(X_val_scaled)[:, 1]
    val_loss_curve.append(log_loss(y_val, val_pred_proba))
    val_acc_curve.append(accuracy_score(y_val, mlp.predict(X_val_scaled)))

st.subheader("MLP Model Summary")
st.write(f"Hidden layers: {mlp.hidden_layer_sizes}")
st.write(f"Activation: {mlp.activation}")
st.write(f"L2 alpha: {mlp.alpha}")
st.write(f"Total parameters (approx): {sum(p.size for p in mlp.coefs_ + mlp.intercepts_)}")

# Test Evaluation
test_pred_proba = mlp.predict_proba(X_test_scaled)[:, 1]
test_pred = (test_pred_proba >= 0.5).astype(int)
test_accuracy = accuracy_score(y_test, test_pred)
test_precision = precision_score(y_test, test_pred)
test_recall = recall_score(y_test, test_pred)

st.subheader("Test Results")
st.write(f"Accuracy: {test_accuracy:.4f}")
st.write(f"Precision: {test_precision:.4f}")
st.write(f"Recall: {test_recall:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, test_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
st.pyplot(fig)

# Training Curves
st.subheader("Training Curves")
fig1, ax1 = plt.subplots()
ax1.plot(train_loss_curve, label="Train Loss")
ax1.plot(val_loss_curve, label="Validation Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Loss Curve")
ax1.legend()
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
ax2.plot(train_acc_curve, label="Train Accuracy")
ax2.plot(val_acc_curve, label="Validation Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Accuracy Curve")
ax2.legend()
st.pyplot(fig2)

# Correct vs Incorrect
correct = (test_pred == y_test).sum()
incorrect = (test_pred != y_test).sum()
st.subheader("Correct vs Incorrect Predictions")
fig3, ax3 = plt.subplots()
ax3.bar(["Correct", "Incorrect"], [correct, incorrect])
for i, v in enumerate([correct, incorrect]):
    ax3.text(i, v + 1, str(v), ha="center")
ax3.set_ylabel("Count")
ax3.set_title("Prediction Results")
st.pyplot(fig3)