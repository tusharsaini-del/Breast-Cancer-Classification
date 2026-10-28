
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.pipeline import Pipeline

# 1. Create output folders
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("=" * 60)
print(" BREAST CANCER CLASSIFICATION USING NEURAL NETWORK")
print("=" * 60)

# 2. Load the dataset
dataset = load_breast_cancer()

X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = dataset.target

# Dataset target: 0 = malignant, 1 = benign
print("\nDataset loaded successfully!")
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])
print("Target classes:", dataset.target_names)

print("\nClass distribution:")
print(pd.Series(y).map({0: "malignant", 1: "benign"}).value_counts())

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 4. Build a preprocessing + Neural Network pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("neural_network", MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=1000,
        early_stopping=True,
        random_state=42
    ))
])

# 5. Train the model
print("\nTraining Neural Network...")
model.fit(X_train, y_train)
print("Training completed!")

# 6. Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 7. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
report = classification_report(
    y_test,
    y_pred,
    target_names=dataset.target_names,
    digits=4,
    zero_division=0
)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"ROC-AUC:  {roc_auc:.4f}")
print("\nClassification Report:")
print(report)
print("Confusion Matrix:")
print(cm)

# 8. Save metrics to a text file
with open("results/metrics.txt", "w", encoding="utf-8") as f:
    f.write("BREAST CANCER CLASSIFICATION RESULTS\n")
    f.write("=" * 45 + "\n")
    f.write(f"Dataset samples: {len(X)}\n")
    f.write(f"Features: {X.shape[1]}\n")
    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Testing samples: {len(X_test)}\n")
    f.write(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
    f.write(f"ROC-AUC: {roc_auc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(np.array2string(cm))

# 9. Plot confusion matrix
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=dataset.target_names,
    yticklabels=dataset.target_names
)
plt.title("Confusion Matrix - Breast Cancer Classification")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=300)
plt.close()

# 10. Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"Neural Network (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Breast Cancer Classification")
plt.legend()
plt.tight_layout()
plt.savefig("results/roc_curve.png", dpi=300)
plt.close()

# 11. Plot training loss curve
nn = model.named_steps["neural_network"]

plt.figure(figsize=(7, 5))
plt.plot(nn.loss_curve_)
plt.xlabel("Training Iteration")
plt.ylabel("Loss")
plt.title("Neural Network Training Loss")
plt.tight_layout()
plt.savefig("results/training_loss.png", dpi=300)
plt.close()

# 12. Save trained model
joblib.dump(model, "models/breast_cancer_model.joblib")

print("\nFiles saved successfully:")
print("- results/metrics.txt")
print("- results/confusion_matrix.png")
print("- results/roc_curve.png")
print("- results/training_loss.png")
print("- models/breast_cancer_model.joblib")

print("\nProject execution completed!")
print("Note: This is an educational model, not a medical diagnostic tool.")
