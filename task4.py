# ml_model_implementation.py

# --- Section 1: Import Libraries ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB # Or other classification models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os # To handle output directories if needed

# --- Section 2: Load the Dataset ---
from sklearn.datasets import load_iris

print("--- Loading Dataset ---")
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

print("Features (X) head:")
print(X.head())
print("\nTarget (y) head:")
print(y.head())
print("\nTarget names:", iris.target_names)

# --- Section 3: Exploratory Data Analysis (EDA) - Basic ---
print("\n--- Performing Basic EDA ---")
print("\nDataset Info:")
X.info()
print("\nDataset Description:")
print(X.describe())
print("\nMissing values:")
print(X.isnull().sum())

# Save target distribution plot to a file
plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.title('Distribution of Target Classes')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1, 2], labels=iris.target_names)
# Ensure an 'output' directory exists if you want to save plots
output_dir = 'output_plots'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'target_distribution.png'))
plt.close() # Close the plot to free memory

# --- Section 4: Data Preprocessing ---
print("\n--- Performing Data Preprocessing ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data scaled.")

# --- Section 5: Model Selection and Training ---
print("\n--- Training Model ---")
model = GaussianNB()
model.fit(X_train_scaled, y_train)
print("Model training complete.")

# --- Section 6: Model Prediction ---
print("\n--- Generating Predictions ---")
y_pred = model.predict(X_test_scaled)
print("Predictions on test set generated.")

# --- Section 7: Model Evaluation ---
print("\n--- Evaluating Model Performance ---")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
# You might want to capture this output to a file if running non-interactively
report = classification_report(y_test, y_pred, target_names=iris.target_names)
print(report)
with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
    f.write(report)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close() # Close the plot

print(f"Evaluation results (accuracy, report, confusion matrix) saved to '{output_dir}'.")

# --- Section 8: Make a Single Prediction (Optional) ---
print("\n--- Demonstrating Single Prediction ---")
new_data_point = np.array([[5.1, 3.5, 1.4, 0.2]]) # Example Iris features
new_data_point_scaled = scaler.transform(new_data_point)
predicted_class_index = model.predict(new_data_point_scaled)[0]
predicted_class_name = iris.target_names[predicted_class_index]

print(f"New data point: {new_data_point}")
print(f"Predicted class index: {predicted_class_index}")
print(f"Predicted class name: {predicted_class_name}")

print("\nScript finished.")