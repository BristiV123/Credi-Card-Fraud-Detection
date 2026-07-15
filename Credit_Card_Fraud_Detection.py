import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
try:
    data = pd.read_csv("creditcard.csv")
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: creditcard.csv file not found.")
    exit()

# Display first 5 rows
print(data.head())

# Dataset Information
print(data.info())

# Missing Values
print(data.isnull().sum())

# Dataset Shape
print("Dataset Shape:", data.shape)

# Step 5: Fraud vs Non-Fraud Analysis

print("\nTransaction Class Count:")
print(data["Class"].value_counts())

# Count Plot
plt.figure(figsize=(6,4))

sns.countplot(x="Class", data=data)

plt.title("Fraud vs Non-Fraud Transactions")
plt.xlabel("Class (0 = Genuine, 1 = Fraud)")
plt.ylabel("Number of Transactions")

plt.savefig("IMAGES/fraud_vs_nonfraud.png", dpi=300, bbox_inches="tight")

plt.show()

# Step 6: Features and Target

X = data.drop("Class", axis=1)
y = data["Class"]

# Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression Model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("Model trained successfully!")

# Step 7: Prediction

y_pred = model.predict(X_test)

# Model Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Confusion Matrix Graph

from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.savefig("IMAGES/confusion_matrix.png", dpi=300, bbox_inches="tight")

plt.show()

# Fraud Distribution Graph

plt.figure(figsize=(6,4))

data["Class"].value_counts().plot(kind="bar", color=["green", "red"])

plt.title("Fraud vs Non-Fraud Transactions")
plt.xlabel("Class (0 = Genuine, 1 = Fraud)")
plt.ylabel("Count")

plt.savefig("IMAGES/fraud_distribution.png", dpi=300, bbox_inches="tight")

plt.show()