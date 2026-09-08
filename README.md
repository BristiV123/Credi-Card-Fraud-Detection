# 💳 Credit Card Fraud Detection System

A Machine Learning based Credit Card Fraud Detection system developed using Python and Streamlit.

## 📌 Project Overview

This project analyzes credit card transaction data and uses Machine Learning to identify potentially fraudulent transactions.

The project includes:

- Dataset analysis
- Fraud vs genuine transaction analysis
- Data visualization
- Logistic Regression model
- Model evaluation
- Confusion matrix
- Interactive Streamlit dashboard
- Fraud prediction

## 🚀 Features

### 📊 Dataset Analysis
- Dataset shape
- Feature information
- Missing value analysis
- Statistical summary
- Fraud and genuine transaction count

### 📈 Data Visualization
- Fraud vs genuine transaction chart
- Fraud distribution
- Transaction percentage visualization
- Confusion matrix visualization

### 🤖 Machine Learning

The project uses:

**Logistic Regression**

The dataset is divided into:

- 80% Training Data
- 20% Testing Data

### 📋 Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

### 🔮 Fraud Prediction

The Streamlit application allows users to enter transaction feature values and receive a prediction:

- ✅ Genuine Transaction
- 🚨 Fraudulent Transaction

## 🛠️ Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

## 📁 Project Structure

```text
Credit_Card_Fraud_Detection/
│
├── Credit_Card_Fraud_Detection.py
├── app.py
├── creditcard.csv
├── requirements.txt
├── README.md
├── .gitignore
│
└── IMAGES/
    ├── fraud_vs_nonfraud.png
    ├── confusion_matrix.png
    └── fraud_distribution.png