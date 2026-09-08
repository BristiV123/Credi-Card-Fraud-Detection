import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("💳 Credit Card Fraud Detection System")
st.markdown(
    "### Machine Learning Based Fraud Detection Dashboard"
)

st.write(
    "This application analyzes credit card transactions "
    "and detects potentially fraudulent transactions using "
    "Logistic Regression."
)

# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("creditcard.csv")


try:
    data = load_data()

except FileNotFoundError:
    st.error(
        "❌ creditcard.csv file not found. "
        "Please place the dataset in the same folder as app.py."
    )
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Dashboard",
        "📊 Dataset Analysis",
        "📈 Visualizations",
        "🤖 Fraud Detection Model",
        "🔮 Prediction"
    ]
)

# ============================================================
# BASIC INFORMATION
# ============================================================

total_transactions = len(data)
total_features = data.shape[1]

fraud_count = int(data["Class"].sum())
genuine_count = total_transactions - fraud_count

fraud_percentage = (
    fraud_count / total_transactions * 100
    if total_transactions > 0
    else 0
)

# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.header("📊 Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Transactions",
            f"{total_transactions:,}"
        )

    with col2:
        st.metric(
            "Total Features",
            total_features
        )

    with col3:
        st.metric(
            "Genuine Transactions",
            f"{genuine_count:,}"
        )

    with col4:
        st.metric(
            "Fraud Transactions",
            f"{fraud_count:,}"
        )

    st.divider()

    st.subheader("🚨 Fraud Detection Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Fraud Percentage",
            f"{fraud_percentage:.4f}%"
        )

    with col2:
        st.metric(
            "Genuine Percentage",
            f"{100 - fraud_percentage:.4f}%"
        )

    st.divider()

    st.subheader("🔍 Dataset Preview")

    st.dataframe(
        data.head(10),
        width="stretch"
    )

    st.info(
        "💡 Class = 0 represents Genuine transactions "
        "and Class = 1 represents Fraud transactions."
    )


# ============================================================
# DATASET ANALYSIS
# ============================================================

elif page == "📊 Dataset Analysis":

    st.header("📊 Dataset Analysis")

    st.subheader("Dataset Shape")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", f"{data.shape[0]:,}")

    with col2:
        st.metric("Columns", data.shape[1])

    st.divider()

    st.subheader("📋 Dataset Information")

    info_df = pd.DataFrame({
        "Column": data.columns,
        "Data Type": data.dtypes.astype(str),
        "Missing Values": data.isnull().sum().values
    })

    st.dataframe(
        info_df,
        width="stretch"
    )

    st.divider()

    st.subheader("📈 Statistical Summary")

    st.dataframe(
        data.describe(),
        width="stretch"
    )

    st.divider()

    st.subheader("🚨 Fraud vs Genuine Transactions")

    class_count = data["Class"].value_counts()

    result_df = pd.DataFrame({
        "Transaction Type": [
            "Genuine",
            "Fraud"
        ],
        "Count": [
            int(class_count.get(0, 0)),
            int(class_count.get(1, 0))
        ]
    })

    st.dataframe(
        result_df,
        width="stretch"
    )


# ============================================================
# VISUALIZATIONS
# ============================================================

elif page == "📈 Visualizations":

    st.header("📈 Data Visualizations")

    # --------------------------------------------------------
    # Fraud vs Genuine Count
    # --------------------------------------------------------

    st.subheader("🚨 Fraud vs Genuine Transactions")

    fig1, ax1 = plt.subplots(figsize=(8, 5))

    sns.countplot(
        x="Class",
        data=data,
        ax=ax1
    )

    ax1.set_title("Fraud vs Non-Fraud Transactions")
    ax1.set_xlabel("Class (0 = Genuine, 1 = Fraud)")
    ax1.set_ylabel("Number of Transactions")

    st.pyplot(fig1)

    plt.close(fig1)

    # --------------------------------------------------------
    # Fraud Distribution
    # --------------------------------------------------------

    st.subheader("📊 Transaction Class Distribution")

    class_values = data["Class"].value_counts()

    fig2, ax2 = plt.subplots(figsize=(8, 5))

    class_values.plot(
        kind="bar",
        ax=ax2
    )

    ax2.set_title("Fraud Distribution")
    ax2.set_xlabel("Class")
    ax2.set_ylabel("Count")

    st.pyplot(fig2)

    plt.close(fig2)

    # --------------------------------------------------------
    # Pie Chart
    # --------------------------------------------------------

    st.subheader("🥧 Genuine vs Fraud Percentage")

    fig3, ax3 = plt.subplots(figsize=(7, 7))

    ax3.pie(
        [genuine_count, fraud_count],
        labels=["Genuine", "Fraud"],
        autopct="%1.2f%%",
        startangle=90
    )

    ax3.set_title("Transaction Distribution")

    st.pyplot(fig3)

    plt.close(fig3)


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

elif page == "🤖 Fraud Detection Model":

    st.header("🤖 Machine Learning Model")

    st.write(
        "Model used: Logistic Regression"
    )

    # Features and Target

    X = data.drop("Class", axis=1)
    y = data["Class"]

    # Train-Test Split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    st.write(
        f"Training samples: {len(X_train):,}"
    )

    st.write(
        f"Testing samples: {len(X_test):,}"
    )

    # Train Model

    with st.spinner("Training Logistic Regression model..."):

        model = LogisticRegression(
            max_iter=1000
        )

        model.fit(
            X_train,
            y_train
        )

        y_pred = model.predict(X_test)

    st.success("✅ Model trained successfully!")

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    st.subheader("📊 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            f"{accuracy:.4f}"
        )

    with col2:
        st.metric(
            "Precision",
            f"{precision:.4f}"
        )

    with col3:
        st.metric(
            "Recall",
            f"{recall:.4f}"
        )

    with col4:
        st.metric(
            "F1 Score",
            f"{f1:.4f}"
        )

    st.divider()

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    st.subheader("🔲 Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    fig4, ax4 = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        ax=ax4
    )

    ax4.set_title("Confusion Matrix")
    ax4.set_xlabel("Predicted")
    ax4.set_ylabel("Actual")

    st.pyplot(fig4)

    plt.close(fig4)

    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    st.subheader("📋 Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df,
        width="stretch"
    )


# ============================================================
# PREDICTION
# ============================================================

elif page == "🔮 Prediction":

    st.header("🔮 Credit Card Transaction Prediction")

    st.write(
        "Enter transaction feature values to predict "
        "whether the transaction is Genuine or Fraud."
    )

    # Features and Target

    X = data.drop("Class", axis=1)
    y = data["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Train Model

    with st.spinner("Preparing prediction model..."):

        model = LogisticRegression(
            max_iter=1000
        )

        model.fit(
            X_train,
            y_train
        )

    st.subheader("📝 Enter Transaction Details")

    # Create input fields dynamically

    input_values = {}

    for column in X.columns:

        default_value = float(
            data[column].median()
        )

        input_values[column] = st.number_input(
            column,
            value=default_value
        )

    # Prediction Button

    if st.button(
        "🔍 Detect Fraud",
        width="stretch"
    ):

        input_df = pd.DataFrame(
            [input_values]
        )

        prediction = model.predict(
            input_df
        )[0]

        probability = model.predict_proba(
            input_df
        )[0]

        fraud_probability = probability[1] * 100
        genuine_probability = probability[0] * 100

        st.divider()

        if prediction == 1:

            st.error(
                "🚨 FRAUDULENT TRANSACTION DETECTED"
            )

            st.metric(
                "Fraud Probability",
                f"{fraud_probability:.2f}%"
            )

        else:

            st.success(
                "✅ TRANSACTION APPEARS GENUINE"
            )

            st.metric(
                "Genuine Probability",
                f"{genuine_probability:.2f}%"
            )

        st.subheader("📊 Prediction Probability")

        probability_df = pd.DataFrame({
            "Class": [
                "Genuine",
                "Fraud"
            ],
            "Probability (%)": [
                genuine_probability,
                fraud_probability
            ]
        })

        st.dataframe(
            probability_df,
            width="stretch"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "💳 Credit Card Fraud Detection System | "
    "Python • Machine Learning • Streamlit"
)