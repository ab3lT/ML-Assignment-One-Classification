import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Load the data
@st.cache_data
def load_data(filepath, filename):
    data = pd.read_csv(filepath + filename)

    # Replace non-numeric values with appropriate encodings
    data = data.replace({'No': 0, 'Yes': 1})

    return data

data = load_data('../data/', 'loan_data.csv')

st.title("Loan Data Dashboard")

# Display data overview
if st.sidebar.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.write(data.head())

# Display data summary
if st.sidebar.checkbox("Show data summary"):
    st.subheader("Data Summary")
    st.write(data.describe())

# Display data types
if st.sidebar.checkbox("Show data types"):
    st.subheader("Data Types")
    st.write(data.dtypes)

# Display missing values
if st.sidebar.checkbox("Show missing values"):
    st.subheader("Missing Values")
    st.write(data.isnull().sum())

# Data visualization
if st.sidebar.checkbox("Show data visualization"):
    st.subheader("Visualizations")

    # Heatmap
    if st.checkbox("Correlation Matrix"):
        st.write("### Correlation Matrix")
        numeric_data = data.select_dtypes(include=[np.number])

        st.write("### Numeric Data Preview")
        st.write(numeric_data.head())

        plt.figure(figsize=(15, 8))
        sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
        st.pyplot(plt)

    # Target label proportions
    if st.checkbox("Target Label Proportions"):
        label_prop = data['loan_status'].value_counts()
        plt.pie(label_prop.values, labels=['Rejected (0)', 'Approved (1)'], autopct='%.2f')
        plt.title('Target Label Proportions')
        st.pyplot(plt)

    # Histograms
    if st.checkbox("Numerical Columns Histogram"):
        numeric_data = data.select_dtypes(include=[np.number])
        numeric_data.hist(bins=30, figsize=(12, 10))
        st.pyplot(plt)

# Data preprocessing
st.sidebar.subheader("Data Preprocessing")
if st.sidebar.checkbox("Start Preprocessing"):
    st.subheader("Preprocessed Data")

    # Mapping and encoding
    gender_mapping = {'male': 0, 'female': 1}
    home_ownership_mapping = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3}
    loan_intent_mapping = {'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2, 'VENTURE': 3, 'HOMEIMPROVEMENT': 4, 'DEBTCONSOLIDATION': 5}
    education_mapping = {'High School': 0, 'Associate': 1, 'Bachelor': 2, 'Master': 3, 'Doctorate': 4}

    data['person_gender'] = data['person_gender'].map(gender_mapping)
    data['person_home_ownership'] = data['person_home_ownership'].map(home_ownership_mapping)
    data['loan_intent'] = data['loan_intent'].map(loan_intent_mapping)
    data['person_education'] = data['person_education'].map(education_mapping)

    st.write(data.head())

# Model training and evaluation
st.sidebar.subheader("Model Training")
if st.sidebar.checkbox("Train and Evaluate Models"):
    st.subheader("Model Training and Evaluation")

    # Feature selection
    threshold = 0.1
    correlation_matrix = data.corr()
    high_corr_features = correlation_matrix.index[abs(correlation_matrix["loan_status"]) > threshold].tolist()
    high_corr_features.remove("loan_status")
    print("======================================high_corr_features======================================")
    print(high_corr_features)
    X = data[high_corr_features]
    Y = data["loan_status"]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Logistic Regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))

    # KNN
    k = 3
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    st.write("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))

    # Confusion matrix
    if st.checkbox("Confusion Matrix"):
        conf_matrix = confusion_matrix(y_test, y_pred_knn)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Predicted Negative", "Predicted Positive"],
                    yticklabels=["Actual Negative", "Actual Positive"])
        plt.title("Confusion Matrix Heatmap")
        st.pyplot(plt)

    # Save model
    joblib.dump(knn, 'model.pkl')
    st.write("Model saved as model.pkl")

# Model testing
# st.sidebar.subheader("Model Testing")
# if st.sidebar.checkbox("Test Model"):
#     st.subheader("Test the Model with User Inputs")

#     # Load saved model
#     model = joblib.load('model.pkl')
#     import json
#     joblib.dump(knn, 'model.pkl')
#     with open('features.json', 'w') as f:
#         json.dump(high_corr_features, f)


#     # User input form
#     st.write("### Input Features")
#     person_age = st.number_input("Person Age", min_value=18, max_value=100, value=25)
#     person_income = st.number_input("Person Income", min_value=0, value=50000)
#     person_emp_exp = st.number_input("Person Employment Experience (years)", min_value=0, value=5)
#     loan_amnt = st.number_input("Loan Amount", min_value=0, value=10000)
#     loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, max_value=1.0, value=0.2)
#     cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=5)
#     credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)

#     # Ensure the input matches the training feature set
#     input_data = pd.DataFrame({
#         'person_age': [person_age],
#         'person_income': [person_income], #ok 
#         'person_home_ownership': [person_income], #ok 
#         'loan_int_rate': [person_emp_exp],
#         'loan_amnt': [loan_amnt],
#         'loan_percent_income': [loan_percent_income], #ok
#         'cb_person_cred_hist_length': [cb_person_cred_hist_length],
#         'previous_loan_defaults_on_file': [credit_score]
#     })

#     # Align with training features
#     aligned_features = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']
#     missing_features = [feature for feature in aligned_features if feature not in input_data.columns]
#     for feature in missing_features:
#         input_data[feature] = 0

#     # Ensure the column order matches
#     input_data = input_data[aligned_features]

#     # Make prediction
#     if st.button("Predict Loan Status"):
#         prediction = model.predict(input_data)
#         result = "Approved" if prediction[0] == 1 else "Rejected"
#         st.write(f"### Loan Status Prediction: {result}")
# Model testing
st.sidebar.subheader("Model Testing")
if st.sidebar.checkbox("Test Model"):
    st.subheader("Test the Model with User Inputs")

    # Load saved model
    model = joblib.load('model.pkl')

    # Define training features
    training_features = ["person_income", 
                          "person_home_ownership", 
                          "loan_amnt", 
                          "loan_int_rate", 
                          "loan_percent_income", 
                          "previous_loan_defaults_on_file"]

    # User input form
    st.write("### Input Features")
    person_income = st.number_input("Person Income", min_value=0, value=50000)
    person_home_ownership = st.selectbox("Person Home Ownership", options=["RENT", "OWN", "MORTGAGE", "OTHER"])
    loan_amnt = st.number_input("Loan Amount", min_value=0, value=10000)
    loan_int_rate = st.number_input("Loan Interest Rate", min_value=0.0, max_value=100.0, value=5.0)
    loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, max_value=1.0, value=0.2)
    previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults on File", options=["No", "Yes"])

    # Map categorical inputs to numeric values
    home_ownership_mapping = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3}
    defaults_mapping = {'No': 0, 'Yes': 1}

    input_data = pd.DataFrame({
        "person_income": [person_income],
        "person_home_ownership": [home_ownership_mapping[person_home_ownership]],
        "loan_amnt": [loan_amnt],
        "loan_int_rate": [loan_int_rate],
        "loan_percent_income": [loan_percent_income],
        "previous_loan_defaults_on_file": [defaults_mapping[previous_loan_defaults_on_file]],
    })

    # Align with training features
    for feature in training_features:
        if feature not in input_data.columns:
            input_data[feature] = 0

    # Ensure the column order matches
    input_data = input_data[training_features]

    # Make prediction
    if st.button("Predict Loan Status"):
        prediction = model.predict(input_data)
        result = "Approved" if prediction[0] == 1 else "Rejected"
        st.write(f"### Loan Status Prediction: {result}")
