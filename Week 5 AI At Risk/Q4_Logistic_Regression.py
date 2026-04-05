"""
Q4: Loan Approval Prediction using Logistic Regression
Predict whether a loan applicant's profile is relevant for approval.
Uses the Week5-Loan_Approval.xlsx dataset.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# Load data
df = pd.read_excel("Week5-Loan_Approval.xlsx", sheet_name="data")

print("=" * 80)
print("Q4: LOAN APPROVAL PREDICTION USING LOGISTIC REGRESSION")
print("=" * 80)

print(f"\nDataset shape: {df.shape}")
print(f"\nColumn info:")
print(df.info())

print(f"\n--- Missing Values ---")
print(df.isnull().sum())

# Data Cleaning
print(f"\n--- Data Cleaning ---")

# Fill missing values
df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
df["Married"].fillna(df["Married"].mode()[0], inplace=True)
df["Dependents"].fillna(df["Dependents"].mode()[0], inplace=True)
df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)
df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)
df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median(), inplace=True)
df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)

print("Missing values filled using mode (categorical) and median (numerical).")
print(f"Remaining missing values: {df.isnull().sum().sum()}")

# Convert categorical variables to numeric
print(f"\n--- Converting Categorical Variables to Numeric ---")

# Create label encoders for each categorical column
label_encoders = {}
categorical_cols = [
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "Property_Area", "Loan_Status"
]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"  {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Define features and target
feature_cols = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Property_Area"
]

X = df[feature_cols]
y = df["Loan_Status"]

print(f"\nFeatures used: {feature_cols}")
print(f"Target variable: Loan_Status")
print(f"Target distribution:\n{pd.Series(y).value_counts()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'=' * 50}")
print(f"MODEL EVALUATION")
print(f"{'=' * 50}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print(f"\nClassification Report:")
# Get target names
target_names = label_encoders["Loan_Status"].classes_
print(classification_report(y_test, y_pred, target_names=target_names))

print(f"Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  {'':<15} Predicted N  Predicted Y")
print(f"  {'Actual N':<15} {cm[0][0]:<12} {cm[0][1]}")
print(f"  {'Actual Y':<15} {cm[1][0]:<12} {cm[1][1]}")

# Feature importance (coefficients)
print(f"\nFeature Importance (Logistic Regression Coefficients):")
coef_df = pd.DataFrame({
    "Feature": feature_cols,
    "Coefficient": model.coef_[0]
}).sort_values("Coefficient", ascending=False)
for _, row in coef_df.iterrows():
    direction = "+" if row["Coefficient"] > 0 else "-"
    print(f"  {direction} {row['Feature']:<25}: {row['Coefficient']:.4f}")

# ============================================================
# PREDICT FOR THE GIVEN APPLICANT
# ============================================================
print(f"\n{'=' * 80}")
print(f"PREDICTION FOR GIVEN APPLICANT")
print(f"{'=' * 80}")

# Applicant details
applicant = {
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "2.0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 1000,
    "LoanAmount": 800,
    "Loan_Amount_Term": 240,
    "Credit_History": 1,
    "Property_Area": "Urban",
}

print(f"\nApplicant Profile:")
for key, val in applicant.items():
    print(f"  {key:<25}: {val}")

# Encode the applicant's categorical features
applicant_encoded = {}
for col in feature_cols:
    if col in label_encoders:
        val = str(applicant[col])
        le = label_encoders[col]
        if val in le.classes_:
            applicant_encoded[col] = le.transform([val])[0]
        else:
            print(f"  WARNING: '{val}' not found in {col} classes: {list(le.classes_)}")
            applicant_encoded[col] = 0
    else:
        applicant_encoded[col] = applicant[col]

# Create input array
input_data = pd.DataFrame([applicant_encoded])[feature_cols]
print(f"\nEncoded Input: {applicant_encoded}")

# Predict
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0]

# Decode prediction
pred_label = label_encoders["Loan_Status"].inverse_transform([prediction])[0]

print(f"\n{'=' * 50}")
print(f"PREDICTION RESULT")
print(f"{'=' * 50}")
print(f"  Loan Approval: {'APPROVED (Y)' if pred_label == 'Y' else 'REJECTED (N)'}")
print(f"  Probability of Rejection (N): {probability[0]:.4f} ({probability[0]*100:.2f}%)")
print(f"  Probability of Approval  (Y): {probability[1]:.4f} ({probability[1]*100:.2f}%)")
print(f"\nConclusion: The applicant with the given profile is "
      f"{'LIKELY TO BE APPROVED' if pred_label == 'Y' else 'LIKELY TO BE REJECTED'} "
      f"for the loan with {max(probability)*100:.1f}% confidence.")
