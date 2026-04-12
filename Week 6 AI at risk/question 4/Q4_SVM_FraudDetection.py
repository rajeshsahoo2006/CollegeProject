"""
BADM 566 - Homework 6, Q4
SVM Fraud Detection Model
--------------------------
Using the Week6-Fraud_data.xlsx dataset, build an SVM classifier to
predict whether transactions are Normal or Fraudulent.

Answers:
  a. Descriptive measures of 'amt' + fraud/normal counts
  b. Normalized frequency graph for category & is_fraud
  c. Confusion matrix and error analysis (Precision, Recall, Accuracy, F1)
  d. Predict a specific transaction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, accuracy_score, f1_score
)
import os
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "Week6-Fraud_data.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load Data ──────────────────────────────────────────────────────────────
print("Loading fraud data (this may take a few minutes due to file size)...")
df_train = pd.read_excel(INPUT_FILE, sheet_name="fraudTrain")
df_test = pd.read_excel(INPUT_FILE, sheet_name="fraudTest")
print(f"Training set: {len(df_train)} rows")
print(f"Testing set:  {len(df_test)} rows")

# Combine for unified preprocessing, then split back
df_train["source"] = "TRAINING"
df_test["source"] = "TESTING"
df = pd.concat([df_train, df_test], ignore_index=True)

# Drop the unnamed index column if present
if df.columns[0] in [None, "Unnamed: 0"] or str(df.columns[0]).isdigit():
    df = df.drop(df.columns[0], axis=1)

print(f"Combined dataset: {len(df)} rows, {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}\n")

# ══════════════════════════════════════════════════════════════════════════
# Q4a: Descriptive measures of 'amt' + fraud/normal counts
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Q4a: DESCRIPTIVE MEASURES OF 'amt' COLUMN")
print("=" * 70)
print(f"  Count:          {df['amt'].count()}")
print(f"  Min:            {df['amt'].min():.2f}")
print(f"  Max:            {df['amt'].max():.2f}")
print(f"  Mean:           {df['amt'].mean():.2f}")
print(f"  Std Deviation:  {df['amt'].std():.2f}")

print(f"\nFraud vs Normal Transaction Counts:")
fraud_counts = df["is_fraud"].value_counts()
print(f"  Normal (0):     {fraud_counts.get(0, 0)}")
print(f"  Fraudulent (1): {fraud_counts.get(1, 0)}")
print(f"  Fraud Rate:     {fraud_counts.get(1, 0) / len(df) * 100:.2f}%")

# ══════════════════════════════════════════════════════════════════════════
# Q4b: Normalized frequency graph for category & is_fraud
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Q4b: NORMALIZED FREQUENCY GRAPHS")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Plot 1: Category frequency by fraud status (normalized)
cat_fraud = df.groupby(["category", "is_fraud"]).size().unstack(fill_value=0)
cat_fraud_norm = cat_fraud.div(cat_fraud.sum(axis=0), axis=1)
cat_fraud_norm.plot(kind="bar", ax=axes[0], color=["steelblue", "crimson"],
                    edgecolor="black", alpha=0.8)
axes[0].set_title("Normalized Frequency: Category by Fraud Status", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Transaction Category", fontsize=11)
axes[0].set_ylabel("Normalized Frequency", fontsize=11)
axes[0].legend(["Normal (0)", "Fraud (1)"], loc="upper right")
axes[0].tick_params(axis="x", rotation=45)

# Plot 2: is_fraud distribution
fraud_dist = df["is_fraud"].value_counts(normalize=True)
fraud_dist.plot(kind="bar", ax=axes[1], color=["steelblue", "crimson"],
                edgecolor="black", alpha=0.8)
axes[1].set_title("Normalized Frequency: is_fraud Distribution", fontsize=13, fontweight="bold")
axes[1].set_xlabel("is_fraud", fontsize=11)
axes[1].set_ylabel("Normalized Frequency", fontsize=11)
axes[1].set_xticklabels(["Normal (0)", "Fraud (1)"], rotation=0)

plt.tight_layout()
graph_path = os.path.join(OUTPUT_DIR, "Q4b_normalized_frequency.png")
plt.savefig(graph_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Graph saved: {graph_path}")

# ══════════════════════════════════════════════════════════════════════════
# DATA PREPROCESSING FOR SVM
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DATA PREPROCESSING")
print("=" * 70)

# Feature engineering: add distance between customer and merchant location
df["dist_lat"] = abs(df["lat"] - df["merch_lat"])
df["dist_long"] = abs(df["long"] - df["merch_long"])
df["geo_distance"] = np.sqrt(df["dist_lat"]**2 + df["dist_long"]**2)

# Extract hour from transaction time (fraud often occurs at unusual hours)
df["trans_hour"] = pd.to_datetime(df["trans_date_trans_time"]).dt.hour

# Log of amount (helps SVM with skewed distributions)
df["log_amt"] = np.log1p(df["amt"])

# Features to use (as specified in homework)
# Drop: first, last, street, city, job, dob, trans_num, cc_num (high cardinality / identifiers)
feature_cols = ["category", "amt", "log_amt", "gender", "state", "zip",
                "lat", "long", "city_pop", "unix_time", "merch_lat", "merch_long",
                "geo_distance", "trans_hour"]

target_col = "is_fraud"

# Encode categorical variables
label_encoders = {}
categorical_cols = ["category", "gender", "state"]
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"  Encoded '{col}': {len(le.classes_)} unique values")

# Split back into train/test using Data_FLAG
train_mask = df["source"] == "TRAINING"
test_mask = df["source"] == "TESTING"

X_train = df.loc[train_mask, feature_cols].values
y_train = df.loc[train_mask, target_col].values
X_test = df.loc[test_mask, feature_cols].values
y_test = df.loc[test_mask, target_col].values

print(f"\n  Training samples: {len(X_train)}")
print(f"  Testing samples:  {len(X_test)}")
print(f"  Features used:    {feature_cols}")

# Handle missing values
X_train = pd.DataFrame(X_train, columns=feature_cols).fillna(0).values
X_test = pd.DataFrame(X_test, columns=feature_cols).fillna(0).values

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── Subsample for SVM (SVM doesn't scale well to millions of rows) ─────
# Use stratified sampling to keep fraud ratio
print("\n  Note: Subsampling data for SVM tractability...")
from sklearn.utils import resample

np.random.seed(42)
n_train_sample = 50000  # Subsample for training
n_test_sample = 20000   # Subsample for testing

# Stratified subsample for training
train_fraud_idx = np.where(y_train == 1)[0]
train_normal_idx = np.where(y_train == 0)[0]
fraud_ratio = len(train_fraud_idx) / len(y_train)
n_fraud_train = max(int(n_train_sample * fraud_ratio), len(train_fraud_idx))  # keep all fraud
n_normal_train = n_train_sample - n_fraud_train

sampled_fraud_train = train_fraud_idx  # keep ALL fraud samples
sampled_normal_train = np.random.choice(train_normal_idx, size=min(n_normal_train, len(train_normal_idx)), replace=False)
train_idx = np.concatenate([sampled_fraud_train, sampled_normal_train])
np.random.shuffle(train_idx)

X_train_s = X_train_scaled[train_idx]
y_train_s = y_train[train_idx]

# Stratified subsample for testing
test_fraud_idx = np.where(y_test == 1)[0]
test_normal_idx = np.where(y_test == 0)[0]
n_fraud_test = len(test_fraud_idx)
n_normal_test = min(n_test_sample, len(test_normal_idx))

sampled_normal_test = np.random.choice(test_normal_idx, size=n_normal_test, replace=False)
test_idx = np.concatenate([test_fraud_idx, sampled_normal_test])
np.random.shuffle(test_idx)

X_test_s = X_test_scaled[test_idx]
y_test_s = y_test[test_idx]

print(f"  Subsampled training: {len(X_train_s)} (fraud: {sum(y_train_s == 1)}, normal: {sum(y_train_s == 0)})")
print(f"  Subsampled testing:  {len(X_test_s)} (fraud: {sum(y_test_s == 1)}, normal: {sum(y_test_s == 0)})")

# ══════════════════════════════════════════════════════════════════════════
# TRAIN SVM MODEL
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TRAINING SVM MODEL")
print("=" * 70)

# Use RBF kernel SVM with class_weight='balanced' to handle class imbalance
svm_model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    class_weight="balanced",
    random_state=42,
    verbose=False
)

print("Training SVM (RBF kernel, balanced class weights)...")
svm_model.fit(X_train_s, y_train_s)
print("Training complete!")

# ══════════════════════════════════════════════════════════════════════════
# Q4c: Error Analysis - Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Q4c: ERROR ANALYSIS - CONFUSION MATRIX")
print("=" * 70)

y_pred = svm_model.predict(X_test_s)

cm = confusion_matrix(y_test_s, y_pred)
TN, FP, FN, TP = cm.ravel()

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
accuracy = (TP + TN) / (TP + FP + TN + FN)
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nConfusion Matrix:")
print(f"                  Predicted Normal  Predicted Fraud")
print(f"  Actual Normal:      {TN:>8}          {FP:>8}")
print(f"  Actual Fraud:       {FN:>8}          {TP:>8}")

print(f"\n  True Positives  (TP): {TP}")
print(f"  False Positives (FP): {FP}")
print(f"  True Negatives  (TN): {TN}")
print(f"  False Negatives (FN): {FN}")

print(f"\n  Precision (P) = TP/(TP+FP) = {TP}/({TP}+{FP}) = {precision:.4f}")
print(f"  Recall (R)    = TP/(TP+FN) = {TP}/({TP}+{FN}) = {recall:.4f}")
print(f"  Accuracy      = (TP+TN)/(TP+FP+TN+FN) = ({TP}+{TN})/({TP}+{FP}+{TN}+{FN}) = {accuracy:.4f}")
print(f"  F1 Score      = 2*P*R/(P+R) = {f1:.4f}")

# Plot confusion matrix heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal (0)", "Fraud (1)"],
            yticklabels=["Normal (0)", "Fraud (1)"],
            ax=ax, annot_kws={"size": 14})
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("Actual Label", fontsize=12)
ax.set_title("SVM Fraud Detection - Confusion Matrix", fontsize=14, fontweight="bold")
cm_path = os.path.join(OUTPUT_DIR, "Q4c_confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nConfusion matrix heatmap saved: {cm_path}")

# Full classification report
print("\nFull Classification Report:")
print(classification_report(y_test_s, y_pred, target_names=["Normal", "Fraud"]))

# ══════════════════════════════════════════════════════════════════════════
# Q4d: Predict a Specific Transaction
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Q4d: PREDICT SPECIFIC TRANSACTION")
print("=" * 70)

# Transaction details from homework
new_txn = {
    "category": "shopping_pos",
    "amt": 1318.89,
    "log_amt": np.log1p(1318.89),
    "gender": "M",
    "state": "AK",
    "zip": 99921,
    "lat": 55.4732,
    "long": -133.1171,
    "city_pop": 1920,
    "unix_time": 1386023256,
    "merch_lat": 54.801713,
    "merch_long": -133.669108,
    "geo_distance": np.sqrt((55.4732 - 54.801713)**2 + (-133.1171 - (-133.669108))**2),
    "trans_hour": 22,  # 22:27
}

print("\nTransaction Details:")
print(f"  trans_date_trans_time: 12/2/2020 22:27")
print(f"  cc_num:      3.588E+15")
print(f"  merchant:    fraud_Torphy-Goyette")
print(f"  category:    shopping_pos")
print(f"  amt:         1318.89")
print(f"  name:        Jason Johnson")
print(f"  gender:      M")
print(f"  location:    5942 Thomas Park, Craig, AK 99921")
print(f"  lat/long:    55.4732 / -133.1171")
print(f"  city_pop:    1920")
print(f"  job:         Commissioning editor")
print(f"  dob:         6/17/1997")
print(f"  unix_time:   1386023256")
print(f"  merch_lat:   54.801713")
print(f"  merch_long:  -133.669108")
print(f"  Actual:      is_fraud = 1 (Fraudulent)")

# Encode categorical values using the same encoders
for col in categorical_cols:
    if col in new_txn:
        le = label_encoders[col]
        if new_txn[col] in le.classes_:
            new_txn[col] = le.transform([new_txn[col]])[0]
        else:
            # Handle unseen category: use -1
            new_txn[col] = -1

# Create feature vector
new_txn_features = np.array([[new_txn[col] for col in feature_cols]])
new_txn_scaled = scaler.transform(new_txn_features)

prediction = svm_model.predict(new_txn_scaled)[0]

print(f"\n  >> SVM Prediction: {'FRAUDULENT (1)' if prediction == 1 else 'NORMAL (0)'}")
print(f"  >> Actual Label:   FRAUDULENT (1)")
print(f"  >> Prediction {'CORRECT' if prediction == 1 else 'INCORRECT'}")

print("\n" + "=" * 70)
print("ALL Q4 TASKS COMPLETE")
print("=" * 70)
