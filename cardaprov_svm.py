import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import warnings
import json
import os

warnings.filterwarnings("ignore")

# ================================
# 1. LOAD & DERIVE TARGET
# ================================
df = pd.read_csv('merged_card_data_random30000.csv')

# Derive target: 1 if any STATUS in ['2','3','4','5'], else 0
def derive_risk(status_series):
    bad = {'2', '3', '4', '5'}
    return 1 if any(str(s) in bad for s in status_series) else 0

client_df = df.groupby('ID').agg({
    'CODE_GENDER': 'first',
    'FLAG_OWN_CAR': 'first',
    'FLAG_OWN_REALTY': 'first',
    'CNT_CHILDREN': 'first',
    'AMT_INCOME_TOTAL': 'first',
    'NAME_INCOME_TYPE': 'first',
    'NAME_EDUCATION_TYPE': 'first',
    'NAME_FAMILY_STATUS': 'first',
    'NAME_HOUSING_TYPE': 'first',
    'DAYS_BIRTH': 'first',
    'DAYS_EMPLOYED': 'first',
    'FLAG_MOBIL': 'first',
    'FLAG_WORK_PHONE': 'first',
    'FLAG_PHONE': 'first',
    'FLAG_EMAIL': 'first',
    'OCCUPATION_TYPE': 'first',
    'CNT_FAM_MEMBERS': 'first',
    'STATUS': derive_risk
}).rename(columns={'STATUS': 'target'})

# Feature engineering
client_df['AGE'] = -client_df['DAYS_BIRTH'] / 365.25
client_df['YEARS_EMPLOYED'] = np.where(client_df['DAYS_EMPLOYED'] > 0, 0, -client_df['DAYS_EMPLOYED'] / 365.25)
client_df.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], axis=1, inplace=True)

print(f"Unique Clients: {len(client_df)}")
print(f"Target Distribution:\n{client_df['target'].value_counts(normalize=True)}")

# ================================
# 2. EDA
# ================================
print("\n" + "="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)
print("\nMissing Values:")
print(client_df.isnull().sum())

numeric_cols = client_df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('target')

# ================================
# 3. CLEANING
# ================================
print("\n" + "="*60)
print("DATA CLEANING")
print("="*60)

# Impute missing values
for col in client_df.columns:
    if client_df[col].isnull().sum() > 0:
        if client_df[col].dtype == 'object':
            client_df[col].fillna(client_df[col].mode()[0], inplace=True)
        else:
            client_df[col].fillna(client_df[col].median(), inplace=True)

# Cap outliers using IQR
def cap_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return np.clip(series, lower, upper)

for col in numeric_cols:
    client_df[col] = cap_outliers(client_df[col])

# Encode categorical
cat_cols = client_df.select_dtypes(include='object').columns.tolist()
df_encoded = pd.get_dummies(client_df, columns=cat_cols, drop_first=True)
feature_names = df_encoded.drop('target', axis=1).columns.tolist()

# ================================
# 4. PREPROCESSING
# ================================
X = df_encoded.drop('target', axis=1).values.astype(np.float64)  # Force float64
y = df_encoded['target'].values

# Standardize
mean_X = np.mean(X, axis=0)
var_X = np.var(X, axis=0)
std_X = np.sqrt(var_X + 1e-8)

X_scaled = (X - mean_X) / std_X

# Add bias term
X_scaled = np.c_[np.ones(X_scaled.shape[0]), X_scaled]
feature_names = ['bias'] + feature_names

# Convert labels: 0 -> -1, 1 -> 1
y_svm = np.where(y == 0, -1, 1)

# Train-test split
np.random.seed(42)
idx = np.random.permutation(len(y_svm))
train_size = int(0.8 * len(y_svm))
X_train, X_test = X_scaled[idx[:train_size]], X_scaled[idx[train_size:]]
y_train, y_test = y_svm[idx[:train_size]], y_svm[idx[train_size:]]

print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ================================
# 5. SVM FROM SCRATCH
# ================================
class LinearSVM:
    def __init__(self, C=1.0, lr=0.001, n_iters=1000):
        self.C = C
        self.lr = lr
        self.n_iters = n_iters
        self.w = None

    def fit(self, X, y):
        n, f = X.shape
        self.w = np.zeros(f, dtype=np.float64)
        for _ in range(self.n_iters):
            perm = np.random.permutation(n)
            X_shuf, y_shuf = X[perm], y[perm]
            for i in range(n):
                margin = y_shuf[i] * np.dot(X_shuf[i], self.w)
                if margin >= 1:
                    grad = 2 * (1/self.C) * self.w
                else:
                    grad = 2 * (1/self.C) * self.w - y_shuf[i] * X_shuf[i]
                grad = grad.astype(np.float64)
                self.w = self.w - self.lr * grad / n

    def predict(self, X):
        return np.sign(np.dot(X, self.w))

    def decision_function(self, X):
        return np.dot(X, self.w)

# ================================
# 6. HYPERPARAMETER TUNING
# ================================
print("\n" + "="*60)
print("HYPERPARAMETER TUNING")
print("="*60)

param_grid = {
    'C': [0.1, 1.0, 10.0],
    'lr': [0.001, 0.0001],
    'n_iters': [10, 20]
}

best_acc = 0
best_params = None

for C, lr, n_iters in product(param_grid['C'], param_grid['lr'], param_grid['n_iters']):
    svm = LinearSVM(C=C, lr=lr, n_iters=n_iters)
    svm.fit(X_train, y_train)
    pred = svm.predict(X_test)
    acc = np.mean(pred == y_test)
    if acc > best_acc:
        best_acc = acc
        best_params = {'C': C, 'lr': lr, 'n_iters': n_iters}
    print(f"C={C}, lr={lr}, iters={n_iters} -> Acc: {acc:.4f}")

print(f"\nBest Params: {best_params}")
print(f"Best Accuracy: {best_acc:.4f}")

# Train final model
final_svm = LinearSVM(**best_params)
final_svm.fit(X_train, y_train)

# ================================
# 7. MODEL EVALUATION
# ================================
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

y_pred = final_svm.predict(X_test)
y_scores = final_svm.decision_function(X_test)
y_pred_bin = (y_pred == 1).astype(int)
y_test_bin = (y_test == 1).astype(int)

accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.4f}")

# Ensure confusion matrix is integer
cm = np.zeros((2, 2), dtype=int)
for true, pred in zip(y_test_bin, y_pred_bin):
    cm[true][pred] += 1
tn, fp, fn, tp = cm.ravel()
print(f"Confusion Matrix:\n{cm}")

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

def compute_roc_auc(y_true, y_scores):
    desc_idx = np.argsort(y_scores)[::-1]
    y_true = y_true[desc_idx]
    tps = np.cumsum(y_true)
    fps = np.arange(1, len(y_true) + 1) - tps
    tpr = tps / tps[-1] if tps[-1] > 0 else tps
    fpr = fps / fps[-1] if fps[-1] > 0 else fps
    fpr = np.r_[0, fpr, 1]
    tpr = np.r_[0, tpr, 1]
    auc = np.trapz(tpr, fpr)
    return auc, fpr, tpr

auc, fpr, tpr = compute_roc_auc(y_test_bin, y_scores)
print(f"ROC-AUC: {auc:.4f}")

# ================================
# 8. MODEL INTERPRETATION
# ================================
print("\n" + "="*60)
print("MODEL INTERPRETATION")
print("="*60)

weights = np.abs(final_svm.w)
importances = pd.Series(weights, index=feature_names).sort_values(ascending=False)
print("\nTop 10 Risk Drivers:")
print(importances.head(10))

# ================================
# 9. SAVE MODEL ARTIFACTS
# ================================
model_artifacts = {
    'weights': final_svm.w.tolist(),
    'mean': mean_X.tolist(),
    'std': std_X.tolist(),
    'feature_names': ['bias'] + feature_names,
    'best_params': best_params,
    'numeric_cols': numeric_cols,
    'cat_cols': cat_cols
}
model_artifacts.update({
    'X_test': X_test.tolist(),
    'y_test': y_test.tolist(),
    'y_scores': y_scores.tolist(),
    'cm': cm.tolist(),
    'fpr': fpr.tolist(),
    'tpr': tpr.tolist(),
    'auc': float(auc),
    'importances': importances.to_dict()
})

os.makedirs('model', exist_ok=True)
with open('model/credit_risk_svm.json', 'w') as f:
    json.dump(model_artifacts, f)

# Add this in SECTION 9: SAVE MODEL ARTIFACTS

print("\nModel saved to 'model/credit_risk_svm.json'")

# ================================
# 10. DEPLOYMENT: USER INPUT PREDICTION
# ================================
print("\n" + "="*60)
print("DEPLOYMENT: PREDICT ON NEW RECORD")
print("="*60)

def preprocess_input(user_data):
    input_df = pd.DataFrame([user_data])
    
    # Feature engineering
    input_df['AGE'] = -input_df['DAYS_BIRTH'] / 365.25
    input_df['YEARS_EMPLOYED'] = np.where(input_df['DAYS_EMPLOYED'] > 0, 0, -input_df['DAYS_EMPLOYED'] / 365.25)
    input_df.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], axis=1, inplace=True)
    
    # Impute
    for col in input_df.columns:
        if input_df[col].isnull().sum() > 0:
            if col in numeric_cols:
                input_df[col].fillna(client_df[col].median(), inplace=True)
            else:
                input_df[col].fillna(client_df[col].mode()[0], inplace=True)
    
    # Cap outliers
    for col in numeric_cols:
        if col in input_df.columns:
            input_df[col] = cap_outliers(input_df[col])
    
    # One-hot encode
    input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)
    
    # Align columns
    for col in feature_names[1:]:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded.reindex(columns=feature_names[1:], fill_value=0)
    
    # Scale
    X_input = input_encoded.values.astype(np.float64)
    X_input = (X_input - mean_X) / std_X
    
    # Add bias
    X_input = np.c_[np.ones(1), X_input]
    return X_input

def predict_risk(user_data):
    X_input = preprocess_input(user_data)
    score = np.dot(X_input, final_svm.w)[0]
    prediction = "YES" if score > 0 else "NO"
    risk_prob = 1 / (1 + np.exp(-score))
    return prediction, risk_prob, score

# === INTERACTIVE INPUT ===
print("\n" + "-"*60)
print("INTERACTIVE PREDICTION MODE")
print("-"*60)
print("Enter customer details (press Enter for defaults):")

def get_input(prompt, default):
    val = input(f"{prompt} [{default}]: ").strip()
    if not val:
        return default
    try:
        return float(val) if '.' in val else int(val)
    except:
        print(f"Invalid input, using default: {default}")
        return default

user_input = {
    'CODE_GENDER': get_input("Gender (M/F)", 'M'),
    'FLAG_OWN_CAR': get_input("Own Car (Y/N)", 'N'),
    'FLAG_OWN_REALTY': get_input("Own Realty (Y/N)", 'Y'),
    'CNT_CHILDREN': get_input("Number of Children", 0),
    'AMT_INCOME_TOTAL': get_input("Annual Income", 180000),
    'NAME_INCOME_TYPE': get_input("Income Type", 'Working'),
    'NAME_EDUCATION_TYPE': get_input("Education", 'Secondary / secondary special'),
    'NAME_FAMILY_STATUS': get_input("Family Status", 'Married'),
    'NAME_HOUSING_TYPE': get_input("Housing Type", 'House / apartment'),
    'DAYS_BIRTH': get_input("Days Since Birth (negative)", -15000),
    'DAYS_EMPLOYED': get_input("Days Employed (negative or positive)", -2000),
    'FLAG_MOBIL': get_input("Has Mobile (1/0)", 1),
    'FLAG_WORK_PHONE': get_input("Has Work Phone (1/0)", 0),
    'FLAG_PHONE': get_input("Has Phone (1/0)", 0),
    'FLAG_EMAIL': get_input("Has Email (1/0)", 0),
    'OCCUPATION_TYPE': get_input("Occupation", ''),
    'CNT_FAM_MEMBERS': get_input("Family Members", 2)
}

pred, prob, score = predict_risk(user_input)
print(f"\nFINAL PREDICTION:")
print(f"  -> High Risk of Default: {pred}")
print(f"  -> Risk Probability: {prob:.1%}")
print(f"  -> Decision Score: {score:.4f}")

# ================================
# 11. FINAL PLOTS (FIXED: Confusion Matrix dtype)
# ================================
print("\n" + "="*60)
print("VISUALIZATION")
print("="*60)

# ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.4f})', color='navy', linewidth=2)
plt.plot([0,1], [0,1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Confusion Matrix - Fixed: Use integer matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm.astype(int), annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Low Risk', 'High Risk'],
            yticklabels=['Low Risk', 'High Risk'])
plt.title('Confusion Matrix')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# Feature Importance
plt.figure(figsize=(10, 7))
top_n = 15
importances.head(top_n).plot(kind='barh', color='teal')
plt.title(f'Top {top_n} Risk Drivers')
plt.xlabel('Absolute Weight')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Score Distribution
plt.figure(figsize=(10, 6))
plt.hist(y_scores[y_test == -1], bins=50, alpha=0.7, label='Low Risk', color='green')
plt.hist(y_scores[y_test == 1], bins=50, alpha=0.7, label='High Risk', color='red')
plt.axvline(0, color='black', linestyle='--', label='Decision Boundary')
plt.xlabel('Decision Score')
plt.ylabel('Frequency')
plt.title('Decision Score Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# User Prediction Risk Bar
plt.figure(figsize=(8, 4))
risk_pct = prob * 100
color = 'red' if pred == "YES" else 'green'
plt.barh(['Risk Level'], [risk_pct], color=color, alpha=0.8)
plt.xlim(0, 100)
plt.xlabel('Default Risk (%)')
plt.title(f'Customer Risk: {"HIGH" if pred == "YES" else "LOW"}')
plt.text(risk_pct + 2, 0, f'{risk_pct:.1f}%', va='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("DEPLOYMENT & VISUALIZATION COMPLETE")
print("Model trained, saved, and ready for real-time predictions.")
print("Use predict_risk(dict) for any new customer.")