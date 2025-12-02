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
# 1. LOAD DATA
# ================================
df = pd.read_csv('credefault.csv')
df = df.drop('ID', axis=1)

print("Data Shape:", df.shape)

# ================================
# 2. EXPLORATORY DATA ANALYSIS
# ================================
print("\n" + "="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Distribution:")
print(df['default.payment.next.month'].value_counts(normalize=True))

# ================================
# 3. DATA CLEANING & PREPROCESSING
# ================================
print("\n" + "="*60)
print("DATA CLEANING & PREPROCESSING")
print("="*60)

# 3.1: Fill missing
df.fillna(df.median(numeric_only=True), inplace=True)

# 3.2: Cap outliers using IQR
def cap_outliers(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)
    return df

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('default.payment.next.month')
df = cap_outliers(df, numeric_cols)

# Clip negative bill amounts
bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
df[bill_cols] = df[bill_cols].clip(lower=0)

# 3.3: One-hot encode categorical
cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Save feature names after encoding
feature_names = df_encoded.drop('default.payment.next.month', axis=1).columns.tolist()

# 3.4: Features and target
X = df_encoded.drop('default.payment.next.month', axis=1).values.astype(np.float64)
y = df_encoded['default.payment.next.month'].values

# 3.5: Standardize (safe)
mean_X = np.mean(X, axis=0)
std_X = np.std(X, axis=0, ddof=1)
std_X = np.where(std_X == 0, 1.0, std_X)
std_X = np.clip(std_X, 1e-8, None)
X_scaled = (X - mean_X) / std_X

# Add bias term
X_scaled = np.c_[np.ones(X_scaled.shape[0]), X_scaled]
feature_names = ['bias'] + feature_names

# Convert labels: 0 to -1, 1 to 1
y_svm = np.where(y == 0, -1, 1)

# ================================
# 4. TRAIN-TEST SPLIT
# ================================
np.random.seed(42)
indices = np.random.permutation(len(y_svm))
train_size = int(0.8 * len(y_svm))

X_train = X_scaled[indices[:train_size]]
X_test  = X_scaled[indices[train_size:]]
y_train = y_svm[indices[:train_size]]
y_test  = y_svm[indices[train_size:]]

print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ================================
# 5. SVM FROM SCRATCH
# ================================
class LinearSVM:
    def __init__(self, C=1.0, lr=0.001, n_iters=1000):
        self.C = float(C)
        self.lr = float(lr)
        self.n_iters = int(n_iters)
        self.w = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features, dtype=np.float64)

        for _ in range(self.n_iters):
            idx = np.random.permutation(n_samples)
            X_shuf, y_shuf = X[idx], y[idx]
            for i in range(n_samples):
                x_i, y_i = X_shuf[i], y_shuf[i]
                margin = y_i * np.dot(x_i, self.w)
                if margin >= 1:
                    grad = (1.0 / self.C) * self.w
                else:
                    grad = (1.0 / self.C) * self.w - y_i * x_i
                self.w = self.w - self.lr * grad / n_samples

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
    'n_iters': [10]
}

best_score = 0
best_params = None

for C, lr, n_iters in product(param_grid['C'], param_grid['lr'], param_grid['n_iters']):
    svm = LinearSVM(C=C, lr=lr, n_iters=n_iters)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    acc = np.mean(y_pred == y_test)
    if acc > best_score:
        best_score = acc
        best_params = {'C': C, 'lr': lr, 'n_iters': n_iters}

print(f"\nBest Params: {best_params}")
print(f"Best Accuracy: {best_score:.4f}")

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

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_bin, y_pred_bin)
tn, fp, fn, tp = cm.ravel()
print(f"Confusion Matrix:\n{cm}")

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

def compute_roc_auc(y_true, y_scores):
    desc_idx = np.argsort(y_scores)[::-1]
    y_true = y_true[desc_idx]
    y_scores = y_scores[desc_idx]
    distinct = np.where(np.diff(y_scores))[0]
    thresholds = np.r_[distinct, len(y_scores) - 1]
    tps = np.cumsum(y_true)[thresholds]
    fps = 1 + thresholds - tps
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]
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
print("\nTop 10 Important Features:")
print(importances.head(10))

# ================================
# 9. SAVE MODEL ARTIFACTS
# ================================
model_artifacts = {
    'weights': final_svm.w.tolist(),
    'mean': mean_X.tolist(),
    'std': std_X.tolist(),
    'feature_names': ['bias'] + feature_names,
    'best_params': best_params
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
with open('model/svm_credit_model.json', 'w') as f:
    json.dump(model_artifacts, f)

# Add this in SECTION 9: SAVE MODEL ARTIFACTS


print("\nModel saved to 'model/svm_credit_model.json'")

# ================================
# 10. DEPLOYMENT: USER INPUT PREDICTION
# ================================
print("\n" + "="*60)
print("DEPLOYMENT: PREDICT ON NEW RECORD")
print("="*60)

def preprocess_input(user_data):
    input_df = pd.DataFrame([user_data])
    input_df.fillna(df.median(numeric_only=True), inplace=True)
    input_df = cap_outliers(input_df, numeric_cols)
    input_df[bill_cols] = input_df[bill_cols].clip(lower=0)
    input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)
    for col in feature_names[1:]:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded.reindex(columns=feature_names[1:], fill_value=0)
    X_input = input_encoded.values.astype(np.float64)
    X_input = (X_input - mean_X) / std_X
    X_input = np.c_[np.ones(1), X_input]
    return X_input

def predict_default(user_data):
    X_input = preprocess_input(user_data)
    score = np.dot(X_input, final_svm.w)[0]
    prediction = "YES" if score > 0 else "NO"
    risk_prob = 1 / (1 + np.exp(-score))
    return prediction, risk_prob, score

# === INTERACTIVE INPUT ===
print("\n" + "-"*60)
print("INTERACTIVE PREDICTION MODE")
print("-"*60)
print("Enter values one by one (or press Enter to use defaults):")

def get_input(prompt, default):
    val = input(f"{prompt} [{default}]: ").strip()
    try:
        return float(val) if val else default
    except:
        return default

interactive_input = {
    'LIMIT_BAL': get_input("Credit Limit", 20000),
    'SEX': int(get_input("Sex (1=male, 2=female)", 2)),
    'EDUCATION': int(get_input("Education (1=grad,2=uni,3=hs,4=other)", 2)),
    'MARRIAGE': int(get_input("Marriage (1=married,2=single,3=other)", 1)),
    'AGE': get_input("Age", 24),
    'PAY_0': int(get_input("PAY_0 (repay Sep)", 2)),
    'PAY_2': int(get_input("PAY_2 (repay Aug)", 2)),
    'PAY_3': int(get_input("PAY_3 (repay Jul)", -1)),
    'PAY_4': int(get_input("PAY_4 (repay Jun)", -1)),
    'PAY_5': int(get_input("PAY_5 (repay May)", -2)),
    'PAY_6': int(get_input("PAY_6 (repay Apr)", -2)),
    'BILL_AMT1': get_input("Bill Sep", 3913),
    'BILL_AMT2': get_input("Bill Aug", 3102),
    'BILL_AMT3': get_input("Bill Jul", 689),
    'BILL_AMT4': get_input("Bill Jun", 0),
    'BILL_AMT5': get_input("Bill May", 0),
    'BILL_AMT6': get_input("Bill Apr", 0),
    'PAY_AMT1': get_input("Pay Sep", 0),
    'PAY_AMT2': get_input("Pay Aug", 689),
    'PAY_AMT3': get_input("Pay Jul", 0),
    'PAY_AMT4': get_input("Pay Jun", 0),
    'PAY_AMT5': get_input("Pay May", 0),
    'PAY_AMT6': get_input("Pay Apr", 0)
}

pred_int, prob_int, score_int = predict_default(interactive_input)
print(f"\nFINAL PREDICTION:")
print(f"  -> Will Default: {pred_int}")
print(f"  -> Risk Probability: {prob_int:.1%}")
print(f"  -> Decision Score: {score_int:.4f}")

# ================================
# 11. FINAL SUMMARY & ALL PLOTS
# ================================
print("\n" + "="*60)
print("FINAL SUMMARY & VISUALIZATION")
print("="*60)

print(f"Best Hyperparameters: {best_params}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
print(f"ROC-AUC: {auc:.4f}")
print(f"Top Risk Factor: {importances.index[0]}")

# === PLOT 1: ROC CURVE ===
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', color='navy', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# === PLOT 2: CONFUSION MATRIX ===
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# === PLOT 3: FEATURE IMPORTANCE ===
plt.figure(figsize=(10, 8))
top_n = 15
importances.head(top_n).plot(kind='barh', color='teal')
plt.title(f'Top {top_n} Features Influencing Default Risk')
plt.xlabel('Absolute Weight (|w|)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# === PLOT 4: PREDICTION DISTRIBUTION ===
plt.figure(figsize=(10, 6))
plt.hist(y_scores[y_test == -1], bins=50, alpha=0.7, label='No Default', color='green')
plt.hist(y_scores[y_test == 1], bins=50, alpha=0.7, label='Default', color='red')
plt.axvline(0, color='black', linestyle='--', label='Decision Boundary')
plt.xlabel('SVM Decision Score')
plt.ylabel('Frequency')
plt.title('Distribution of SVM Decision Scores by Class')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# === PLOT 5: USER PREDICTION VISUAL ===
plt.figure(figsize=(8, 5))
risk_val = prob_int * 100
color = 'red' if pred_int == "YES" else 'green'
plt.barh(['Risk Probability'], [risk_val], color=color, alpha=0.8)
plt.xlim(0, 100)
plt.xlabel('Default Risk (%)')
plt.title(f'User Input Prediction: {"HIGH RISK" if pred_int == "YES" else "LOW RISK"}')
plt.text(risk_val + 2, 0, f'{risk_val:.1f}%', va='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("DEPLOYMENT & VISUALIZATION COMPLETE")
print("All plots displayed. Model ready for real-time use.")
print("Save this script and reuse for predictions anytime.")