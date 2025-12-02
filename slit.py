import streamlit as st
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

# ========================
# LOAD MODEL
# ========================
@st.cache_data
def load_model(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

# ========================
# RECOMPUTE METRICS & PLOTS ON-THE-FLY
# ========================
def recompute_metrics_and_plots(X_test, y_test, weights, feature_names):
    y_scores = np.dot(X_test, weights)
    y_pred = np.sign(y_scores)
    y_pred_bin = (y_pred == 1).astype(int)
    y_test_bin = (y_test == 1).astype(int)

    # Confusion Matrix
    cm = np.zeros((2, 2), dtype=int)
    for true, pred in zip(y_test_bin, y_pred_bin):
        cm[true][pred] += 1

    # ROC-AUC
    desc_idx = np.argsort(y_scores)[::-1]
    y_true = y_test_bin[desc_idx]
    tps = np.cumsum(y_true)
    fps = np.arange(1, len(y_true) + 1) - tps
    tpr = tps / tps[-1] if tps[-1] > 0 else tps
    fpr = fps / fps[-1] if fps[-1] > 0 else fps
    fpr = np.r_[0, fpr, 1]
    tpr = np.r_[0, tpr, 1]
    auc = np.trapz(tpr, fpr)

    # Importance
    importances = pd.Series(np.abs(weights), index=feature_names).sort_values(ascending=False)

    return y_scores, y_test, cm, fpr, tpr, auc, importances

# ========================
# PLOT FUNCTIONS
# ========================
def plot_roc(fpr, tpr, auc):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'ROC (AUC = {auc:.4f})', color='navy', linewidth=2)
    ax.plot([0,1], [0,1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

def plot_cm(cm):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Low Risk', 'High Risk'], yticklabels=['Low Risk', 'High Risk'], ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
    st.pyplot(fig)
    plt.close()

def plot_importance(importances):
    fig, ax = plt.subplots(figsize=(10, 7))
    top_n = 15
    importances.head(top_n).plot(kind='barh', color='teal', ax=ax)
    ax.set_title(f'Top {top_n} Risk Drivers')
    ax.set_xlabel('Absolute Weight')
    ax.invert_yaxis()
    st.pyplot(fig)
    plt.close()

def plot_score_dist(y_scores, y_test):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y_scores[y_test == -1], bins=50, alpha=0.7, label='Low Risk', color='green')
    ax.hist(y_scores[y_test == 1], bins=50, alpha=0.7, label='High Risk', color='red')
    ax.axvline(0, color='black', linestyle='--', label='Decision Boundary')
    ax.set_xlabel('Decision Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Decision Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

def plot_user_risk(prob, pred):
    fig, ax = plt.subplots(figsize=(8, 4))
    risk_pct = prob * 100
    color = 'red' if pred == "YES" else 'green'
    ax.barh(['Risk Level'], [risk_pct], color=color, alpha=0.8)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Default Risk (%)')
    ax.set_title(f'Customer Risk: {"HIGH" if pred == "YES" else "LOW"}')
    ax.text(risk_pct + 2, 0, f'{risk_pct:.1f}%', va='center', fontsize=12, fontweight='bold')
    st.pyplot(fig)
    plt.close()

# ========================
# PREDICTION FUNCTIONS
# ========================
def predict_default(user_data, artifacts):
    input_df = pd.DataFrame([user_data])
    input_df.fillna(input_df.median(numeric_only=True), inplace=True)

    def cap_outliers(s):
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return np.clip(s, q1 - 1.5*iqr, q3 + 1.5*iqr)
    for col in input_df.select_dtypes(include=[np.number]).columns:
        input_df[col] = cap_outliers(input_df[col])

    bill_cols = [f'BILL_AMT{i}' for i in range(1,7) if f'BILL_AMT{i}' in input_df.columns]
    input_df[bill_cols] = input_df[bill_cols].clip(lower=0)

    cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
    input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)

    for col in artifacts['feature_names'][1:]:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded.reindex(columns=artifacts['feature_names'][1:], fill_value=0)

    X = (input_encoded.values - np.array(artifacts['mean'])) / np.array(artifacts['std'])
    X = np.c_[np.ones(1), X]
    score = np.dot(X, np.array(artifacts['weights']))[0]
    prob = 1 / (1 + np.exp(-score))
    pred = "YES" if score > 0 else "NO"
    return pred, prob, score

def predict_approval(user_data, artifacts):
    input_df = pd.DataFrame([user_data])
    input_df['AGE'] = -input_df['DAYS_BIRTH'] / 365.25
    input_df['YEARS_EMPLOYED'] = np.where(input_df['DAYS_EMPLOYED'] > 0, 0, -input_df['DAYS_EMPLOYED'] / 365.25)
    input_df.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], axis=1, inplace=True)

    for col in input_df.columns:
        if input_df[col].isnull().sum() > 0:
            if col in artifacts['numeric_cols']:
                input_df[col].fillna(input_df[col].median(), inplace=True)
            else:
                input_df[col].fillna(input_df[col].mode()[0], inplace=True)

    def cap_outliers(s):
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return np.clip(s, q1 - 1.5*iqr, q3 + 1.5*iqr)
    for col in artifacts['numeric_cols']:
        if col in input_df.columns:
            input_df[col] = cap_outliers(input_df[col])

    input_encoded = pd.get_dummies(input_df, columns=artifacts['cat_cols'], drop_first=True)
    for col in artifacts['feature_names'][1:]:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded.reindex(columns=artifacts['feature_names'][1:], fill_value=0)

    X = (input_encoded.values - np.array(artifacts['mean'])) / np.array(artifacts['std'])
    X = np.c_[np.ones(1), X]
    score = np.dot(X, np.array(artifacts['weights']))[0]
    prob = 1 / (1 + np.exp(-score))
    pred = "YES" if score > 0 else "NO"
    return pred, prob, score

# ========================
# STREAMLIT APP
# ========================
st.title("Credit Prediction Interface")

default_model_path = '/Users/devk/Documents/MLops/model/svm_credit_model.json'
approval_model_path = '/Users/devk/Documents/MLops/model/credit_risk_svm.json'

default_model = load_model(default_model_path)
approval_model = load_model(approval_model_path)

tab1, tab2 = st.tabs(["Credit Default Prediction", "Credit Card Approval"])

with tab1:
    st.header("Credit Default Prediction")
    
    # Recompute metrics
    X_test = np.array(default_model['X_test'])
    y_test = np.array(default_model['y_test'])
    weights = np.array(default_model['weights'])
    feature_names = default_model['feature_names']
    y_scores, y_test, cm, fpr, tpr, auc, importances = recompute_metrics_and_plots(X_test, y_test, weights, feature_names)

    with st.form(key='default_form'):
        limit_bal = st.number_input("LIMIT_BAL", value=20000.0)
        sex = st.selectbox("SEX", options=[1, 2], index=1)
        education = st.selectbox("EDUCATION", options=[1, 2, 3, 4], index=1)
        marriage = st.selectbox("MARRIAGE", options=[1, 2, 3], index=0)
        age = st.number_input("AGE", value=24.0)
        pay_0 = st.number_input("PAY_0", value=2.0)
        pay_2 = st.number_input("PAY_2", value=2.0)
        pay_3 = st.number_input("PAY_3", value=-1.0)
        pay_4 = st.number_input("PAY_4", value=-1.0)
        pay_5 = st.number_input("PAY_5", value=-2.0)
        pay_6 = st.number_input("PAY_6", value=-2.0)
        bill_amt1 = st.number_input("BILL_AMT1", value=3913.0)
        bill_amt2 = st.number_input("BILL_AMT2", value=3102.0)
        bill_amt3 = st.number_input("BILL_AMT3", value=689.0)
        bill_amt4 = st.number_input("BILL_AMT4", value=0.0)
        bill_amt5 = st.number_input("BILL_AMT5", value=0.0)
        bill_amt6 = st.number_input("BILL_AMT6", value=0.0)
        pay_amt1 = st.number_input("PAY_AMT1", value=0.0)
        pay_amt2 = st.number_input("PAY_AMT2", value=689.0)
        pay_amt3 = st.number_input("PAY_AMT3", value=0.0)
        pay_amt4 = st.number_input("PAY_AMT4", value=0.0)
        pay_amt5 = st.number_input("PAY_AMT5", value=0.0)
        pay_amt6 = st.number_input("PAY_AMT6", value=0.0)
        
        submit_default = st.form_submit_button("Predict Default")
    
    if submit_default:
        default_input = {
            'LIMIT_BAL': limit_bal, 'SEX': sex, 'EDUCATION': education, 'MARRIAGE': marriage, 'AGE': age,
            'PAY_0': pay_0, 'PAY_2': pay_2, 'PAY_3': pay_3, 'PAY_4': pay_4, 'PAY_5': pay_5, 'PAY_6': pay_6,
            'BILL_AMT1': bill_amt1, 'BILL_AMT2': bill_amt2, 'BILL_AMT3': bill_amt3, 'BILL_AMT4': bill_amt4, 'BILL_AMT5': bill_amt5, 'BILL_AMT6': bill_amt6,
            'PAY_AMT1': pay_amt1, 'PAY_AMT2': pay_amt2, 'PAY_AMT3': pay_amt3, 'PAY_AMT4': pay_amt4, 'PAY_AMT5': pay_amt5, 'PAY_AMT6': pay_amt6
        }
        pred, prob, score = predict_default(default_input, default_model)
        st.write(f"Will Default: {pred}")
        st.write(f"Risk Probability: {prob:.1%}")
        st.write(f"Decision Score: {score:.4f}")

    st.header("Model Plots - Credit Default")
    plot_roc(fpr, tpr, auc)
    plot_cm(cm)
    plot_importance(importances)
    plot_score_dist(y_scores, y_test)
    if 'prob' in locals():
        plot_user_risk(prob, pred)

with tab2:
    st.header("Credit Card Approval")
    
    # Recompute metrics
    X_test = np.array(approval_model['X_test'])
    y_test = np.array(approval_model['y_test'])
    weights = np.array(approval_model['weights'])
    feature_names = approval_model['feature_names']
    y_scores, y_test, cm, fpr, tpr, auc, importances = recompute_metrics_and_plots(X_test, y_test, weights, feature_names)

    with st.form(key='approval_form'):
        code_gender = st.selectbox("CODE_GENDER", options=['M', 'F'], index=0)
        flag_own_car = st.selectbox("FLAG_OWN_CAR", options=['Y', 'N'], index=1)
        flag_own_realty = st.selectbox("FLAG_OWN_REALTY", options=['Y', 'N'], index=0)
        cnt_children = st.number_input("CNT_CHILDREN", value=0)
        amt_income_total = st.number_input("AMT_INCOME_TOTAL", value=180000.0)
        name_income_type = st.text_input("NAME_INCOME_TYPE", value="Working")
        name_education_type = st.text_input("NAME_EDUCATION_TYPE", value="Secondary / secondary special")
        name_family_status = st.text_input("NAME_FAMILY_STATUS", value="Married")
        name_housing_type = st.text_input("NAME_HOUSING_TYPE", value="House / apartment")
        days_birth = st.number_input("DAYS_BIRTH", value=-15000)
        days_employed = st.number_input("DAYS_EMPLOYED", value=-2000)
        flag_mobil = st.number_input("FLAG_MOBIL", value=1)
        flag_work_phone = st.number_input("FLAG_WORK_PHONE", value=0)
        flag_phone = st.number_input("FLAG_PHONE", value=0)
        flag_email = st.number_input("FLAG_EMAIL", value=0)
        occupation_type = st.text_input("OCCUPATION_TYPE", value="")
        cnt_fam_members = st.number_input("CNT_FAM_MEMBERS", value=2.0)
        
        submit_approval = st.form_submit_button("Predict Approval")
    
    if submit_approval:
        approval_input = {
            'CODE_GENDER': code_gender, 'FLAG_OWN_CAR': flag_own_car, 'FLAG_OWN_REALTY': flag_own_realty,
            'CNT_CHILDREN': cnt_children, 'AMT_INCOME_TOTAL': amt_income_total,
            'NAME_INCOME_TYPE': name_income_type, 'NAME_EDUCATION_TYPE': name_education_type,
            'NAME_FAMILY_STATUS': name_family_status, 'NAME_HOUSING_TYPE': name_housing_type,
            'DAYS_BIRTH': days_birth, 'DAYS_EMPLOYED': days_employed,
            'FLAG_MOBIL': flag_mobil, 'FLAG_WORK_PHONE': flag_work_phone, 'FLAG_PHONE': flag_phone, 'FLAG_EMAIL': flag_email,
            'OCCUPATION_TYPE': occupation_type, 'CNT_FAM_MEMBERS': cnt_fam_members
        }
        pred, prob, score = predict_approval(approval_input, approval_model)
        st.write(f"Approval: {pred}")
        st.write(f"Risk Probability: {prob:.1%}")
        st.write(f"Decision Score: {score:.4f}")

    st.header("Model Plots - Credit Card Approval")
    plot_roc(fpr, tpr, auc)
    plot_cm(cm)
    plot_importance(importances)
    plot_score_dist(y_scores, y_test)
    if 'prob' in locals():
        plot_user_risk(prob, pred)

# ========================
# FOOTER
# ========================
st.markdown("---")
st.caption("SVM from Scratch • Real-time Inference • Plots Recomputed On-Fly")