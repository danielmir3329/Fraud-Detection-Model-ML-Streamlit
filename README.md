# Fraud Detection Model (Machine Learning + Streamlit)

A machine learning project that detects potentially fraudulent financial transactions using a full workflow: **EDA → feature engineering → preprocessing pipeline → model training/evaluation → model export → Streamlit app for predictions**.

---

##  What this project does

- Explores a transaction dataset to understand fraud patterns (EDA + charts)
- Engineers balance-based anomaly features
- Trains a **Logistic Regression** classifier using a robust **scikit-learn Pipeline**
- Handles heavy class imbalance using `class_weight="balanced"`
- Evaluates model performance with a **classification report** + **confusion matrix**
- Exports a reusable model file: `fraud_detection_pipeline.pkl`
- Includes a **Streamlit app** (you wrote after the notebook) to run predictions interactively

---

##  Dataset

This notebook loads:
-  https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset?resource=download
- `AIML Dataset.csv`

Expected key columns used in the notebook include:
- `type`
- `amount`
- `oldbalanceOrg`, `newbalanceOrig`
- `oldbalanceDest`, `newbalanceDest`
- `nameOrig`, `nameDest`
- `step`
- `isFraud` (target)
- `isFlaggedFraud`

✅ **Tip:** Put your CSV in a `data/` folder and update the path in the notebook/app:
```python
df = pd.read_csv("data/AIML Dataset.csv")
