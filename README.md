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

# Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to understand transaction behavior and identify potential fraud indicators.

The analysis included:

- Distribution of transaction amounts
- Frequency of transaction types
- Balance changes before and after transactions
- Comparison between fraudulent and legitimate transactions

EDA helps identify anomalies and patterns that may signal suspicious financial activity.

Common visualizations included:

- Transaction amount distributions
- Fraud vs non-fraud comparisons
- Balance change patterns
- Feature correlation analysis

---

# Feature Engineering

Feature engineering was performed to capture abnormal balance movements that could indicate fraudulent behavior.

### Engineered Features

Balance Difference (Origin Account)

```r
balanceDiffOrig <- oldbalanceOrg - newbalanceOrig
```

Balance Difference (Destination Account)

```r
balanceDiffDest <- newbalanceDest - oldbalanceDest
```

These features measure inconsistencies between expected and actual balance changes during transactions.

---

# Machine Learning Model

A Logistic Regression classification model was implemented to predict whether a transaction is fraudulent.

Logistic regression is widely used for binary classification problems and provides interpretable results.

### Modeling Pipeline

The modeling process includes:

- Data preprocessing
- Feature scaling
- Encoding categorical variables
- Model training
- Model evaluation

The model classifies transactions into:

- Legitimate transactions
- Fraudulent transactions

---

# Model Evaluation

The model was evaluated using several classification metrics.

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Because fraud detection datasets are highly imbalanced, precision and recall provide more meaningful insights than accuracy alone.

---

# Confusion Matrix Interpretation

| Actual | Predicted Legitimate | Predicted Fraud |
|------|---------------------|----------------|
| Legitimate | True Negatives | False Positives |
| Fraud | False Negatives | True Positives |

Reducing false negatives is particularly important in fraud detection because missed fraudulent transactions can result in financial loss.

---

# Model Export

After training, the machine learning pipeline was saved so it can be reused without retraining the model.

```python
joblib.dump(pipeline, "fraud_detection_pipeline.pkl")
```

Saving the trained model allows it to be integrated into applications or deployed for real-time predictions.

---

# Streamlit Application

To make the model interactive, a Streamlit application was developed.

The Streamlit interface allows users to enter transaction details and receive fraud predictions from the trained model.

### Streamlit Features

- Interactive input fields for transaction data
- Real-time fraud prediction
- Probability score output
- Integration with the trained machine learning pipeline

### Running the Application

```bash
streamlit run app.py
```

After launching the application, users can test different transaction scenarios and see how the model classifies them.

---

# Project Structure

```
fraud-detection-project
│
├── Fraud Detection Model Using ML.ipynb
├── fraud_detection_pipeline.pkl
├── app.py
├── dataset.csv
└── README.Rmd
```

---

# Tools and Technologies

This project was developed using the following technologies:

- Python
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- Streamlit

These tools support data analysis, machine learning modeling, and application deployment.

---

