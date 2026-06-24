# ML Projects

A collection of machine learning projects built during my AI/ML engineering learning journey.

## Projects

### S23 — XGBoost Wine Classification
- Dataset: Wine (sklearn built-in)
- Algorithm: XGBoost, Random Forest, Logistic Regression comparison
- Result: 94.98% cross-validation accuracy

### S24 — SVM Breast Cancer Classification
- Dataset: Breast Cancer (sklearn built-in)
- Algorithm: SVM with StandardScaler
- Result: 97.36% cross-validation accuracy

### S25 — Ridge & Lasso Regression
- Dataset: Restaurant (Zomato)
- Algorithms: Ridge, Lasso with alpha tuning
- Key learning: Lasso coefficient collapse at default alpha

### S26 — Medical Insurance Charges Prediction
- Dataset: insurance.csv (Kaggle)
- Algorithms: Random Forest, XGBoost, Linear Regression comparison
- Result: R2 = 0.8535 (Random Forest)

### S27 — Neural Network on Insurance Data
- Framework: TensorFlow/Keras
- Architecture: Dense(64) → Dense(32) → Dense(1)
- Result: R2 = 0.8757 with scaling + log transform

### S28 — Feature Engineering on Insurance Data
- Feature engineering: age_bmi, age_smoker interaction features
- Log transformation on target
- Result: R2 improved from 0.8535 → 0.8757

### S29 — Model Deployment with Flask
- Saved model with joblib
- Flask REST API serving predictions over HTTP
- Endpoint: POST /predict

## Tech Stack
Python, scikit-learn, XGBoost, TensorFlow/Keras, Flask, pandas, numpy

## Author
[raj-kapse](https://github.com/raj-kapse)
