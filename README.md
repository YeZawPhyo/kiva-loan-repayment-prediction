# Predicting Microfinance Loan Repayment Using Machine Learning on the KIVA Platform

MSc Data Analytics with Banking and Finance — Research Skills for Computing
Ye Zaw Phyo (35017940)

## About
This project uses machine learning to predict whether completed KIVA
microfinance loans are repaid, and to rank the features that drive
repayment. Five models are compared: Logistic Regression, Random Forest,
XGBoost, CatBoost, and SVD with Logistic Regression.

## Files
- ML_test_3.ipynb — data cleaning, modelling, and evaluation
- streamlit_app-1.py — interactive prototype
- kiva_dashboard_artifacts/ — saved trained models and results
- requirements.txt — libraries needed to run the app

## Key finding
Loan structure, geography, and timing predict repayment more strongly
than borrower profile. CatBoost performed best (F1 0.93, recall 0.95,
AUC 0.999).

## Data
Loan data is used under KIVA's Developer Terms of Service for
non-commercial academic use. The dataset itself is not included in this
repository for licensing and ethics reasons.

## Live prototype
https://r227nxf3l8vgbdda6r6pbb.streamlit.app/
