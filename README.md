# Lloyds Bank Customer Churn Prediction

This project aims to predict customer churn for Lloyds Bank using machine learning. The goal is to help the bank proactively identify high-risk customers and enable early interventions to improve retention.

Key highlights:

Data sourced from customer demographics, transaction history, and support interactions.

Targeted feature construction and interaction terms based on domain-specific patterns.

Comparison of Logistic Regression, Random Forest, and XGBoost.

Final model selection using SMOTE, GridSearchCV, and business-driven metric evaluation.

Post-model analysis includes calibration, SHAP interpretation, and churn risk scoring.

## Project Overview

- **Problem**: Customer churn causes revenue loss. Predicting which customers are likely to churn allows timely business action.
- **Approach**: Built and evaluated multiple models including Logistic Regression, Random Forest, and XGBoost.
- **Outcome**: A tuned model with optimized threshold achieved strong recall and AUC, suitable for early churn detection.

## Technologies Used
- Python (Pandas, NumPy,Matplotlib,seaborn,scipy, Scikit-learn,LogisticRegression,Random Forest,XGBoost)
- SMOTE (Imbalanced-learn)
- GridSearchCV
- SHAP for model explainability
- Colab Notebook

## Project Workflow

1. **Data Preparation**  
   - Merged multiple customer data sources  
   - Cleaned and encoded features

2. **Exploratory Data Analysis (EDA)**  
   - Analyzed churn distribution, spending patterns, and customer demographics

3. **Feature Engineering**  
   - Created time-based features (e.g., DaysSinceLastLogin)  
   - Constructed interaction terms (e.g., HighSpent_LowLogin)

4. **Model Training & Tuning**  
   - Used SMOTE to balance the dataset  
   - Applied GridSearchCV on Random Forest & XGBoost  
   - Tuned classification threshold to improve recall

5. **Evaluation**  
   - Compared models on precision, recall, F1-score, and AUC  
   - Visualized confusion matrix, ROC curve

6. **Risk Scoring**  
   - Scored customers into Low, Medium, High churn risk groups  
   - Exported risk-labeled results

## Final Model Performance

| Metric      | Tuned XGBoost | Tuned Random Forest |
|-------------|---------------|---------------------|
| Recall      | 0.42          | 0.45                |
| Precision   | 0.52          | 0.48                |
| F1-score    | 0.47          | 0.46                |
| AUC         | 0.74          | **0.75**            |

Final model selected: **Random Forest** (better AUC and recall after threshold tuning)

## Project Structure
├── data/ # Input datasets (not included here)
├── notebook.ipynb # Main analysis and modeling notebook
├── outputs/ # Model results / visualizations
├── requirements.txt # Python dependencies
└── README.md # Project overview (this file)

## Key Takeaways

- Threshold tuning significantly improved model recall, which is essential in churn scenarios.
- Business-oriented evaluation (recall, F1-score) was prioritized over accuracy.
- Model insights using SHAP helped explain predictions and support decision-making.



