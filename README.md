# Health Insurance Cost Prediction

An end-to-end machine learning project that predicts health insurance claim amounts based on demographic and health-related features.

The project covers the complete machine learning workflow — from data cleaning and exploratory data analysis to preprocessing, model training, hyperparameter tuning, model comparison, and deployment using Streamlit.

## Technologies Used

Python

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

XGBoost

Joblib

Streamlit

Jupyter Notebook

Git & GitHub

## Live Demo

 **[Try the Live Application](https://health-insurance-cost-prediction-9skjsx7fahrnhjia8dghks.streamlit.app/)**

---

## Problem Statement

Insurance claim amounts can vary significantly depending on factors such as age, BMI, blood pressure, smoking status, diabetes, gender, and number of children.

The objective of this project is to build a regression model that can learn the relationship between these customer attributes and their insurance claim amount, and then predict the expected claim for a new customer.

### Business Objective

The model can be used as a predictive tool to estimate potential insurance claim amounts based on customer information.

This can help in:

- Estimating potential claim costs
- Understanding factors associated with higher claim amounts
- Supporting risk assessment
- Assisting insurance-related decision making

---

## Project Objective

Build and evaluate multiple regression models and select the model that provides the best predictive performance for insurance claim amounts.

The models evaluated in this project are:

- Linear Regression
- Polynomial Regression
- Random Forest Regressor
- XGBoost Regressor

Hyperparameter tuning is performed using `GridSearchCV` for the tree-based models.

---

## Dataset

The dataset contains **1,340 records** and **10 original columns**.

The available features include:

| Feature | Description |
|---|---|
| `Id` | Unique identifier for each customer |
| `age` | Age of the customer |
| `gender` | Gender of the customer |
| `bmi` | Body Mass Index |
| `bloodpressure` | Blood pressure |
| `diabetic` | Whether the customer is diabetic |
| `children` | Number of children |
| `smoker` | Whether the customer is a smoker |
| `region` | Customer's region |
| `claim` | Insurance claim amount — target variable |

The dataset initially contained a small number of missing values in `age` and `region`. Since only a small number of records were affected, rows containing missing values were removed, leaving **1,332 records** for modeling.

---

## Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the dataset and identify relationships between features and the target variable.

The analysis included:

- Dataset structure and data types
- Missing-value analysis
- Duplicate-value checking
- Statistical summary
- Distribution analysis
- Categorical feature analysis
- Correlation analysis
- Relationship between customer attributes and claim amount

### Example Insights

The analysis showed noticeable differences in claim amounts across customer groups, particularly with respect to smoking status.

This helped provide an understanding of which features may have stronger relationships with insurance claim amounts before model training.

---

## Data Preprocessing

The following preprocessing steps were performed:

1. Checked the dataset for missing values.
2. Checked for duplicate records.
3. Removed rows containing missing values.
4. Separated features and target variable.
5. Encoded categorical variables.
6. Split the dataset into training and testing sets.
7. Applied feature scaling to numerical features.

### Categorical Encoding

The categorical features were converted into numerical representations using `LabelEncoder`.

Encoded features include:

- `gender`
- `diabetic`
- `smoker`

The encoders were saved so that the same transformations could be applied to new inputs in the deployed application.

### Feature Scaling

Numerical features were scaled using `StandardScaler`.

The fitted scaler was saved using `joblib` and reused during inference so that user inputs in the Streamlit application undergo the same transformation used during model training.

---

## Machine Learning Models

### 1. Linear Regression

Linear Regression was used as a baseline model.

It provides a simple reference point against which more complex models can be compared.

### 2. Polynomial Regression

Polynomial Regression was evaluated to capture potential non-linear relationships between the features and insurance claim amount.

### 3. Random Forest Regressor

Random Forest was used to capture non-linear relationships and feature interactions through an ensemble of decision trees.

Hyperparameters were tuned using `GridSearchCV`.

### 4. XGBoost Regressor

XGBoost was used as the final candidate model because of its ability to model complex non-linear relationships and interactions between features.

Hyperparameter tuning was performed using `GridSearchCV` with 3-fold cross-validation.

---

## Hyperparameter Tuning

`GridSearchCV` was used to search through different hyperparameter combinations and select the best-performing configuration based on the R² scoring metric.

This was applied to the tree-based regression models.

The best estimator returned by the grid search was then evaluated on the held-out test set.

---

## Model Evaluation

The models were evaluated using:

### R² Score

Measures how much of the variation in the target variable is explained by the model.

Higher is better.

### MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted claim amounts.

Lower is better.

### RMSE — Root Mean Squared Error

Measures prediction error while giving greater weight to larger errors.

Lower is better.

---

## Model Comparison

The reported test-set results were:

| Model | R² ↑ | MAE ↓ | RMSE ↓ |
|---|---:|---:|---:|
| Linear Regression | 0.7318 | 4431.20 | 5737.52 |
| Random Forest | 0.8028 | 3770.16 | 4919.84 |
| **XGBoost** | **0.8186** | **3671.95** | **4717.94** |

### Final Model

**XGBoost Regressor** achieved the best R² score and the lowest MAE and RMSE among the evaluated models.

Therefore, XGBoost was selected as the final model for deployment.

> **R²: 0.8186**  
> **MAE: 3671.95**  
> **RMSE: 4717.94**

---

## Deployment

The trained model was deployed using **Streamlit**.

The deployed application allows a user to enter:

- Age
- BMI
- Number of children
- Blood pressure
- Gender
- Diabetes status
- Smoking status

The application then:

```text
User Input
    ↓
Categorical Encoding
    ↓
Feature Scaling
    ↓
Trained XGBoost Model
    ↓
Predicted Insurance Claim