
# Credit Card Fraud Detection

This repository contains code for credit card fraud detection using a dataset of credit card transactions. The dataset includes information on transactions, with a target variable indicating whether the transaction is fraudulent or not. The code performs several data preprocessing, exploratory data analysis (EDA), feature scaling, and machine learning model building tasks to predict fraudulent transactions.

## Dependencies

This project requires the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scipy`
- `sklearn`
- `xgboost`

You can install these libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn xgboost
```

## Dataset

The dataset used is a CSV file named `creditcard.csv`. It contains the following columns:

- `Time`: The number of seconds elapsed between this transaction and the first transaction in the dataset.
- `Amount`: The transaction amount for each credit card transaction.
- `Class`: The target variable (0 for non-fraudulent transactions, 1 for fraudulent transactions).
- Other anonymized features (e.g., `V1`, `V2`, ... `V28`).

### Note:
The dataset is highly imbalanced, with far more non-fraudulent transactions than fraudulent ones.

## Code Overview

The following steps are performed in the code:

1. **Data Loading and Inspection**: 
   - Loads the dataset and inspects its shape, structure, and summary statistics.
   
2. **Exploratory Data Analysis (EDA)**: 
   - Visualizes the distribution of the `Time` and `Amount` features.
   - Analyzes the distribution of fraudulent vs. non-fraudulent transactions.
   - Displays a correlation heatmap for numerical features.

3. **Data Preprocessing**: 
   - Scales the `Time` and `Amount` columns using `StandardScaler`.
   - Splits the dataset into training and testing sets.
   - Creates a balanced dataset by randomly selecting non-fraudulent transactions to match the number of fraudulent transactions.

4. **Feature Visualization**:
   - Visualizes features with high positive and negative correlations with the target variable (`Class`).

5. **Outlier Removal**: 
   - Removes extreme outliers based on the Interquartile Range (IQR) method.

6. **Dimensionality Reduction**:
   - Applies t-SNE for dimensionality reduction and visualizes the two-dimensional representation of the data.

7. **Model Training and Evaluation**:
   - Trains and evaluates several classification models, including:
     - Logistic Regression (LR)
     - Linear Discriminant Analysis (LDA)
     - K-Nearest Neighbors (KNN)
     - Decision Tree Classifier (CART)
     - Support Vector Machine (SVM)
     - XGBoost (XGB)
     - Random Forest Classifier (RF)
   - Models are evaluated using cross-validation with ROC-AUC as the evaluation metric.

8. **Random Forest Visualization**:
   - Extracts and visualizes one decision tree from the trained Random Forest model.

## Results

The models' performance is evaluated using ROC-AUC, and the results are visualized in a boxplot to compare the classification algorithms.



