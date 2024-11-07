
# Smoking Prediction Model

This project aims to predict whether an individual smokes based on various features, using multiple machine learning models. The notebook performs data preprocessing, exploratory data analysis, model training, evaluation, and model selection to find the most effective classifier for this task.

## Project Structure

- **Data Loading**: Loads data from a CSV file (`train.csv`). Ensure that this file has a `smoking` column, which serves as the target variable.
- **Preprocessing**: Removes irrelevant columns and handles missing values if necessary.
- **Exploratory Data Analysis (EDA)**: Analyzes the correlation of features with the target variable to identify relevant features.
- **Model Training and Comparison**:
  - Splits data using StratifiedKFold cross-validation to ensure balanced classes in each fold.
  - Trains various classifiers: **Random Forest**, **Logistic Regression**, **Decision Tree**.
  - Tunes hyperparameters with **Grid Search** to optimize each model's performance.
  - Evaluates models using key metrics like **accuracy** and **ROC AUC score**.
- **Model Selection**: Compares models to select the best one based on cross-validated performance.

## Requirements

- **Python 3.7+**
- Libraries:
  - `pandas`
  - `seaborn`
  - `matplotlib`
  - `scikit-learn`
  - `mlxtend`

Install the required packages with:

```bash
pip install pandas seaborn matplotlib scikit-learn mlxtend
```

## Usage

1. **Clone the repository** and ensure that `train.csv` is in the same directory as the notebook.
2. **Run the notebook cells** in sequence to perform data preprocessing, model training, hyperparameter tuning, and evaluation.

## Workflow

1. **Define Metrics**: Choose evaluation metrics like accuracy and ROC AUC to assess model performance.
2. **Train Models**: Use Logistic Regression, Decision Tree, and Random Forest, testing different hyperparameter values.
3. **Cross-Validate Models**: Use StratifiedKFold to ensure balanced training and validation splits.
4. **Hyperparameter Tuning**: Use Grid Search for fine-tuning each model’s parameters.
5. **Compare Models**: Analyze cross-validated results to select the model that performs best on the chosen metrics.

## Example Code Snippet

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define parameter grid for RandomForest
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, None],
    'max_features': ['auto', 'sqrt'],
    'min_samples_split': [2, 5, 10]
}

rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")
```

## Results

The models are evaluated using **accuracy** and **ROC AUC**. Feature correlation analysis offers insights into which factors influence smoking likelihood. Based on the final scores, the best-performing model is selected for further analysis or deployment.

## Notes

This project demonstrates how to compare multiple classifiers systematically and use cross-validation and hyperparameter tuning for model selection. It provides insights into feature importance and model selection techniques for binary classification tasks.
