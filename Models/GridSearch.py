from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import json
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from collections import Counter

"""
    Perform grid search to find the best hyperparameters for classifiers.

    These functions are designed to automate the search for optimal hyperparameters
    for various classifiers using cross-validated grid search, saving both the best
    parameters and detailed logs of the CV results.

    Parameters
    ----------
    X : array-like or DataFrame
        Feature data for training.
    y : array-like
        Target labels corresponding to X.
    metric : str, default "roc_auc"
        Scoring metric for evaluating model performance during grid search.

    Returns
    -------
    None
        Saves best parameters to a JSON file and detailed CV results to a timestamped log.
    """
<<<<<<< HEAD

def XgboostSearch(X, y, metric="roc_auc"):
    """
    Performs hyperparameter search for an XGBoost classifier using GridSearchCV 
    and saves the best parameters and results to files.

    Parameters
    ----------
    X : array-like or pd.DataFrame
        Feature matrix for training.

    y : array-like
        Target vector (binary classification labels: 0 and 1).

    metric : str, optional (default="roc_auc")
        Scoring metric used for model evaluation during cross-validation.
        Must be a valid scoring name recognized by scikit-learn.

    Returns
    -------
    None
        This function performs training and saves the output, 
        but does not return a fitted model.

    Side Effects
    ------------
    - Saves the best hyperparameters as a JSON file:
        "savedModels/XGBClassifier_best_params.json"
    - Logs all tested parameter combinations and their CV scores to a timestamped .txt file.

    Notes
    -----
    - Automatically computes `scale_pos_weight` to handle class imbalance.
    - Uses a 3-fold cross-validation strategy.
    - The `n_jobs=2` parameter in GridSearchCV enables limited parallel processing.
    """
    
    def compute_scale_pos_weight(y):
        counter = Counter(y)
        neg, pos = counter[0], counter[1]
        return neg / pos
=======
def compute_scale_pos_weight(y):
        counter = Counter(y)
        neg, pos = counter[0], counter[1]
        return neg / pos
def XgboostSearch(X, y, metric="roc_auc"):
>>>>>>> ee51f77272fab1c97acd9dea7fcee1eab9341ed9
    
    spw = compute_scale_pos_weight(y)

    param_grid_xgb = {
        'n_estimators': [300, 500, 700],
        'learning_rate': [0.1, 0.01],
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.9],
        'colsample_bytree': [0.8],
        'gamma': [0, 0.2, 0.4],
        'reg_lambda': [0, 1],
        'scale_pos_weight': [spw],
    }

    model = XGBClassifier(random_state=42)
    grid_search_xgb = GridSearchCV(
        model,
        param_grid=param_grid_xgb,
        cv=3,
        scoring=metric,
        n_jobs=2
    )

    print("Xgboost start")
    grid_search_xgb.fit(X, y)

    best_params = grid_search_xgb.best_params_

    # Save best parameters
    with open("savedModels/XGBClassifier_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    # Save detailed log with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"savedModels/XGBClassifier_{timestamp}.txt"
    with open(log_path, "w") as f:
        f.write(f"Best Params: {best_params}\n")
        f.write(f"Best CV Score: {grid_search_xgb.best_score_:.4f}\n\n")
        for p, m, s in zip(grid_search_xgb.cv_results_['params'],
                           grid_search_xgb.cv_results_['mean_test_score'],
                           grid_search_xgb.cv_results_['std_test_score']):
            f.write(f"{str(p):80s} → {m:.4f} ± {s:.4f}\n")


def LightgbmSearch(X, y, metric="roc_auc"):
    """
    Performs hyperparameter search for a LightGBM classifier using GridSearchCV 
    and saves the best parameters and search results to files.

    Parameters
    ----------
    X : array-like or pd.DataFrame
        Feature matrix used for training.

    y : array-like
        Target vector (binary classification labels: 0 and 1).

    metric : str, optional (default="roc_auc")
        Scoring metric used during cross-validation.
        Must be a valid scikit-learn scoring metric.

    Returns
    -------
    None
        This function runs model training and saves results to disk, 
        but does not return a model.

    Side Effects
    ------------
    - Saves best hyperparameters to:
        "savedModels/LGBMClassifier_best_params.json"
    - Logs all parameter combinations and their CV scores to a timestamped file.

    Notes
    -----
    - Uses `is_unbalance=True` to account for class imbalance.
    - Applies 3-fold cross-validation with parallel execution (n_jobs=2).
    - Adjust the hyperparameter grid as needed for specific tasks or data.
    """
    
    param_grid_lgb = {
        'n_estimators': [300, 500, 700],
        'learning_rate': [0.01, 0.05],
        'max_depth': [3, 5, 7],
        'subsample': [0.9],
        'colsample_bytree': [0.8],
        'min_split_gain': [0.1, 0],
        'reg_lambda': [0, 1, 3],
        'is_unbalance': [True]
    }

    model = LGBMClassifier(random_state=42)
    grid_search_lgb = GridSearchCV(
        model,
        param_grid=param_grid_lgb,
        cv=3,
        scoring=metric,
        n_jobs=2
    )

    print("LGBM start")
    grid_search_lgb.fit(X, y)

    best_params = grid_search_lgb.best_params_

    # Save best parameters
    with open("savedModels/LGBMClassifier_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    # Save log with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"savedModels/LGBMClassifier_{timestamp}.txt"
    with open(log_path, "w") as f:
        f.write(f"Best Params: {best_params}\n")
        f.write(f"Best CV Score: {grid_search_lgb.best_score_:.4f}\n\n")
        for p, m, s in zip(grid_search_lgb.cv_results_['params'],
                           grid_search_lgb.cv_results_['mean_test_score'],
                           grid_search_lgb.cv_results_['std_test_score']):
            f.write(f"{str(p):80s} → {m:.4f} ± {s:.4f}\n")


def AdaSearch(X, y, metric='roc_auc'):
    """
    Performs hyperparameter tuning for an AdaBoost classifier using GridSearchCV
    and saves the best parameters and results to files.

    Parameters
    ----------
    X : array-like or pd.DataFrame
        Feature matrix for training.

    y : array-like
        Target vector (binary classification labels).

    metric : str, optional (default='roc_auc')
        Scoring metric used during cross-validation. Should be a valid 
        scikit-learn scoring string.

    Returns
    -------
    None
        The function trains the model, saves the best parameters to a JSON file,
        and logs all parameter combinations and their CV scores to a timestamped text file.

    Side Effects
    ------------
    - Saves best parameters to "savedModels/AdaBoost_best_params.json"
    - Saves detailed grid search results in a timestamped text file in "savedModels/"
    """
    param_grid_ada = {
        'n_estimators': [50, 100, 150], 
        'learning_rate': [1, 0.1, 0.5],        
        'estimator': [
            DecisionTreeClassifier(max_depth=4, random_state=42),
            DecisionTreeClassifier(max_depth=6, random_state=42),
            DecisionTreeClassifier(max_depth=7, random_state=42)
        ]
    }

    model = AdaBoostClassifier(random_state=42)
    grid_search_ada = GridSearchCV(model, param_grid_ada, cv=3, scoring=metric, n_jobs=2)
    
    print("Ada start")
    grid_search_ada.fit(X, y)
    
    best_params = grid_search_ada.best_params_

    # Save best parameters to JSON
    with open("savedModels/AdaBoost_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    # Save log with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"savedModels/AdaBoost_{timestamp}.txt"
    with open(log_path, "w") as f:
        f.write(f"Best Params: {best_params}\n")
        f.write(f"Best CV Score: {grid_search_ada.best_score_:.4f}\n\n")
        for p, m, s in zip(grid_search_ada.cv_results_['params'],
                           grid_search_ada.cv_results_['mean_test_score'],
                           grid_search_ada.cv_results_['std_test_score']):
            f.write(f"{str(p):80s} → {m:.4f} ± {s:.4f}\n")

            
def RFSearch(X, y, metric='roc_auc'):
    """
    Performs hyperparameter tuning for a Random Forest classifier using GridSearchCV
    and saves the best parameters and results to files.

    Parameters
    ----------
    X : array-like or pd.DataFrame
        Feature matrix used for training.

    y : array-like
        Target vector (binary or multiclass labels).

    metric : str, optional (default='roc_auc')
        Scoring metric for model evaluation during cross-validation.
        Should be a valid scoring string recognized by scikit-learn.

    Returns
    -------
    None
        The function fits the model, saves the best hyperparameters to a JSON file,
        and logs detailed CV results to a timestamped text file.

    Side Effects
    ------------
    - Saves best parameters in "savedModels/RandomForest_best_params.json"
    - Saves all tested hyperparameter combinations and scores in a timestamped log file.
    """
    param_grid_rf = {
        'n_estimators': [20, 50, 100, 500],
        'max_depth': [10, 30, None],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced']
    }

    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search_rf = GridSearchCV(model, param_grid_rf, cv=3, scoring=metric, verbose=2)
    
    print("RF start")
    grid_search_rf.fit(X, y)
    
    best_params = grid_search_rf.best_params_

    # Save best params to JSON file
    with open("savedModels/RandomForest_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    # Save detailed CV results log with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"savedModels/RandomForest_{timestamp}.txt"
    with open(log_path, "w") as f:
        f.write(f"Best Params: {best_params}\n")
        f.write(f"Best CV Score: {grid_search_rf.best_score_:.4f}\n\n")
        for p, m, s in zip(grid_search_rf.cv_results_['params'],
                           grid_search_rf.cv_results_['mean_test_score'],
                           grid_search_rf.cv_results_['std_test_score']):
            f.write(f"{str(p):80s} → {m:.4f} ± {s:.4f}\n")






def LogisticRegressionSearch(X, y, metric='roc_auc'):
    """
    Performs hyperparameter tuning for a Logistic Regression model using GridSearchCV
    and saves the best parameters and detailed results to files.

    Parameters
    ----------
    X : array-like or pd.DataFrame
        Feature matrix for training.

    y : array-like
        Target vector (binary or multiclass labels).

    metric : str, optional (default='roc_auc')
        Scoring metric for model evaluation during cross-validation.
        Must be a valid scoring metric recognized by scikit-learn.

    Returns
    -------
    None
        Fits the Logistic Regression model, saves the best hyperparameters to a JSON file,
        and logs the full grid search results in a timestamped text file.

    Side Effects
    ------------
    - Saves best parameters in "savedModels/LogisticRegression_best_params.json"
    - Saves CV results log to a timestamped file under "savedModels/"
    """
    param_grid_lr = {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear'],  
        'class_weight': ['balanced'],
        'max_iter': [500]
    }

    model = LogisticRegression(random_state=42)
    grid_search_lr = GridSearchCV(model, param_grid_lr, cv=3, scoring=metric, n_jobs=2)
    
    print("LR start")
    grid_search_lr.fit(X, y)
    
    best_params = grid_search_lr.best_params_

    # Save best parameters to JSON
    with open("savedModels/LogisticRegression_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    # Save detailed log with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"savedModels/LogisticRegression_{timestamp}.txt"
    with open(log_path, "w") as f:
        f.write(f"Best Params: {best_params}\n")
        f.write(f"Best CV Score: {grid_search_lr.best_score_:.4f}\n\n")
        for p, m, s in zip(grid_search_lr.cv_results_['params'],
                           grid_search_lr.cv_results_['mean_test_score'],
                           grid_search_lr.cv_results_['std_test_score']):
            f.write(f"{str(p):80s} → {m:.4f} ± {s:.4f}\n")


def DecisionTreeSearch(X, y, metric='roc_auc'):
    """
    Performs hyperparameter tuning for a Decision Tree classifier using GridSearchCV
    and saves the best parameters and detailed cross-validation results to files.

    Parameters
    ----------
    X : array-like or pd.DataFrame
        Feature matrix used for training.

    y : array-like
        Target vector (labels for classification).

    metric : str, optional (default='roc_auc')
        Scoring metric for model evaluation during cross-validation.
        Must be a valid scikit-learn scoring string.

    Returns
    -------
    None
        Fits the model, saves the best hyperparameters to a JSON file,
        and logs the grid search results to a timestamped text file.

    Side Effects
    ------------
    - Saves best parameters in "savedModels/DecisionTree_best_params.json"
    - Saves detailed CV results in a timestamped log file within "savedModels/"
    """
    param_grid_dt = {
        'max_depth': [4, 6, 8, None],
        'ccp_alpha': [0.0, 0.001, 0.01],
        'criterion': ['gini', 'entropy'],
        'class_weight': ['balanced']
    }

    model = DecisionTreeClassifier(random_state=42)
    grid_search_dt = GridSearchCV(model, param_grid_dt, cv=3, scoring=metric, n_jobs=2)
    
    print("DecisionTree start")
    grid_search_dt.fit(X, y)
    
    best_params = grid_search_dt.best_params_

    # Save best parameters to JSON file
    with open("savedModels/DecisionTree_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    # Save detailed CV results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"savedModels/DecisionTree_{timestamp}.txt"
    with open(log_path, "w") as f:
        f.write(f"Best Params: {best_params}\n")
        f.write(f"Best CV Score: {grid_search_dt.best_score_:.4f}\n\n")
        for p, m, s in zip(grid_search_dt.cv_results_['params'],
                           grid_search_dt.cv_results_['mean_test_score'],
                           grid_search_dt.cv_results_['std_test_score']):
            f.write(f"{str(p):80s} → {m:.4f} ± {s:.4f}\n")



