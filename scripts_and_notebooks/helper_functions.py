### This Python script contains several helper functions used throughout the `model_and_vis.ipynb`

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import time

from lightgbm import LGBMClassifier, LGBMRegressor
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler

import warnings
import requests
from bs4 import BeautifulSoup
import shap


def convert_to_cat(data, cols=['league', 'pos', 'decade']):
    """
    - Function designed to convert specified columns to 'category' type
    - data: pandas DataFrame containing the data
    - cols: a list of column names to convert to 'category' type (columns must exist within the DataFrame)
    """
    # Check if data is a pandas DataFrame
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame")
    
    # Check if columns exist in the DataFrame
    if not all(col in data.columns for col in cols):
        raise ValueError("Columns missing from pandas DataFrame")
    
    data[cols] = data[cols].astype('category')
    return data

def model_and_tune(data, target_var, space, predictors, stratify_col, stop_rounds=100,
                   n_splits=5, test_size=0.3, random_state=1, max_evals=1000):
    """
    Function to run and tune a LightGBM binary classification model using different DataFrames or parameters.
    Returns a tuple of datasets, model objects, and best parameters.

    - data: pandas DataFrame
    - target_var: str, target variable to predict
    - space: Hyperopt parameter tuning space
    - predictors: list of features used to predict the target variable
    - stratify_col: str, variable used to stratify the sampling
    - stop_rounds: int, early stopping rounds for LightGBM cv
    - n_splits: int, number of folds in StratifiedKFold
    - test_size: float, test set proportion
    - random_state: int, seed for reproducibility
    - max_evals: int, number of evaluations for hypertuning
    """
    # Suppress specific warning messages
    warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm')
    warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame")
        
    if not isinstance(target_var, str):
        raise TypeError("`target_var` must be a string")
    
    if not isinstance(stratify_col, str):
        raise TypeError("`stratify_col` must be a string")
    
    start_time = time.time()
    
    X = data[predictors].copy() # predictors df
    y = data[target_var].astype('int64') # predicted column
    
    # determine the numeric and categorical features
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(include=['category', 'object']).columns.tolist()
    
    # Standardize numeric features
    scaler = StandardScaler()
    X[numeric_features] = scaler.fit_transform(X[numeric_features])
    
    if stratify_col is None: ## no stratified-sampling
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    else: ## stratified sampling
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=data.loc[X.index, stratify_col]
        )

    print(f"X_train is {X_train.shape[0]} rows by {X_train.shape[1]} columns \n")
    print(f"X_test is {X_test.shape[0]} rows by {X_test.shape[1]} columns \n")

    # Prepare the LightGBM dataset
    lgb_train = lgb.Dataset(X_train.copy(), label=y_train, 
                            categorical_feature=cat_features, free_raw_data=False)
    lgb_test = lgb.Dataset(X_test.copy(), label=y_test, reference=lgb_train, 
                           categorical_feature=cat_features, free_raw_data=False)

    # Calculate class weights for imbalance
    positive = len(data[data[target_var] == 1]) # positive classes
    negative = len(data[data[target_var] == 0]) # negative classes
    weight = negative / positive # scale_pos_weight

    print(f"Number of positive values = {positive} \n")
    print(f"Number of negative values = {negative} \n")

    # Define the objective function for optimization
    def objective(params):
        try:
            # Convert certain parameters to integers
            ## number of boosting rounds
            params['num_boost_round'] = int(params['num_boost_round'])
            params['num_leaves'] = int(params['num_leaves'])
            params['min_data_in_leaf'] = int(params['min_data_in_leaf'])
            ## Bagging frequency
            params['bagging_freq'] = int(params['bagging_freq'])

            params['scale_pos_weight'] = weight # weights for different classes
            params['force_row_wise'] = True
            params['feature_pre_filter'] = False
            params['objective'] = 'binary' # binary classification

            # Custom F1 score evaluation metric
            def f1_score_eval(y_pred, dataset):
                y_true = dataset.get_label()
                y_pred_labels = np.round(y_pred)  # Convert probabilities to 0 or 1
                return 'f1_score', f1_score(y_true, y_pred_labels), True

            # Define cross-validation strategy
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

            # Perform cross-validation and store metrics
            cv_results = lgb.cv(
                params, 
                train_set=lgb_train, 
                folds=cv, 
                feval=f1_score_eval, 
                eval_train_metric=True,  # Enable to track training metrics
		        stratified=True,
                callbacks=[lgb.early_stopping(stopping_rounds=stop_rounds)]
            )

            # Extract the best F1 score
            mean_f1 = cv_results['valid f1_score-mean'][-1]

            print(f"F1-Score: {mean_f1}")
            print("=" * 100)

            return {'loss': -mean_f1, 'status': STATUS_OK, 'cv_results': cv_results}

        except Exception as e:
            print(f"Exception: {e}")
            return {'loss': np.inf, 'status': STATUS_OK}

    # Plot the learning curves
    def plot_learning_curves(cv_results, metric):
    	## Extract the validation and training metrics means
        train_metric = cv_results[f'train {metric}-mean']
        valid_metric = cv_results[f'valid {metric}-mean']

        ## Plot both curves
        plt.plot(train_metric, label='Train')
        plt.plot(valid_metric, label='Valid')
        
        ## Label the plot
        plt.xlabel('Boosting Rounds')
        plt.ylabel(metric)
        plt.title(f"Learning Curve ({metric})")
        
        ## Add a Legend
        plt.legend()
        
        plt.show()

    # Optimize hyperparameters using Hyperopt
    trials = Trials()
    best = fmin(
        fn=objective, #use custom defined objective
        space=space,
        algo=tpe.suggest, # Tree-structured Parzen Estimator (TPE)
        max_evals=max_evals,
        trials=trials, # store the results
        rstate=np.random.default_rng(random_state),
        verbose=False
    )

    # Extract the best trial parameters and learning curves
    best_trial_vals = trials.best_trial['misc']['vals']
    best_params = {key: val[0].item() for key, val in best_trial_vals.items()}
    
    # Convert certain parameters to integers
    best_params['num_boost_round'] = int(best_params['num_boost_round'])
    best_params['num_leaves'] = int(best_params['num_leaves'])
    best_params['min_data_in_leaf'] = int(best_params['min_data_in_leaf'])
    best_params['bagging_freq'] = int(best_params['bagging_freq'])
    best_params['scale_pos_weight'] = weight

    best_params.update({
        'force_row_wise': True,
        'feature_pre_filter': False,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'random_state': int(1)
    })

    # Plot learning curves for F1-Score after training
    best_trial = trials.best_trial
    # extract the cv_results for plotting
    cv_results = best_trial['result']['cv_results']
    plot_learning_curves(cv_results, 'f1_score')

    end_time = time.time()
    print(f"\n Total time to Hypertune Model = {(end_time - start_time) / 60} minutes")

    return X, y, X_train, y_train, X_test, y_test, lgb_train, lgb_test, best_params


# Train the model
def eval_model(model_tuple, params=None):
    """
    - model_tuple: tuple containing the datasets, model objects, and best parameters
    - params: dictionary of parameters to use for training the model
    """
    # Unpack the tuple into respective variables
    X, y, X_train, y_train, X_test, y_test, lgb_train, lgb_test, best_params = model_tuple

    # Custom evaluation function for f1_score
    def f1_score_eval(y_pred, dataset):
        y_true = dataset.get_label()
        y_pred_labels = np.round(y_pred)  # Convert probabilities to 0 or 1
        return 'f1_score', f1_score(y_true, y_pred_labels), True

    # If params is not provided, use best_params from the tuple
    if params is None:
        params = best_params

    # Train the LightGBM model
    lgbm_cl = lgb.train(params=params,
                        train_set=lgb_train,
                        valid_sets=[lgb_test],
                        feval=f1_score_eval)

    # Return the trained model and the datasets
    return lgbm_cl, X_train, y_train, X_test, y_test, lgb_train

def conf_mat_model(eval_tuple, output_dict=True):
    """
    - eval_tuple: tuple containing the datasets, model objects, and best parameters
    - output_dict: boolean, whether to output the classification report as a dictionary
    Function is used to calculate the confusion matrix for the model evaluation
    """
    # unpack the tuple
    lgbm_cl, X_train, y_train, X_test, y_test, lgb_train = eval_tuple
    
    # Predict the probabilities on the test set
    y_pred_prob = lgbm_cl.predict(X_test)
    
    # Convert probabilities to 0 or 1
    y_pred_labels = np.round(y_pred_prob)
    
    if output_dict:
        # Generate the classification report
        class_report = classification_report(y_test, y_pred_labels, output_dict=True)
    else:
    	# Generate the classification report without the dictionary
    	class_report = classification_report(y_test, y_pred_labels, output_dict=False)
    
    # Display the classification report
    return class_report

def feat_imp_eval(eval_tuple):
    """
    - eval_tuple: tuple containing the datasets, model objects, and best parameters
    Function is used to extract the feature importance values for the model evaluation
    """
    # unpack the tuple
    lgbm_cl, X_train, y_train, X_test, y_test, lgb_train = eval_tuple

    # Extract feature importance
    importance = lgbm_cl.feature_importance(importance_type='gain')
    feature_names = lgb_train.feature_name
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    
    importance_df['imp_pct'] = (importance_df['Importance'] / importance_df['Importance'].sum()) * 100
    
    # Sort the DataFrame by importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    return importance_df

def make_preds(data, predictors, eval_tuple, target, cols_to_round=None, round_cols=False):
    """
    - data: pandas DataFrame containing the data
    - eval_tuple: tuple containing the datasets, model objects, and best parameters
    - target: str, target variable to predict
    - cols_to_round: list of columns to round (optional)
    - round_cols: boolean, whether to round the specified columns (optional)
    """
    lgbm_cl, X_train, y_train, X_test, y_test, lgb_train = eval_tuple

    X = data[predictors].copy()

    # If rounding is requested, apply it to the specified columns
    if round_cols:
        X[cols_to_round] = X[cols_to_round].round()
        X[cols_to_round] = X[cols_to_round].astype('category')
    else:
        X = X.copy()

    ## Separate the numeric and categorical features
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(include=['category', 'object']).columns.tolist()

    # initialize the StandardScaler
    scaler = StandardScaler()

    # Standardize the numeric features
    X[numeric_features] = scaler.fit_transform(X[numeric_features])

    # Predict the probabilities using the LightGBM model
    y_pred_prob = lgbm_cl.predict(X)
    
    # Add the predictions to the original dataframe
    data[target] = y_pred_prob

    return data

# Function to extract the player's position from their MLB page using CSS selectors
def get_player_position(mlbamid):
    """
    - mlbamid: int, MLBAM ID for the player
    """
    # Create the base MLB.com URL
    base_url = "https://www.mlb.com/player/"
    
    # Create the search url by combining the base_url and the mlbamid
    url = base_url + str(mlbamid)
    
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Use the CSS selector to find the position
    ## Found using the Inspect Element tool in the browser
    position_element = soup.select("#player-header > div > div.player-header--vitals > ul > li:nth-child(1)")
    
    # If the position is found, return the text for the position
    if position_element:
        return position_element[0].get_text().strip() # return the position string
    return None

def plot_shap_ranks(eval_tuple, feat_imp_df):
    """
    - eval_tuple: tuple containing the datasets, model objects, and best parameters
    - feat_imp_df: DataFrame containing the feature importance values
    """
    ## Suppress specific warning messages
    warnings.filterwarnings("ignore", message="LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray")

    # unpack the tuple
    lgbm_cl, X_train, y_train, X_test, y_test, lgb_train = eval_tuple
    
    start_time = time.time()
    
    # Initialize the SHAP explainer
    explainer = shap.TreeExplainer(lgbm_cl, feature_perturbation='tree_path_dependent')
    # Calculate the SHAP explainer object
    shap_exp = explainer(X_train)
    # Calculate SHAP values (for binary classification, focus on class 1 SHAP values)
    shap_vals = explainer.shap_values(X_train)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]  # Use SHAP values for the positive class in binary classification

    # Determine the number of features
    n_features = X_train.shape[1]
    
    # Generate the bar plot to display all features
    shap.plots.bar(shap_exp, max_display=n_features)
    
    end_time = time.time()
    total = (end_time - start_time) / 60
    print(f"Total Time to Generate Plot: {total} minutes")

    # Convert SHAP values to a DataFrame
    shap_values_df = pd.DataFrame(shap_vals, columns=X_train.columns)
    
    # Calculate mean absolute SHAP value for each feature
    shap_summary = shap_values_df.abs().mean().sort_values(ascending=False)
    
    # Convert to DataFrame and add a ranking column
    shap_rankings = pd.DataFrame({
        'Feature': shap_summary.index,
        'mean_shap_value': shap_summary.values
    }).reset_index(drop=True)
    
    # Display the top-ranked features
    shap_rankings = pd.merge(shap_rankings, feat_imp_df,
            on='Feature', how='left')
    # Calculate the importance rank for each feature
    shap_rankings['imp_Rank'] = shap_rankings['imp_pct'].rank(ascending=False, method='dense')
    # Calculate the SHAP rank for each feature
    shap_rankings['shap_Rank'] = shap_rankings['mean_shap_value'].rank(ascending=False, method='dense')
    # Calculate the average rank for each feature
    ## I favored SHAP because I'm concerned about wideset applicability
    shap_rankings['avg_Rank'] = (0.45 * shap_rankings['imp_Rank']) + (0.55 * shap_rankings['shap_Rank'])
    
    # Sort the DataFrame by Rank (optional, if you want to see the ranked order)
    shap_rankings = shap_rankings.sort_values(by='avg_Rank').reset_index(drop=True)
    
    # Display the DataFrame with the ranks
    return shap_rankings

def update_mod_eval(model_tuple, best_predictors, stratify_col, target_var, data,
                    test_size=0.3, random_state=1):
    """
    - model_tuple: tuple containing the datasets, model objects, and best parameters
    - best_predictors: list of features to use for training the model
    - stratify_col: str, variable used to stratify the sampling
    - target_var: str, target variable to predict
    - data: pandas DataFrame containing the data
    - test_size: float, test set proportion
    - random_state: int, seed for reproducibility
    """
    # Unpack the tuple into respective variables
    X, y, X_train, y_train, X_test, y_test, lgb_train, lgb_test, best_params = model_tuple
    
    X = data[best_predictors].copy()
    y = data[target_var].astype('int64') # predicted column

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(include=['category', 'object']).columns.tolist()
    
    # Standardize numeric features
    scaler = StandardScaler()
    X[numeric_features] = scaler.fit_transform(X[numeric_features])

    if stratify_col is None:
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    else:
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=data.loc[X.index, stratify_col]
        )

    # Define preprocessing steps
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X_train.select_dtypes(include=['category', 'object']).columns.tolist()

    # Prepare the LightGBM dataset
    lgb_train = lgb.Dataset(X_train.copy(), label=y_train, 
                            categorical_feature=cat_features, free_raw_data=False)
    lgb_test = lgb.Dataset(X_test.copy(), label=y_test, reference=lgb_train, 
                           categorical_feature=cat_features, free_raw_data=False)

# Custom evaluation function for f1_score
    def f1_score_eval(y_pred, dataset):
        y_true = dataset.get_label()
        y_pred_labels = np.round(y_pred)  # Convert probabilities to 0 or 1
        return 'f1_score', f1_score(y_true, y_pred_labels), True

    # Train the LightGBM model
    lgbm_cl = lgb.train(params=best_params,
                        train_set=lgb_train,
                        valid_sets=[lgb_test],
                        feval=f1_score_eval)

    # Return the trained model and the datasets
    return lgbm_cl, X_train, y_train, X_test, y_test, lgb_train

def round_vote_getters(vote_getters, data):
    """
    - vote_getters: DataFrame containing the vote getters
    - data: DataFrame containing the data
    """
    # merge the vote getters with the data
    data = data.merge(vote_getters,
                     on=['Name', 'Team', 'league', 'pos'],
                     how='left')
    # Fill missing values in 'vote_getter' with 0
    data.loc[data['vote_getter_y'].isna(), 'vote_getter_x'] = 0
    # drop the 'vote_getter_y' column
    data.drop(columns='vote_getter_y', inplace=True)
    # rename the 'vote_getter_x' column to 'vote_getter'
    data.rename(columns={'vote_getter_x':'vote_getter'}, inplace=True)
    # convert the 'vote_getter' column to integer
    data['vote_getter'] = data['vote_getter'].round().astype('int64')
    df_copy = data.copy()

    return df_copy

def replace_league(data):
    """
    - data: DataFrame containing the data
    """
    # Brewers were in the AL until 1997
    data.loc[(data['Team'] == 'MIL') & (data['Season'] < 1998), 'league'] = 'AL'
    # Astros were in the NL until 2012
    data.loc[(data['Team'] == 'HOU') & (data['Season'] < 2013), 'league'] = 'NL'

    return data

def add_roy_column(data, roy):
    """
    - data: DataFrame containing the data
    - roy: DataFrame containing the Rookie of the Year winners
    """
    # Rename columns in `roy` to match `batters`
    roy_renamed = roy.rename(columns={'yearID': 'Season', 
                                      'lgID': 'league', 
                                      'playerID': 'key_bbref'}
                            )

    data.drop(columns='rookie_of_the_year', inplace=True)
    
    # Perform a left merge to add the 'rookie_of_the_year'
    merged_df = data.merge(
        roy_renamed[['Season', 'league', 'key_bbref', 'rookie_of_the_year']],  # Select the needed columns from `roy`
        on=['Season', 'league', 'key_bbref'],  # Merge on the matching columns
        how='left',  # Left merge to keep all rows from `batters`,
        indicator=True
    )

    # Fill missing values in 'rookie_of_the_year' with 0
    merged_df['rookie_of_the_year'] = merged_df['rookie_of_the_year'].fillna(0)

    return merged_df

def add_interactions(col1, col2, data):
    """
    - col1: str, first column to interact
    - col2: str, second column to interact
    - data: DataFrame containing the data
    - Used for creating interaction variables between numeric variables
    """
    data[col1 + '_int_' + col2] = data[col1] * data[col2]
    
    return data

def evaluate_predictors(rank_df, model_tuple, data, stratify_col, target_var, 
                        conf_mat_func, conf_mat_cat = 'macro avg',
                        metric='f1-score', min_preds=1, max_preds=20):
    """
    - rank_df: DataFrame containing the feature rankings
    - model_tuple: tuple containing the datasets, model objects, and best parameters
    - data: DataFrame containing the data
    - stratify_col: str, variable used to stratify the sampling
    - target_var: str, target variable to predict
    - conf_mat_func: function to calculate the confusion matrix
    - conf_mat_cat: str, category to extract from the confusion matrix
    - metric: str, metric to extract from the confusion matrix
    - min_preds: int, minimum number of predictors to remove
    - max_preds: int, maximum number of predictors to remove
    """
    best_f1 = 0
    best_num_preds = 0
    best_conf_mat = None

    for num_preds in range(min_preds, max_preds + 1):
        # Select the top predictors
        top_predictors = rank_df['Feature'][:-num_preds]
        
        # Evaluate the model
        eval_results = update_mod_eval(model_tuple=model_tuple, best_predictors=top_predictors,
                                       data=data, stratify_col=stratify_col, target_var=target_var)
        
        # Get confusion matrix
        conf_mat = conf_mat_func(eval_tuple=eval_results)
        
        # Extract weighted avg recall from confusion matrix
        weighted_avg_f1 = conf_mat[conf_mat_cat][metric]
        
        # Check if this is the best recall so far
        if weighted_avg_f1 > best_f1:
            best_f1 = weighted_avg_f1
            best_num_preds = num_preds
            best_conf_mat = conf_mat

    # Return the best number of predictors and the corresponding confusion matrix and recall
    return {
        'num_preds_to_remove': best_num_preds,
        'best_f1': best_f1,
        'best_conf_mat': best_conf_mat
    }
