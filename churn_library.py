"""
Code for churn prediction modelling
Owner: marcospiau
Date: February 3, 2022
"""

import logging
import shutil
import tempfile

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tabulate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import minmax_scale


def import_data(path: str) -> pd.DataFrame:
    """Read csv data into pandas.

    Args:
        path (str): a path to the csv

    Returns:
        pd.DataFrame: pandas dataframe.

    Raises:
        FileNotFoundError: if input path doest not exists
        AssertionError: if load data is empty
    """
    try:
        df = pd.read_csv(path)
        assert not df.empty
        logging.info('SUCCESS: Loaded csv from %s', path)
        return df
    except FileNotFoundError as err:
        logging.error(err)
        raise
    except AssertionError:
        logging.error('Data appears to be empty')
        raise


def perform_eda(df: pd.DataFrame,
                cat_columns: List[str],
                quant_columns: List[str],
                response: str = 'Churn',
                output_dir: str = './images/eda') -> None:
    """Perform eda on df and save figures to images folder.

    Args:
        df (pd.DataFrame): input dataframe
        cat_columns (List[str]): categoric columns
        quant_columns (List[str]): continuous features
        response (str): target column name. Defaults to 'Churn'.
        output_dir (str): Output directory. Defaults to ./images/eda.

    """
    # Initialize directory Tree
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ['target', 'categorical_features', 'quantitative_features']:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # # Target column
    df[response].value_counts(dropna=False).plot.bar(grid=True)
    plt.title(f'{response} distribution')
    plt.xlabel('Value')
    plt.ylabel('Counts')
    plt.tight_layout()
    plt.savefig(output_dir / 'target' / 'target_distribution.png')

    # # Categorical features - univariate
    for col in cat_columns:
        df[col].value_counts(dropna=False).plot.bar(grid=True)
        plt.title(f'Univariate distribution of feature {col}')
        plt.xlabel('Value')
        plt.ylabel('Counts')
        plt.tight_layout()
        plt.savefig(output_dir / 'categorical_features' /
                    f'univariate_distribution_{col}.png')
        plt.clf()
        plt.close()

    # Categorical features - mean response
    for col in cat_columns:
        df.groupby(col)[response].mean().sort_values().plot.bar(grid=True)
        plt.title(f'Mean response for {col}')
        plt.xlabel('Value')
        plt.ylabel('Counts')
        plt.tight_layout()
        plt.savefig(output_dir / 'categorical_features' /
                    f'mean_response_{col}.png')
        plt.clf()
        plt.close()

    # Quantitative features - histogram by target value
    for col in quant_columns:
        sns.displot(df,
                    x=col,
                    hue=response,
                    element='step',
                    bins=50,
                    stat='count',
                    common_bins=True,
                    common_norm=True,
                    legend=True)
        plt.title(f'Histogram by target value {col}')
        plt.tight_layout()
        plt.savefig(output_dir / 'quantitative_features' /
                    f'histogram_by_target_{col}.png')
        plt.clf()
        plt.close()

    # Quantitative features - univariate histogram
    for col in quant_columns:
        sns.displot(df,
                    x=col,
                    element='step',
                    bins=50,
                    stat='count',
                    common_bins=True,
                    common_norm=True,
                    legend=True)
        plt.title(f'Univariate histogram {col}')
        plt.tight_layout()
        plt.savefig(output_dir / 'quantitative_features' /
                    f'univariate_histogram_{col}.png')
        plt.clf()
        plt.close()

    # Quantitative columns - correlation
    plt.figure(figsize=(20, 10))
    sns.heatmap(df[quant_columns + [response]].corr().round(2),
                annot=True,
                linewidths=2)
    plt.title('Correlation matrix quantitative columns')
    plt.tight_layout()
    plt.savefig(output_dir / 'quantitative_features' /
                'correlation_matrix_quant_columns.png')
    plt.clf()
    plt.close()


def encoder_helper(df: pd.DataFrame,
                   category_lst: List[str],
                   response: str = 'Churn') -> pd.DataFrame:
    """Helper function to turn each categorical column into a new column with

    Args:
        df (pd.DataFrame): input data.
        category_lst (List[str]): list of columns that contain categorical
            features
        response (str): string of response name [optional argument that could
            be used for naming variables or index y column]. Defaults to
            'Churn'.

    Returns:
        pd.DataFrame: pd.pd.DataFrame with encoded categorical columns, with
        same index from input.
    """

    out = []
    for col in category_lst:
        out.append(
            df.groupby(col)[response].transform('mean').to_frame(
                f'{col}_mean_{response}').copy())
    out = pd.concat(out, axis=1).astype(np.float32)
    return out


def perform_feature_engineering(
    df: pd.DataFrame,
    cat_columns: List[str],
    quant_columns: List[str],
    response: str = 'Churn'
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Perform feature engineering given input data.

    Args:
        df (pd.DataFrame): dataframe with input data
        cat_columns (List[str]): columns to be treated as categorical
        quant_columns (List[str]): continouous columns
        response (str, optional): Target column name. Defaults to 'Churn'.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
            - X_train: X training data
            - X_test: X testing data
            - y_train: y training data
            - y_test: y testing data
    """
    # Continuous columns - normalize between 0 and 1
    logging.info('Starting feature engineering')
    X_num = df[quant_columns].copy()
    X_num[X_num.columns] = minmax_scale(X_num)
    # Encode categorical columns
    X_cat = encoder_helper(df=df, category_lst=cat_columns, response=response)
    # Concat numeric and categoric features
    X = pd.concat([X_num, X_cat], axis=1).astype(np.float32)

    y = df[response].astype(np.int64).copy()
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=42)
    logging.info('Finishing feature engineering')
    return X_train, X_test, y_train, y_test


def classification_report_image(labels_trues_preds: Dict[str, Tuple[np.array,
                                                                    np.array]],
                                output_file: str,
                                figsize=(7, 10),
                                **classification_report_kwargs) -> None:
    """Run classification reports for multiple pairs of y_true and y_pred and
        store as as image.


    Args:
        labels_trues_preds (Dict[str, Tuple[np.array, np.array]]): Dict mapping
            a string identifier to a tuple of arrays containing true and
            predicted target values.
        output_file (str): where to save predictions.
        figsize (tuple, optional): Figure size follows matplotlib format
            (width, height in inches). Defaults to (7, 10).
        **classification_report_kwargs: optional arguments passed to
            sklearn.metrics.classification_report function.

    """
    # Calculate classification report for each prediction
    results = {
        label: classification_report(y_true=y_true,
                                     y_pred=(y_pred >= 0.5).astype(int),
                                     **classification_report_kwargs)
        for label, (y_true, y_pred) in labels_trues_preds.items()
    }
    filler = 60 * '*'
    all_results = '\n'.join(f"{filler}\n{label}\n{filler}\n{result}"
                            for label, result in results.items())
    plt.figure(figsize=figsize)
    plt.text(0, 0, all_results, {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(output_file)
    plt.clf()
    plt.close()


def feature_importance_plot(feature_names: np.ndarray,
                            feature_importances: np.ndarray,
                            output_pth: str) -> None:
    """Save feature importances plot.

    Args:
        feature_names (np.ndarray): array of strings with feature names.
        feature_importances (np.ndarray): array containing feature importances.
        output_pth (str): where to save the plot.
    """
    df = pd.DataFrame()
    df['feature_names'] = feature_names.copy()
    df['feature_importances'] = feature_importances.copy()
    # Biggest absolute values on top
    df = df.iloc[df['feature_importances'].abs().argsort()[::-1]]
    plt.figure(figsize=(10, 10))
    sns.barplot(data=df, x='feature_importances', y='feature_names')
    plt.tight_layout()
    plt.savefig(output_pth)
    plt.clf()
    plt.close()


def train_models(X_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_train: pd.Series, y_test: pd.Series,
                 output_dir: str) -> None:
    """train, store model results: images + scores, and store models

    Args:
        X_train (pd.DataFrame): X training data
        X_test (pd.DataFrame): X testing data
        y_train (pd.Series): y training data
        y_test (pd.Series): y testing data
        output_dir (str): Where to save model and results files.
    """

    lrc = LogisticRegression(random_state=42, n_jobs=-1)
    logging.info('Starting logistic Regression Classifier fitting')
    lrc.fit(X_train, y_train)

    logging.info('Finished logistic Regression Classifier fitting')

    # grid search
    logging.info('Starting random Forest Grid Search')
    rfc = RandomForestClassifier(random_state=42, n_jobs=-1)
    rfc_param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = RandomizedSearchCV(
        estimator=rfc,
        param_distributions=rfc_param_grid,
        cv=5,
        # verbose=10,
        n_iter=2)
    cv_rfc.fit(X_train, y_train)
    logging.info('Finished Random Forest Grid Search')

    # save models
    output_dir = Path(output_dir)
    models_dir = Path(output_dir / 'models')
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(cv_rfc.best_estimator_, models_dir / 'rfc_model.pkl')
    joblib.dump(lrc, models_dir / 'logistic_model.pkl')

    # feature importances plots
    logging.info('Creating feature importance plots')
    results_dir = Path(output_dir / 'results')
    results_dir.mkdir(parents=True, exist_ok=True)
    feature_importance_plot(
        feature_names=np.r_[['Intercept'], lrc.feature_names_in_],
        feature_importances=np.r_[lrc.intercept_,
                                  lrc.coef_.ravel()],
        output_pth=results_dir / 'logistic_model_coefs.png')
    feature_importance_plot(
        feature_names=cv_rfc.best_estimator_.feature_names_in_,
        feature_importances=cv_rfc.best_estimator_.feature_importances_,
        output_pth=results_dir / 'rfc_model_feature_importances.png')

    # Data used for metrics evaluation
    logging.info('Predicting scores for train and test')
    preds_and_trues = {}
    preds_and_trues['Logistic Regression - train'] = (
        y_train, lrc.predict_proba(X_train)[:, 1])
    preds_and_trues['Logistic Regression - test'] = (
        y_test, lrc.predict_proba(X_test)[:, 1])
    preds_and_trues['Random Forest - train'] = (
        y_train, cv_rfc.best_estimator_.predict_proba(X_train)[:, 1])
    preds_and_trues['Random Forest - test'] = (
        y_test, cv_rfc.best_estimator_.predict_proba(X_test)[:, 1])

    logging.info('Creating AUC plots')
    _, ax = plt.subplots()
    for label, (y_true, y_preds) in preds_and_trues.items():
        RocCurveDisplay.from_predictions(y_true=y_true,
                                         y_pred=y_preds,
                                         name=label,
                                         ax=ax)
    plt.savefig(results_dir / 'auc_roc_curves.png')
    plt.clf()
    plt.close()
    del ax

    ## classification reports
    logging.info('Creating classification reports')
    classification_report_image(labels_trues_preds=preds_and_trues,
                                output_file=results_dir /
                                'classification_report_image.png')


def create_output_directory_tree(output_dir: str) -> None:
    """Creates directory tree for outputs.

    Args:
        output_dir (str): base directory for outputs.
    """
    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        logging.error('Output directory %s already exists', output_dir)
        raise

    for subdir in ['images', 'models', 'results', 'logs']:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def main():
    """Function that encapsulates main process."""
    logging.info('Starting script')

    # Create directory tree
    logging.info('Creating output directory tree')
    output_dir = Path('./outputs')
    create_output_directory_tree(output_dir)

    df = import_data('./data/bank_data.csv')
    df['Churn'] = np.where(df['Attrition_Flag'].eq('Attrited Customer'), 1, 0)

    # Define column roles
    cat_columns = [
        'Gender', 'Education_Level', 'Marital_Status', 'Income_Category',
        'Card_Category'
    ]

    quant_columns = [
        'Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'
    ]

    cols_and_types = [(col, 'cat') for col in cat_columns]
    cols_and_types += [(col, 'quant') for col in quant_columns]

    logging.info(
        'Features used:\n%s',
        tabulate.tabulate(cols_and_types,
                          headers=['Feature name', 'Feature type'],
                          tablefmt='pretty'))

    perform_eda(df=df,
                cat_columns=cat_columns,
                quant_columns=quant_columns,
                response='Churn',
                output_dir=output_dir / 'images' / 'eda')

    # feature engineering
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df=df,
        cat_columns=cat_columns,
        quant_columns=quant_columns,
        response='Churn')

    # Model fitting
    train_models(X_train=X_train,
                 X_test=X_test,
                 y_train=y_train,
                 y_test=y_test,
                 output_dir=output_dir)
    logging.info('PROCESS END')


if __name__ == '__main__':
    LOGGING_FORMAT = (
        '%(asctime)s %(levelname)s - %(filename)s - %(funcName)s - %(message)s'
    )
    temp_log_file = tempfile.mktemp()
    logging.basicConfig(filename=temp_log_file,
                        level=logging.INFO,
                        filemode='w',
                        format=LOGGING_FORMAT,
                        datefmt='%Y-%m-%d %H:%M:%S')
    main()
    shutil.copy(temp_log_file, './outputs/logs/results.log')
    Path(temp_log_file).unlink()
