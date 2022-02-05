"""
Code for churn prediction modelling
Owner: marcospiau
Date: February 3, 2022
"""

import logging
import shutil
from sys import excepthook
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
import argparse
import yaml
import copy
from importlib import import_module
import sklearn


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


def load_yaml(path: str) -> Dict:
    with open(path, 'r') as handler:
        return yaml.load(handler, Loader=yaml.FullLoader)


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

    for subdir in [
            'images/eda/categorical_features',
            'images/eda/quantitative_features', 'images/eda/target', 'models',
            'results', 'logs'
    ]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


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
    output_dir.mkdir(parents=False, exist_ok=True)
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


def feature_importance_plot(model: sklearn.base.BaseEstimator,
                            output_pth: str,
                            feature_names: np.ndarray = None) -> None:
    """Save feature importances plot.

    Args:
        feature_names (np.ndarray): array of strings with feature names.
        feature_importances (np.ndarray): array containing feature importances.
        output_pth (str): where to save the plot.
    """
    try:
        assert feature_names is not None or hasattr(model, 'feature_names_in_')
        feature_names = feature_names or getattr(model, 'feature_names_in_')
        feature_names = list(feature_names)
    except ValueError:
        logging.error('feature names should be not None or model must have '
                      '`feature_names_in_` attribute')
        raise

    if hasattr(model, 'coef_'):
        feature_importances = model.intercept_.tolist() + model.coef_.ravel(
        ).tolist()
        feature_names.insert(0, '<INTERCEPT>')
    elif hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_

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


def load_model_cls(cls_name):
    first, second = cls_name.rsplit('.', 1)
    model_cls = getattr(import_module(first), second)
    return model_cls


def run_grid_search(X, y, **grid_search_kwargs) -> sklearn.base.BaseEstimator:
    grid_obj = RandomizedSearchCV(**run_grid_search)
    grid_obj.fit(X, y)
    return grid_obj.best_estimator_


def make_auc_plots(preds_and_trues: Dict[str, Tuple[np.array, np.array]],
                   output_file: str) -> None:
    _, ax = plt.subplots()
    for label, (y_true, y_preds) in preds_and_trues.items():
        RocCurveDisplay.from_predictions(y_true=y_true,
                                         y_pred=y_preds,
                                         name=label,
                                         ax=ax)
    plt.savefig(output_file)
    plt.clf()
    plt.close()
    del ax


def train_models(X_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_train: pd.Series, y_test: pd.Series, config: Dict) -> None:
    """train, store model results: images + scores, and store models

    Args:
        X_train (pd.DataFrame): X training data
        X_test (pd.DataFrame): X testing data
        y_train (pd.Series): y training data
        y_test (pd.Series): y testing data
        output_dir (str): Where to save model and results files.
    """

    output_dir = Path(config['output_dir'])
    models_dir = Path(output_dir / 'models')
    results_dir = Path(output_dir / 'results')
    # models_dir.mkdir(parents=True, exist_ok=True)

    # Train
    best_models = {}
    model_classes = {}
    # Run all grid searches and get best model trained
    logging.info('Starting grid searches')
    for model_name, model_config in config['models'].items():
        logging.info(f'Started {model_name} grid search')
        model_classes[model_name] = model_config['model_cls']
        best_models[model_name] = run_grid_search(
            X=X_train,
            y=y_train,
            **{
                **{
                    'estimator': model_classes[model_name]
                },
                **model_config['grid_params']
            })
        joblib.dump(best_models[model_name],
                    models_dir / f'best_{model_name}.pkl')
        feature_importance_plot(model=best_models[model_name],
                                output_pth=results_dir /
                                f'{model_name}_feature_importances.png')
        logging.info(f'Finished {model_name} grid search')

    logging.info(f'Finished grid searches')

    # Data used for metrics evaluation
    logging.info('Predicting scores for train and test')
    preds_and_trues = {}
    for model_name, model_obj in best_models.items():
        preds_and_trues[f'{model_name} - train'] = (
            model_obj.predict_proba(X_train), y_train)
        preds_and_trues[f'{model_name} - test'] = (
            model_obj.predict_proba(X_test), y_test)

    logging.info('Creating AUC plots')
    make_auc_plots(preds_and_trues, results_dir / 'auc_roc_curves.png')

    ## classification reports
    logging.info('Creating classification reports')
    classification_report_image(labels_trues_preds=preds_and_trues,
                                output_file=results_dir /
                                'classification_report_image.png')


def main(config):
    """Function that encapsulates main process."""
    logging.info('Starting script')

    # Create directory tree
    logging.info('Creating output directory tree')
    output_dir = Path(config['output_dir'])
    create_output_directory_tree(config['output_dir'])

    df = import_data(config['csv_path'])

    # FIXME: make this configurable
    df['Churn'] = np.where(df['Attrition_Flag'].eq('Attrited Customer'), 1, 0)

    # Define column roles
    cat_columns = config['categorical_features']
    quant_columns = config['numeric_features']

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
    parser = argparse.ArgumentParser(
        description='Run modeling for chun analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config',
                        required=True,
                        type=str,
                        help='yaml configuration file')
    args = parser.parse_args()

    LOGGING_FORMAT = (
        '%(asctime)s %(levelname)s - %(filename)s - %(funcName)s - %(message)s'
    )
    temp_log_file = tempfile.mktemp()
    logging.basicConfig(
        # filename=temp_log_file,
        level=logging.INFO,
        filemode='w',
        format=LOGGING_FORMAT,
        datefmt='%Y-%m-%d %H:%M:%S')

    config = load_yaml(args.config)
    main(config)
    # shutil.copy(temp_log_file, './outputs/logs/results.log')
    # Path(temp_log_file).unlink()
