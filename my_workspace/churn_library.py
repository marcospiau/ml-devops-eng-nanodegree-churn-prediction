"""Code for chur prediction modelling"""

# import libraries
import logging
import shutil
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

logging.basicConfig(
    # filename='./test_results.log',
    level=logging.INFO,
    filemode='w',
    # format='%(name)s - %(levelname)s - %(message)s'
    format=
    '%(asctime)s %(levelname)s - %(filename)s - %(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')


def import_data(path: str) -> pd.DataFrame:
    """Read csv data into pandas.

    Args:
        path (str): a path to the csv

    Returns:
        pd.DataFrame: pandas dataframe.
    """
    try:
        df = pd.read_csv(path)
        logging.info('SUCCESS: Loaded csv from %s', path)
        return df
    except FileNotFoundError as err:
        logging.error(err)
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
        response (str): target name. Defaults to 'Churn'.
        output_dir (str): Output directory. Defaults to ./images/eda.
    """
    # Initialize directory Tree
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ['target', 'categorical_features', 'quantitative_features']:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # # Target column
    # df[response].value_counts(dropna=False).plot.bar(grid=True)
    # plt.title(f'{response} distribution')
    # plt.xlabel('Value')
    # plt.ylabel('Counts')
    # plt.tight_layout()
    # plt.savefig(output_dir / 'target' / 'target_distribution.png')

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
                    f'univariate_histogram{col}.png')
        plt.clf()


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
        pd.DataFrame: encoded categorical columns.
    """

    out = []
    for col in category_lst:
        out.append(
            df.groupby(col)[response].transform('mean').to_frame(
                f'{col}_{response}'))
    out = pd.concat(out, axis=1)
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

    Raises:
        NotImplementedError: [description]

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


def classification_report_image(y_train, y_test, y_train_preds_lr,
                                y_train_preds_rf, y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    raise NotImplementedError


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    raise NotImplementedError


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    raise NotImplementedError


if __name__ == '__main__':
    logging.info('Starting script')
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

    logging.info('Numeric columns (%d) \n%s\n', len(quant_columns),
                 '\n'.join(quant_columns))
    logging.info('Categoric columns (%d): \n%s\n', len(cat_columns),
                 '\n'.join(cat_columns))

    # TODO: maybe remove this rmtree
    shutil.rmtree('./images')
    perform_eda(df=df,
                cat_columns=cat_columns,
                quant_columns=quant_columns,
                response='Churn',
                output_dir='./images/eda')

    # feature engineering
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df=df,
        cat_columns=cat_columns,
        quant_columns=quant_columns,
        response='Churn')

    lrc = LogisticRegression(random_state=42, n_jobs=-1)

    logging.info('Fitting logistic Regression Classifier')
    lrc.fit(X_train, y_train)
