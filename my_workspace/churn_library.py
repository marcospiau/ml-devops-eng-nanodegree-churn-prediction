"""Code for chur prediction modelling"""

# import libraries
import logging
import shutil
from pathlib import Path
from typing import List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
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
        assert not(df.empty)
        logging.info('SUCCESS: Loaded csv from %s', path)
        return df
    except FileNotFoundError as err:
        logging.error(err)
        raise
    except AssertionError as err:
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
        response (str): target name. Defaults to 'Churn'.
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
    plt.title(f'Correlation matrix quantitative columns')
    plt.tight_layout()
    plt.savefig(output_dir / 'quantitative_features' /
                f'correlation_matrix_quant_columns.png')
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
        pd.DataFrame: encoded categorical columns.
    """

    out = []
    for col in category_lst:
        out.append(
            df
            .groupby(col)
            [response]
            .transform('mean')
            .to_frame(f'{col}_mean_{response}')
            .copy()
        )
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

def classification_report_image(
    labels_trues_preds: Dict[str, List[np.array, np.array]],
    output_file: str,
    **classification_report_kwargs) -> None:
    """Run classification reports for multiple pairs of y_true and y_pred and
        store as as image.
    """
    # Calculate classification report for each prediction
    results = {
        label: classification_report(
            y_true=y_true,
            y_pred=y_pred,
        )
        for label, (y_true, y_pred) in labels_and_preds.items()
    }
    all_results = {f'{label}\n\n {result}' for label, result in results.items()}
    plt.figure(figsize=(5, 5))
    plt.text(all_results)
    plt.tight_layout()
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
    df['feature_importances'] = feature_importances.ravel().copy()
    # Biggest absolute values on top
    df = df.iloc[df['feature_importances'].abs().argsort()[::-1]]
    plt.figure(figsize=(10, 10))
    sns.barplot(data=df, x='feature_importances', y='feature_names')
    plt.tight_layout()
    plt.savefig(output_pth)
    plt.clf()
    plt.close()


def train_models(X_train: pd.DataFrame,
                 X_test: pd.DataFrame,
                 y_train: pd.Series,
                 y_test: pd.Series,
                 models_dir='./models',
                 results_dir='./results') -> None:
    """train, store model results: images + scores, and store models

    Args:
        X_train (pd.DataFrame): X training data
        X_test (pd.DataFrame): X testing data
        y_train (pd.Series): y training data
        y_test (pd.Series): y testing data
        models_dir (str, optional): Where to save model artifacts.
            Defaults to './models'.
        results_dir (str, optional): Where to save feature importances and
            results plot. Defaults to './results'.
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
    cv_rfc = RandomizedSearchCV(estimator=rfc,
                                param_distributions=rfc_param_grid,
                                cv=5,
                                verbose=10,
                                n_iter=2)
    cv_rfc.fit(X_train, y_train)
    logging.info('Finished Random Forest Grid Search')

    # save models
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(cv_rfc.best_estimator_, models_dir / 'rfc_model.pkl')
    joblib.dump(lrc, models_dir / 'logistic_model.pkl')

    # feature importances plots
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    feature_importance_plot(feature_names=lrc.feature_names_in_,
                            feature_importances=lrc.coef_,
                            output_pth=results_dir /
                            'logistic_model_coefs.png')
    feature_importance_plot(
        feature_names=cv_rfc.best_estimator_.feature_names_in_,
        feature_importances=cv_rfc.best_estimator_.feature_importances_,
        output_pth=results_dir / 'rfc_model_feature_importances.png')

    # Data used for metrics evaluation
    preds_and_trues = {}
    preds_and_trues['lrc_train'] = [lrc.predict_proba(X_train), y_train]
    preds_and_trues['lrc_test'] = [lrc.predict_proba(X_test), y_test]
    preds_and_trues['rfc_train'] = [cv_rfc.best_estimator_.predict_proba(X_train), y_train]
    preds_and_trues['rfc_test'] = [cv_rfc.best_estimator_.predict_proba(X_test), y_test]


    # FIXME: arrumar roc curves
    # TODO: ver como plotar diversos no mesmo axis
    # roc curves
    # for part, X, y in [('train', X_train, y_train), ('test', X_test, y_test)]:
    plt.axis()
    for label, y_true, y_preds in preds_and_trues.items():
        auc_plot = RocCurveDisplay.from_predictions(lrc, X, y)
        auc_plot = RocCurveDisplay.from_estimator(cv_rfc.best_estimator_,
                                                  X,
                                                  y,
                                                  ax=auc_plot.ax_)
        plt.title(f'AUC curves {part}')
        plt.savefig(results_dir / f'auc_{part}.png')
        plt.clf()
        plt.close()
    
    # TODO: arrumar chamada da classification report
    # classification reports
    classification_report_image(
        labels_trues_preds={
            'Logistic Regression (train)',
        },
        output_file=results_dir / 'classification_report_image.png'
        **classification_report_kwargs)    
    classification_report_image(
        y_true=

    )


def main():
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
    shutil.rmtree('./models')
    shutil.rmtree('./results')

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

    # Model fitting
    train_models(X_train=X_train,
                 X_test=X_test,
                 y_train=y_train,
                 y_test=y_test,
                 models_dir='./models',
                 results_dir='./results')

if __name__ == '__main__':
    main()