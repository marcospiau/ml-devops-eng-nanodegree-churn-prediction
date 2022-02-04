# Predict Customer Churn
<<<<<<< HEAD
This repository contain my submission for the project **Predict Customer Churn**, final project for lesson $COLOCAR LESSONS$, first course of [ML DevOps Engineer Nanodegree Udacity](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821).

This project used machine learning models to predict which customers are more likely to cancel their account, using good pratices of coding blabla.
=======
This repository contain my submission for the project **Predict Customer Churn**, project for lesson **2. Clean Code Principles**, included in [ML DevOps Engineer Nanodegree Udacity](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821).

This project used machine learning models to predict which customers are more likely to cancel their account, using clean code principles.
>>>>>>> develop


## How to setup the environment
We use miniconda to manage Python packages installation.

Install minconda (skip if Anaconda or Miniconda are installed):
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

The following command creates a conda env named `uda_1`.
```bash
conda env create -f environment.yml python=3.8 --name uda_1
```

Activate the environement:
```bash
conda activate uda_1
```
## Running Files
How do you run your files? What should happen when you run your files?
## Main script
The main script (`churn_library.py`) performs the entire modeling pipeline:
 - load data
 - perform EDA and save plots
 - performe feature engineering, treating quantitative and categorical features accordingly
 - fits a logistic regression model and perform a grid search over a random forest model, and saves objects that can be used for making predictions on new data
 - saves results for each model

The script can be executed with:
```bash
python3 churn_library.py
```

Below is expected directory tree, with description of each file.
```bash
outputs/
├── images # image outputs
│   └── eda # eda images outputs
│       ├── categorical_features # plots for categorical features:
│       │   ├── mean_response_{feat_name}.png: mean response by category leval, one for each feature
│       │   ├── mean_response_{feat_name}.png: mean response by category leval, one for each feature
│       │   └── univariate_distribution_{feat_name}.png: example counts by category level, one for each feature
│       ├── quantitative_features # plots for quantitative features
│       │   ├── correlation_matrix_quant_columns.png: correlation matrix of all quantitative features (including target column)
│       │   ├── histogram_by_target_{feat_name}.png: histogram by target level
│       │   └── univariate_histogram_{feat_name}.png
│       └── target # plots for target column
│           └── target_distribution.png: target counts by level
├── logs # process logs
    └── results.log
├── models # serialized pickle models
│   ├── logistic_model.pkl # serialized logistic regression model
│   └── rfc_model.pkl # serialized random forest classifier
└── results # Results plots
    ├── auc_roc_curves.png # AUC plot, for each model and data partition (train, test)
    ├── classification_report_image.png # classification report, for each model and data partition (train, test)
    ├── logistic_model_coefs.png # coefficients for logistic regression model
    └── rfc_model_feature_importances.png # feature importances for random forest classifier model
```

## Tests
For tests, we use [`unittest`](https://docs.python.org/3/library/unittest.html) for writing tests and [`pytest`](https://docs.pytest.org/) for running them. Below are code to run tests from commmand line and expected results (passed all tests):
```bash
python3 -m pytest -vv churn_script_logging_and_tests.py

=========================================================================================== test session starts ============================================================================================
platform linux -- Python 3.8.12, pytest-6.2.5, py-1.11.0, pluggy-1.0.0 -- /home/marcospiau/miniconda3/envs/uda_1/bin/python3
cachedir: .pytest_cache
rootdir: XXXX (SECRET)
plugins: xdist-2.5.0, forked-1.4.0
collected 23 items

churn_script_logging_and_tests.py::TestImportData::test_empty_data_error PASSED                                                                                                                      [  4%]
churn_script_logging_and_tests.py::TestImportData::test_inexistent_file_error PASSED                                                                                                                 [  8%]
churn_script_logging_and_tests.py::TestImportData::test_ok_file PASSED                                                                                                                               [ 13%]
churn_script_logging_and_tests.py::TestCreateDirectoryTree::test_correct_creation PASSED                                                                                                             [ 17%]
churn_script_logging_and_tests.py::TestCreateDirectoryTree::test_error_if_exists PASSED                                                                                                              [ 21%]
churn_script_logging_and_tests.py::TestEda::test_categorical_features_plots_created PASSED                                                                                                           [ 26%]
churn_script_logging_and_tests.py::TestEda::test_directory_tree_correct PASSED                                                                                                                       [ 30%]
churn_script_logging_and_tests.py::TestEda::test_quant_features_plots_created PASSED                                                                                                                 [ 34%]
churn_script_logging_and_tests.py::TestEda::test_target_plots PASSED                                                                                                                                 [ 39%]
churn_script_logging_and_tests.py::TestCategoricalEncoder::test_correct_dtype PASSED                                                                                                                 [ 43%]
churn_script_logging_and_tests.py::TestCategoricalEncoder::test_correct_feature_names PASSED                                                                                                         [ 47%]
churn_script_logging_and_tests.py::TestCategoricalEncoder::test_correct_shape PASSED                                                                                                                 [ 52%]
churn_script_logging_and_tests.py::TestCategoricalEncoder::test_index_not_modified PASSED                                                                                                            [ 56%]
churn_script_logging_and_tests.py::TestFeatureEngineering::test_consistent_length_x_y_test PASSED                                                                                                    [ 60%]
churn_script_logging_and_tests.py::TestFeatureEngineering::test_consistent_length_x_y_train PASSED                                                                                                   [ 65%]
churn_script_logging_and_tests.py::TestFeatureEngineering::test_correct_feature_names PASSED                                                                                                         [ 69%]
churn_script_logging_and_tests.py::TestFeatureEngineering::test_x_dtype PASSED                                                                                                                       [ 73%]
churn_script_logging_and_tests.py::TestFeatureEngineering::test_y_dtype PASSED                                                                                                                       [ 78%]
churn_script_logging_and_tests.py::TestClassificationReportPlots::test_kwargs_plot PASSED                                                                                                            [ 82%]
churn_script_logging_and_tests.py::TestClassificationReportPlots::test_simple_plot PASSED                                                                                                            [ 86%]
churn_script_logging_and_tests.py::TestFeatureImportancePlots::test_expected_files_in_dir PASSED                                                                                                     [ 91%]
churn_script_logging_and_tests.py::TestTrain::test_can_open_serialized_models PASSED                                                                                                                 [ 95%]
churn_script_logging_and_tests.py::TestTrain::test_serialized_models_are_saved PASSED                                                                                                                [100%]

=========================================================================================== 23 passed in 17.38s ============================================================================================
```

## Linting
[`pylint`](https://pylint.org/) is used for code linting. Pylint configuration file `.pylintrc` is modified to supress warning about variables commonly used in data science (ie: df, X_train, etc). Below is the command and expected output using `pylint`:
```bash
python3 -m pylint *.py
************* Module churn_library
churn_library.py:281:0: R0914: Too many local variables (17/15) (too-many-locals)

------------------------------------------------------------------
Your code has been rated at 9.97/10 (previous run: 9.97/10, +0.00)
```
Besides the modifications in `.pylintrc`, we also supress `attribute-defined-outside-init` on a specific portion of code (class `MockDataTestCase`), in order to make data mocking possible on tests; improve testing logic in order to not need this workaround is future work.

## Code formatting
For code formatting, a mix of [`yapf`](https://github.com/google/yapf), [`autopep8`](https://github.com/hhatto/autopep8)] and manual adjustments is used.
For code formatting, a mix of [`yapf`](https://github.com/google/yapf), [`autopep8`](https://github.com/hhatto/autopep8)], both using default settings.
To format code using `yapf`:
```bash
yapf -i file_name.py
```
To format using `autopep8`, one can use:
```bash
autopep8 -iaa file_name.py
```
