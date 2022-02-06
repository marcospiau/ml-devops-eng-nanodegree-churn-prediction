"""
Tests for churn_library.py
Owner: marcospiau
Date: February 3, 2022
"""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml

from churn_library import (classification_report_image,
                           create_output_directory_tree, encoder_helper,
                           feature_importance_plot, import_data,
                           load_model_cls, load_yaml, make_auc_plots,
                           perform_eda, perform_feature_engineering,
                           run_grid_search, train_models)


class TestImportData(unittest.TestCase):
    """Tests for data importing"""

    def test_ok_file(self):
        """Test loading of an OK file"""
        df = import_data('./test_data/ok_data.csv')
        self.assertTrue(not df.empty)

    def test_empty_data_error(self):
        """Test loading of a empty file"""
        with self.assertRaises(AssertionError):
            import_data('./test_data/empty_data.csv')

    def test_inexistent_file_error(self):
        """Test loading of a non existent file"""
        with tempfile.TemporaryDirectory() as tmpdirname:
            with self.assertRaises(FileNotFoundError):
                import_data(Path(tmpdirname) / 'this_file_doesnt_exist.csv')


class TestYamlLoading(unittest.TestCase):
    """Tests for yaml loading"""

    def test_load_fake_file(self):
        """Test if saved and loaded content is the same."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            expected_output = {'k1': 1, 'k2': 2, 'k3': {'k4': [1, 2]}}
            tmp_yaml_file = Path(tmpdirname) / 'test_yaml.yml'
            with open(tmp_yaml_file, 'w', encoding='utf8') as outfile:
                yaml.dump(expected_output, outfile)
            loaded_output = load_yaml(tmp_yaml_file)
            self.assertDictEqual(loaded_output, expected_output)


class TestCreateDirectoryTree(unittest.TestCase):
    """Tests for directory creation"""

    def test_error_if_exists(self):
        """Test if error is raised for already existent directory."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            with self.assertRaises(FileExistsError):
                create_output_directory_tree(output_dir=tmpdirname)

    def test_correct_creation(self):
        """Test if error is raised for already existent directory."""
        required_dirs = {
            'images/eda/categorical_features',
            'images/eda/quantitative_features', 'images/eda/target',
            'images/eda', 'images', 'models', 'results', 'logs'
        }
        with tempfile.TemporaryDirectory() as tmpdirname:
            output_dir = Path(tmpdirname) / 'tmp_output'
            required_dirs = set(output_dir / Path(x) for x in required_dirs)
            create_output_directory_tree(output_dir)
            actual_dirs = set(x for x in output_dir.rglob('*') if x.is_dir())
            self.assertSetEqual(required_dirs, actual_dirs)


class MockDataTestCase(unittest.TestCase):
    """Test main flow"""
    @classmethod
    def setUpClass(cls):
        """Initialize fake data and directory for tests"""
        cls.quant_columns = ['quant_1', 'quant_2']
        cls.cat_columns = ['cat_1', 'cat_2']
        cls.response = 'target'
        cls.df = pd.DataFrame({
            'target': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            'quant_1': [0.1, 0.2, 0.3, 0.9, 10, 12, 13, 18, 9, 10],
            'quant_2': [0, 1, 2, 3, 5.5, 6.6, 7.7, 8.8, 9.9, 10],
            'cat_1': ['0', '1', '2', '0', '1', '2', '0', '1', '2', '0'],
            'cat_2': [
                'Good', 'Bad', 'Bad', 'Bad', 'Bad', 'Bad', 'Bad', 'Bad',
                'Good', 'Bad'
            ]
        })
        # repeat data ten times
        cls.df = pd.concat([cls.df] * 10, ignore_index=True)
        cls.output_dir = Path('test_outputs')
        # cls.output_dir = Path(tempfile.mkdtemp())
        if cls.output_dir.is_dir():
            shutil.rmtree(cls.output_dir)
        create_output_directory_tree(cls.output_dir)

    @classmethod
    def tearDownClass(cls):
        if cls.output_dir.is_dir():
            shutil.rmtree(cls.output_dir)


class TestEda(MockDataTestCase):
    """Tests for EDA process"""
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        perform_eda(df=cls.df,
                    cat_columns=cls.cat_columns,
                    quant_columns=cls.quant_columns,
                    response=cls.response,
                    output_dir=cls.output_dir / 'images/eda')

    def test_eda_directory_tree_correct(self):
        """Test if all expcted EDA folders are created"""
        folders_to_match = {
            'quantitative_features', 'categorical_features', 'target'
        }
        actual_folders = set(
            x.name for x in (self.output_dir / 'images/eda').iterdir())
        self.assertSetEqual(folders_to_match, actual_folders)

    def test_eda_categorical_features_plots_created(self):
        """Test if categorical columns EDA plots are created"""
        names_pat = ['mean_response_%s.png', 'univariate_distribution_%s.png']
        files_to_match = set(name % col for col in self.cat_columns
                             for name in names_pat)
        dir_to_test = self.output_dir / 'images/eda/categorical_features'
        created_files = set(x.name for x in dir_to_test.iterdir())
        self.assertSetEqual(files_to_match, created_files)

    def test_eda_quant_features_plots_created(self):
        """Test if quantitative columns EDA plots are created"""
        names_pat = [
            'histogram_by_target_%s.png', 'univariate_histogram_%s.png'
        ]
        files_to_match = set(name % col for col in self.quant_columns
                             for name in names_pat)
        files_to_match.add('correlation_matrix_quant_columns.png')
        dir_to_test = self.output_dir / 'images/eda/quantitative_features'
        created_files = set(x.name for x in dir_to_test.iterdir())
        self.assertSetEqual(files_to_match, created_files)

    def test_eda_target_plot_created(self):
        """Test if target column EDA plots are created"""
        expected_files = {'target_distribution.png'}
        dir_to_test = self.output_dir / 'images/eda/target'
        actual_files = set(x.name for x in dir_to_test.iterdir())
        self.assertSetEqual(expected_files, actual_files)


class TestCategoricalEncoder(MockDataTestCase):
    """Tests for categorical encoding handling"""
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.out = encoder_helper(df=cls.df,
                                 category_lst=cls.cat_columns,
                                 response=cls.response)

    def test_correct_shape(self):
        """Test if output has correct shape"""
        expected_shape = self.df[self.cat_columns].shape
        actual_shape = self.out.shape
        self.assertEqual(expected_shape, actual_shape)

    def test_correct_feature_names(self):
        """Test if output column names follow the expected pattern"""
        expected_names = [
            f'{col}_mean_{self.response}' for col in self.cat_columns
        ]
        actual_names = self.out.columns.tolist()
        self.assertListEqual(expected_names, actual_names)

    def test_index_not_modified(self):
        """Test if output and intput indexes are equal"""
        self.assertIsNone(
            pd.testing.assert_index_equal(self.df.index, self.out.index))

    def test_correct_dtype(self):
        """Test if the output has correct data type"""
        self.assertTrue(self.out.dtypes.eq(np.float32).all())

    def check_correct_shape(self):
        """Check if output shape is same as input"""
        self.assertTupleEqual(self.df[self.cat_columns].shape, self.out.shape)


class TestFeatureEngineering(MockDataTestCase):
    """Test feature engineering process"""
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = \
            perform_feature_engineering(
                df=cls.df,
                cat_columns=cls.cat_columns,
                quant_columns=cls.quant_columns,
                response=cls.response)

    def test_x_dtype(self):
        """Test if Xs dtypes are correct"""
        self.assertTrue(self.X_train.dtypes.eq(np.float32).all())
        self.assertTrue(self.X_test.dtypes.eq(np.float32).all())

    def test_y_dtype(self):
        """Test if y dtype is correct"""
        self.assertEqual(self.y_train.dtype, np.int64)
        self.assertEqual(self.y_test.dtype, np.int64)

    def test_consistent_length_x_y_train(self):
        """Default X and y check for sklearn estimators"""
        try:
            sklearn.utils.check_X_y(self.X_train, self.y_train)
        except BaseException:
            self.fail('Xy test data does not pass sklearn default data check.')
            raise  # forward sklearn exception

    def test_consistent_length_x_y_test(self):
        """Default X and y check for sklearn estimators"""
        try:
            sklearn.utils.check_X_y(self.X_test, self.y_test)
        except BaseException:
            self.fail('Xy test data does not pass sklearn default data check.')
            raise  # forward sklearn exception

    def test_correct_feature_names(self):
        """Test if output feature names are correct"""
        expected_names = self.quant_columns + [
            f'{col}_mean_{self.response}' for col in self.cat_columns
        ]
        self.assertListEqual(expected_names, self.X_train.columns.tolist())
        self.assertListEqual(expected_names, self.X_test.columns.tolist())


class TestClassificationReportPlots(MockDataTestCase):
    """Tests for classification report"""
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        fake_trues = np.r_[np.ones(10), np.zeros(10)]
        fake_preds = np.r_[np.ones(10), np.zeros(10)]
        cls.labels_trues_preds = {
            'clf_1_train': (fake_trues, fake_preds),
            'clf_2_test': (fake_trues, fake_preds)
        }
        cls.results_dir = cls.output_dir / 'results'

    def test_simple_plot(self):
        """Test we can make a simple plot"""
        output_file = self.results_dir / 'test_classification_report_simple.png'
        classification_report_image(labels_trues_preds=self.labels_trues_preds,
                                    output_file=output_file)
        self.assertTrue(Path(output_file).is_file())

    def test_kwargs_plot(self):
        """Test we can plot passing kwargs to sklearn classification report"""
        output_file = self.results_dir / 'test_classification_report_kwargs.png'
        classification_report_image(labels_trues_preds=self.labels_trues_preds,
                                    output_file=output_file,
                                    digits=4)
        self.assertTrue(Path(output_file).is_file())


class TestFeatureImportancePlots(MockDataTestCase):
    """Test for feature importance plots"""
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.feature_names = cls.cat_columns + cls.quant_columns
        cls.feature_importances = np.arange(len(cls.feature_names)) * 10 + 1
        cls.results_dir = cls.output_dir / 'results'

    def test_model_with_coef_and_no_intercept(self):
        """Test for models with coef_ and no intercept_"""
        model = mock.MagicMock()
        model.coef_ = np.arange(10, dtype=float)
        mock.seal(model)
        feature_names = [f'feat_{x}' for x in range(10)]
        output_file = (self.results_dir /
                       'test_model_with_coef_and_no_intercept.png')
        feature_importance_plot(model=model,
                                output_pth=output_file,
                                feature_names=feature_names)
        self.assertTrue(output_file.is_file())

    def test_model_with_coef_and_intercept(self):
        """Test for models with coef_ and intercept_"""
        model = mock.MagicMock()
        model.coef_ = np.arange(10, dtype=float)
        model.intercept_ = np.array([100])
        mock.seal(model)
        feature_names = [f'feat_{x}' for x in range(10)]
        output_file = (self.results_dir /
                       'test_model_with_coef_and_intercept.png')
        feature_importance_plot(model=model,
                                output_pth=output_file,
                                feature_names=feature_names)
        self.assertTrue(output_file.is_file())

    def test_model_with_coef_and_wrong_intercept(self):
        """Test an exception is raised if intercept_ is not a single-element
            array
        """
        model = mock.MagicMock()
        model.coef_ = np.arange(10, dtype=float)
        model.intercept_ = np.zeros(2)
        mock.seal(model)
        feature_names = [f'feat_{x}' for x in range(10)]
        output_file = (self.results_dir /
                       'test_model_with_coef_and_wrong_intercept.png')
        with self.assertRaises(AssertionError):
            feature_importance_plot(model=model,
                                    output_pth=output_file,
                                    feature_names=feature_names)
        self.assertFalse(output_file.is_file())

    def test_model_with_feature_importances(self):
        """Test for models with feature_importances_"""
        model = mock.MagicMock()
        model.feature_importances_ = np.arange(10, dtype=float)
        mock.seal(model)
        feature_names = [f'feat_{x}' for x in range(10)]
        output_file = (self.results_dir / 'test_model_feature_importances.png')
        feature_importance_plot(model=model,
                                output_pth=output_file,
                                feature_names=feature_names)
        self.assertTrue(output_file.is_file())

    def test_model_with_coef_and_feature_names_in(self):
        """Test for models with coef_ and feature_names_in_"""
        model = mock.MagicMock()
        model.coef_ = np.arange(10, dtype=float)
        model.feature_names_in_ = [f'feat_{x}' for x in range(10)]
        mock.seal(model)
        output_file = (self.results_dir /
                       'test_model_with_coef_and_feature_names_in.png')
        feature_importance_plot(model=model, output_pth=output_file)
        self.assertTrue(output_file.is_file())

    def test_no_feature_names_provided(self):
        """Test if raises exception if no feature_names information is
        provided
        """
        model = mock.MagicMock()
        model.coef_ = np.arange(10, dtype=float)
        mock.seal(model)
        output_file = (self.results_dir / 'test_no_feature_names_provided.png')
        with self.assertRaises(AssertionError):
            feature_importance_plot(model=model, output_pth=output_file)
        self.assertFalse(output_file.is_file())

    def test_model_with_coef_intercept_and_feature_names_in(self):
        """Test for models with coef_, intercept_ and feature_names_in_"""
        model = mock.MagicMock()
        model.coef_ = np.arange(10, dtype=float)
        model.intercept_ = np.array([10])
        model.feature_names_in_ = [f'feat_{x}' for x in range(10)]
        mock.seal(model)
        output_file = (
            self.results_dir /
            'test_model_with_coef_intercept_and_feature_names_in.png')
        feature_importance_plot(model=model, output_pth=output_file)
        self.assertTrue(output_file.is_file())

    def test_model_with_feature_importances_and_feature_names_in_(self):
        """Test for models with feature_importances_ and feature_names_in_"""
        model = mock.MagicMock()
        model.feature_importances_ = np.arange(10, dtype=float)
        model.feature_names_in_ = [f'feat_{x}' for x in range(10)]
        mock.seal(model)
        feature_names = [f'feat_{x}' for x in range(10)]
        output_file = (
            self.results_dir /
            'test_model_with_feature_importances_and_feature_names_in_.png')
        feature_importance_plot(model=model,
                                output_pth=output_file,
                                feature_names=feature_names)
        self.assertTrue(output_file.is_file())

    def test_no_feature_importances_provided(self):
        """Test if an exception is raised if no feature importance information
            is provided
        """
        model = mock.MagicMock()
        model.feature_names_in_ = [f'feat_{x}' for x in range(10)]
        mock.seal(model)
        output_file = (self.results_dir /
                       'test_no_feature_importances_provided.png')
        with self.assertRaises(AttributeError):
            feature_importance_plot(model=model, output_pth=output_file)
        self.assertFalse(output_file.is_file())


class TestAUCPlots(MockDataTestCase):
    """Tests for classification report"""
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        fake_trues = np.r_[np.ones(10), np.zeros(10)]
        fake_preds = np.r_[np.ones(10), np.zeros(10)]
        cls.labels_trues_preds = {
            'clf_1_train': (fake_trues, fake_preds),
            'clf_2_test': (fake_trues, fake_preds)
        }
        cls.results_dir = cls.output_dir / 'results'

    def test_plot_single_curve(self):
        """Test we can plot a single curve"""
        output_file = self.results_dir / 'test_plot_single_curve.png'

        make_auc_plots(preds_and_trues={
            'clf_1_train': self.labels_trues_preds['clf_1_train']
        },
            output_file=output_file)
        self.assertTrue(output_file.is_file())

    def test_plot_multiple_curves(self):
        """Test we can plot multiple curves"""
        output_file = self.results_dir / 'test_plot_multiple_curves.png'
        make_auc_plots(preds_and_trues=self.labels_trues_preds,
                       output_file=output_file)
        self.assertTrue(output_file.is_file())


# pylint:disable=import-outside-toplevel
class TestLoadModelCls(unittest.TestCase):
    """Tests for class loading with importlib. It is important to add tests for
        each estimator before using it on yaml config to avoid errors. Each
        test here. We disable pylint import-outside-toplevel because we are
        directly testing modules can be imported, so it makes sense to isolate
        these imports inside each test method.
    """

    def test_random_forest_classifier(self):
        """Test we can use RandomForesClassifier"""
        from sklearn.ensemble import RandomForestClassifier
        cand = load_model_cls('sklearn.ensemble.RandomForestClassifier')
        self.assertEqual(cand, RandomForestClassifier)

    def test_logistic_regression(self):
        """Test we can use LogisticRegression"""
        from sklearn.linear_model import LogisticRegression
        cand = load_model_cls('sklearn.linear_model.LogisticRegression')
        self.assertEqual(cand, LogisticRegression)

    def test_decision_tree_classifier(self):
        """Test we can use DecisionTreeClassifier"""
        from sklearn.tree import DecisionTreeClassifier
        cand = load_model_cls('sklearn.tree.DecisionTreeClassifier')
        self.assertEqual(cand, DecisionTreeClassifier)


class TestRunGrid(MockDataTestCase):
    """Tests for grid search"""
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = \
            perform_feature_engineering(
                df=cls.df,
                cat_columns=cls.cat_columns,
                quant_columns=cls.quant_columns,
                response=cls.response)

    def test_can_run_dummy_grid(self):
        """Test we can run a DummyClassifier grid search"""
        from sklearn.dummy import DummyClassifier
        grid_kwargs = {
            'estimator': DummyClassifier(),
            'n_iter': 4,
            'n_jobs': 1,
            'param_distributions': {
                'strategy':
                ['most_frequent', 'prior', 'stratified', 'uniform'],
            }
        }
        model = run_grid_search(X=self.X_train, y=self.y_train, **grid_kwargs)
        self.assertIsNone(sklearn.utils.validation.check_is_fitted(model))

    def test_can_run_random_forest(self):
        """Test we can run a RandomForestClassifier grid search"""
        from sklearn.ensemble import RandomForestClassifier
        grid_kwargs = {
            'estimator': RandomForestClassifier(),
            'n_iter': 4,
            'n_jobs': 1,
            'param_distributions': {
                'n_estimators': [1, 100],
                'max_depth': [2, 3],
                'max_features': ['sqrt', 'auto']
            }
        }
        model = run_grid_search(X=self.X_train, y=self.y_train, **grid_kwargs)
        self.assertIsNone(sklearn.utils.validation.check_is_fitted(model))

    def test_can_run_logistic_regression(self):
        """Test we can run a RandomForestClassifier grid search"""
        from sklearn.linear_model import LogisticRegression
        grid_kwargs = {
            'estimator': LogisticRegression(),
            'n_iter': 2,
            'n_jobs': 1,
            'param_distributions': {
                'class_weight': ['balanced', None]
            }
        }
        model = run_grid_search(X=self.X_train, y=self.y_train, **grid_kwargs)
        self.assertIsNone(sklearn.utils.validation.check_is_fitted(model))


class TestTrainModels(MockDataTestCase):
    """Tests for train_models function"""
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = \
            perform_feature_engineering(
                df=cls.df,
                cat_columns=cls.cat_columns,
                quant_columns=cls.quant_columns,
                response=cls.response)
        cls.config = {
            'output_dir': cls.output_dir,
            'models': {
                'rfc': {
                    'model_cls': 'sklearn.ensemble.RandomForestClassifier',
                    'grid_params': {
                        'n_iter': 4,
                        'n_jobs': 1,
                        'param_distributions': {
                            'n_estimators': [1, 100],
                            'max_depth': [2, 3],
                            'max_features': ['sqrt', 'auto']
                        }
                    }
                },
                'logistic_model': {
                    'model_cls': 'sklearn.linear_model.LogisticRegression',
                    'grid_params': {
                        'n_iter': 2,
                        'n_jobs': 1,
                        'param_distributions': {
                            'class_weight': ['balanced', None]
                        }
                    }
                }
            }
        }
        cls.models_dir = Path(cls.output_dir / 'models')
        cls.results_dir = Path(cls.output_dir / 'results')

        train_models(X_train=cls.X_train,
                     X_test=cls.X_test,
                     y_train=cls.y_train,
                     y_test=cls.y_test,
                     config=cls.config)

    def test_serialized_models_are_saved(self):
        """Test if serialized models are saved"""
        required_files = set(f'best_{model_name}.pkl'
                             for model_name in self.config['models'])
        actual_files = set(x.name for x in self.models_dir.iterdir())
        self.assertSetEqual(required_files, actual_files)

    def test_feature_importance_plots_are_saved(self):
        """Test if feature importance plots are saved"""
        required_files = set(f'{model_name}_feature_importances.png'
                             for model_name in self.config['models'])
        actual_files = set(x.name for x in self.results_dir.iterdir()
                           if x.name.endswith('_feature_importances.png'))
        self.assertSetEqual(required_files, required_files & actual_files)

    def test_auc_plot_is_saved(self):
        """Test if AUC plot is saved"""
        actual_files = set(x.name for x in self.results_dir.iterdir())
        self.assertIn('auc_roc_curves.png', actual_files)

    def test_classification_report_image_is_saved(self):
        """Test if classification report image is saved"""
        actual_files = set(x.name for x in self.results_dir.iterdir())
        self.assertIn('classification_report_image.png', actual_files)

    def test_load_and_make_predictions_serialized_models(self):
        """Test if we can open the serialized models."""
        for model_name in self.config['models']:
            model = joblib.load(self.models_dir / f'best_{model_name}.pkl')
            # Check if model is fitted
            self.assertIsNone(sklearn.utils.validation.check_is_fitted(model))
            # Check predictions
            preds = model.predict_proba(self.X_train)
            self.assertIsInstance(preds, np.ndarray)
