"""
Tests for churn_library.py
Owner: marcospiau
Date: February 3, 2022
"""

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import shutil
import yaml

from churn_library import (classification_report_image,
                           create_output_directory_tree, encoder_helper,
                           feature_importance_plot, import_data, perform_eda,
                           perform_feature_engineering, train_models,
                           load_yaml)


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
            with open(tmp_yaml_file, 'w') as outfile:
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
        cls.output_dir = Path('tests_outputs_2')
        if cls.output_dir.is_dir():
            shutil.rmtree(cls.output_dir)
        create_output_directory_tree(cls.output_dir)

    @classmethod
    def tearDownClass(cls):
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
                    output_dir=cls.output_dir/'images/eda')

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
        except:
            self.fail('Xy test data does not pass sklearn default data check.')
            raise  # forward sklearn exception

    def test_consistent_length_x_y_test(self):
        """Default X and y check for sklearn estimators"""
        try:
            sklearn.utils.check_X_y(self.X_test, self.y_test)
        except:
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
        cls.results_dir = Path('./test_outputs/results')
        cls.results_dir.mkdir(parents=True, exist_ok=True)

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
        feature_importance_plot(feature_names=cls.feature_names,
                                feature_importances=cls.feature_importances,
                                output_pth=cls.output_ /
                                'test_feature_importances.png')

    def test_models_with_coef(self):
        """Test we can use models with feature importances"""
        required_files = {'test_feature_importances.png'}
        actual_files = set(x.name for x in self.results_dir.iterdir())
        self.assertSetEqual(required_files, required_files & actual_files)        
