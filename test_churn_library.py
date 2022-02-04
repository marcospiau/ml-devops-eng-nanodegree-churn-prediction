"""
Tests for churn_library.py
Owner: marcospiau
Date: February 3, 2022
"""

import unittest
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from churn_library import (classification_report_image, encoder_helper,
                           feature_importance_plot, import_data, perform_eda,
                           perform_feature_engineering, train_models)


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
        with self.assertRaises(FileNotFoundError):
            import_data(str(uuid.uuid4()))


class MockDataTestCase(unittest.TestCase):
    """"Fake data for testing"""
    def create_fake_data(self):
        """Create fake data for testing"""
        self.quant_columns = ['quant_1', 'quant_2']
        self.cat_columns = ['cat_1', 'cat_2']
        self.response = 'target'
        self.df = pd.DataFrame({
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
        self.df = pd.concat([self.df] * 10, ignore_index=True)


class TestEda(MockDataTestCase):
    """Test EDA execution and output files"""
    def setUp(self):
        self.create_fake_data()
        self.output_dir = Path('./test_outputs/images/eda')

        perform_eda(df=self.df,
                    cat_columns=self.cat_columns,
                    quant_columns=self.quant_columns,
                    response=self.response,
                    output_dir=self.output_dir)

    def test_directory_tree_correct(self):
        """Test if all expcted folders are created"""
        folders_to_match = {
            'quantitative_features', 'categorical_features', 'target'
        }
        actual_folders = set(x.name for x in self.output_dir.iterdir())
        self.assertSetEqual(folders_to_match, actual_folders)

    def test_categorical_features_plots_created(self):
        """Test if categorical columns EDA plots are created"""
        names_pat = ['mean_response_%s.png', 'univariate_distribution_%s.png']
        files_to_match = set(name % col for col in self.cat_columns
                             for name in names_pat)
        created_files = set(x.name for x in (self.output_dir /
                                             'categorical_features').iterdir())
        self.assertSetEqual(files_to_match, set(created_files))

    def test_quant_features_plots_created(self):
        """Test if quantitative columns EDA plots are created"""
        names_pat = [
            'histogram_by_target_%s.png', 'univariate_histogram_%s.png'
        ]
        files_to_match = set(name % col for col in self.quant_columns
                             for name in names_pat)
        files_to_match.add('correlation_matrix_quant_columns.png')
        created_files = set(x.name
                            for x in (self.output_dir /
                                      'quantitative_features').iterdir())
        self.assertSetEqual(files_to_match, set(created_files))

    def test_target_plots(self):
        """Test if target column EDA plots are created"""
        expected_files = {'target_distribution.png'}
        actual_files = set(x.name
                           for x in (self.output_dir / 'target').iterdir())
        self.assertSetEqual(expected_files, actual_files)


class TestCategoricalEncoder(MockDataTestCase):
    """Tests for categorical encoding handling"""
    def setUp(self):
        self.create_fake_data()
        self.out = encoder_helper(df=self.df,
                                  category_lst=self.cat_columns,
                                  response=self.response)

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
    def setUp(self):
        self.create_fake_data()
        self.X_train, self.X_test, self.y_train, self.y_test = \
            perform_feature_engineering(
                df=self.df,
                cat_columns=self.cat_columns,
                quant_columns=self.quant_columns,
                response=self.response)

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
    def setUp(self) -> None:
        self.create_fake_data()
        self.labels_trues_preds = {
            'clf_1_train':
            (np.random.choice([0, 1], 10), np.random.choice([0, 1], 10)),
            'clf_2_test': (np.random.choice([0, 1],
                                            10), np.random.choice([0, 1], 10)),
        }
        self.results_dir = Path('./test_outputs/results')
        self.results_dir.mkdir(parents=True, exist_ok=True)

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
    def setUp(self):
        self.create_fake_data()
        self.feature_names = self.cat_columns + self.quant_columns
        self.feature_importances = np.arange(len(self.feature_names)) * 10 + 1
        self.results_dir = Path('./test_outputs/results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        feature_importance_plot(
            feature_names=self.feature_names,
            feature_importances=self.feature_importances,
            output_pth=self.results_dir / 'test_feature_importances.png')

    def test_expected_files_in_dir(self):
        """Test if the expected files are in output directory"""
        required_files = {'test_feature_importances.png'}
        actual_files = set(x.name for x in self.results_dir.iterdir())
        self.assertSetEqual(required_files, required_files & actual_files)


class TestTrain(MockDataTestCase):
    """Tests for train function"""
    def setUp(self):
        self.create_fake_data()
        self.models_dir = Path('./test_outputs/models')
        self.results_dir = Path('./test_outputs/results')
        self.X_train, self.X_test, self.y_train, self.y_test = \
            perform_feature_engineering(
                df=self.df,
                cat_columns=self.cat_columns,
                quant_columns=self.quant_columns,
                response=self.response)
        train_models(X_train=self.X_train,
                     X_test=self.X_test,
                     y_train=self.y_train,
                     y_test=self.y_test,
                     models_dir=self.models_dir,
                     results_dir=self.results_dir)

    def test_serialized_models_are_saved(self):
        """Test if serialized model objects are saved."""
        required_files = {'logistic_model.pkl', 'rfc_model.pkl'}
        actual_files = set(x.name for x in self.models_dir.iterdir())
        self.assertSetEqual(required_files, required_files & actual_files)

    def test_can_open_serialized_models(self):
        """Test if we can open the serialized models."""
        try:
            clf = joblib.load(self.models_dir / 'logistic_model.pkl')
            _ = clf.predict_proba(self.X_train)
        except:
            self.fail('Could not make predictions with loaded serialized '
                      'logistic regression model')
            raise
        try:
            clf = joblib.load(self.models_dir / 'rfc_model.pkl')
            _ = clf.predict_proba(self.X_train)
        except:
            self.fail('Could not make predictions with loaded serialized '
                      'random forest model')
            raise
