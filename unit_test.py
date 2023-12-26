import unittest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import argparse

# Assuming the functions to be tested are defined in a module named 'train'
from train import loadData, preprocessData, buildModel, assessModel

class TestTrain(unittest.TestCase):

    def setUp(self):
        # Load the dataset
        self.data = loadData(r"C:\Users\user\Documents\PRACTICE\train_dataset.csv")
        #parser = argparse.ArgumentParser()
        #self.data = loadData('--testingdata', type=str, required=True, help='Dataset for testing')
        #args = parser.parse_args()

        
        # Splitting the dataset into features and target
        self.X = self.data.iloc[:, :-1]
        self.y = self.data.iloc[:, -1]

        # Splitting the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42)

    def test_loadData(self):
        # Test that the data is loaded correctly
        self.assertIsNotNone(self.data)
        self.assertGreater(len(self.data), 0)

    def test_preprocessData(self):
        # Test preprocessing steps (if any)
        processed_X = preprocessData(self.X)
        # Add specific tests for your preprocessing here
        # Example: self.assertEqual(processed_X.shape[1], expected_number_of_features_after_preprocessing)

    def test_buildModel(self):
        # Test the model building function
        model = buildModel(self.X_train, self.y_train)
        self.assertIsInstance(model, RandomForestClassifier)

    def test_assessModel(self):
        # Test the model assessment function
        model = buildModel(self.X_train, self.y_train)
        accuracy = assessModel(model, self.X_test, self.y_test)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

if __name__ == '__main__':
    unittest.main()
