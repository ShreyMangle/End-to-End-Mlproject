import unittest
import os
import numpy as np
import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TestMLPipeline(unittest.TestCase):

    def test_data_ingestion(self):
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        self.assertTrue(os.path.exists(train_path))
        self.assertTrue(os.path.exists(test_path))

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        self.assertGreater(len(train_df), 0)
        self.assertGreater(len(test_df), 0)

    def test_data_transformation(self):
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(train_path, test_path)

        self.assertIsInstance(train_arr, np.ndarray)
        self.assertIsInstance(test_arr, np.ndarray)
        self.assertTrue(os.path.exists(preprocessor_path))

    def test_model_trainer(self):
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        transformation = DataTransformation()
        train_arr, test_arr, _ = transformation.initiate_data_transformation(train_path, test_path)

        trainer = ModelTrainer()
        score = trainer.initiate_model_trainer(train_arr, test_arr)
        self.assertGreaterEqual(score, 0.0)  

if __name__ == "__main__":
    unittest.main()
