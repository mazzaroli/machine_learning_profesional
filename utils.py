import pandas as pd
import joblib

class Utils:
    def load_from_csv(self, path: str) -> pd.DataFrame:
        """
        Load data from a CSV file.

        Parameters:
        path (str): Path to the CSV file.

        Returns:
        pd.DataFrame: A DataFrame containing the data loaded from the CSV file.
        """
        return pd.read_csv(path)
    
    def load_from_mysql(self):
        """
        Load data from a MySQL database.

        Note:
        This function is empty as MySQL loading functionality is not implemented in this example.
        """
        print('mysql')
    
    def features_target(self, dataset: pd.DataFrame, drop_cols: list, y: str) -> tuple:
        """
        Split the dataset into features (X) and target (y).

        Parameters:
        dataset (pd.DataFrame): The original dataset.
        drop_cols (list): List of columns to drop to obtain features.
        y (str): Name of the target column.

        Returns:
        tuple: A tuple containing feature DataFrame (X) and target Series (y).
        """
        X = dataset.drop(drop_cols, axis=1)
        y = dataset[y]
        return X, y

    def model_export(self, clf, score):
        """
        Export the trained model.

        Note:
        This function is empty as model export functionality is not implemented in this example.

        Parameters:
        clf: Trained model.
        score: Model's score or metric.
        """
        joblib.dump(clf, './models/best_model_'+str(round(score,3))+'.pkl') 