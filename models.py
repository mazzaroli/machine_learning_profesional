import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# Import the Utils class from the utils module
from utils import Utils

class Models:
    def __init__(self) -> None:
        """
        Initialize the Models class.
        
        Initializes regression models and their corresponding hyperparameter grids.
        """
        self.reg = {
            'SVR': SVR(),  # Support Vector Regressor
            'GRADIENT': GradientBoostingRegressor(),  # Gradient Boosting Regressor
        }

        self.params = {
            'SVR': {
                'kernel': ['linear', 'poly', 'rbf'],  # Possible kernel types
                'gamma': ['auto', 'scale'],  # Possible values for gamma
                'C': [1, 5, 10],  # Possible values for regularization parameter C
            },
            'GRADIENT': {
                'loss': ['squared_error', 'absolute_error'],  # Loss functions
                'learning_rate': [0.01, 0.05, 0.1]  # Learning rates
            },
        }

    def grid_training(self, X: pd.DataFrame, y: pd.Series):
        """
        Perform grid search training for regression models.

        Iterates through each regression model and its hyperparameter grid.
        Finds the best model based on cross-validated scores and exports it using the Utils class.

        Parameters:
        X (pd.DataFrame): Features (input data).
        y (pd.Series): Target (output data).
        """
        # Initialize variables to track the best model and its score
        best_score = 999  # Initialize with a high value
        best_model = None

        # Iterate through each regression model and its hyperparameter grid
        for name, reg in self.reg.items():
            # Perform grid search with cross-validation
            grid_reg = GridSearchCV(reg, self.params[name], cv=3).fit(X, y.values.ravel())
            
            # Get the absolute value of the best score from grid search
            score = np.abs(grid_reg.best_score_)

            # Update the best model and its score if a better model is found
            if score < best_score:
                best_score = score
                best_model = grid_reg.best_estimator_

        # Create an instance of the Utils class
        utils = Utils()

        # Export the best model using the Utils class
        utils.model_export(best_model, best_score)
