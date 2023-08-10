import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    data = pd.read_csv('./data/felicidad.csv')

    X = data.drop(['country','rank','score'],axis=1)
    y = data.score

    print(data)

    rfr = RandomForestRegressor()

    grid = {
        'n_estimators': range(4,16),
        'criterion': ['squared_error', 'absolute_error'],
        'max_depth': range(2,11),
    }

    rand_cv = RandomizedSearchCV(rfr, grid, n_iter=10, cv=5,scoring='neg_mean_absolute_error',random_state=42).fit(X,y)
    print(rand_cv.best_estimator_)

    print(rand_cv.predict(X.loc[[0]]))
    