import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import warnings
warnings.simplefilter("ignore")

if __name__ == "__main__":
    df_heart = pd.read_csv('./data/heart.csv')
    print(df_heart)
    print(df_heart.target.describe())
    
    # Target
    # count    1025.000000
    # mean        0.513171
    # std         0.500070
    # min         0.000000
    # 25%         0.000000
    # 50%         1.000000
    # 75%         1.000000
    # max         1.000000

    X = df_heart.drop('target', axis=1)
    y = df_heart.target

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.35,random_state=42)

    classifier = {
        'KNeighbors': KNeighborsClassifier(),
        'LinearSCV': LinearSVC(),
        'SVC': SVC(),
        'SGDC': SGDClassifier(),
        'DecisionTree': DecisionTreeClassifier()
    }

    for name,model in classifier.items():
        model_class = model.fit(X_train,y_train,)
        y_pred = model.predict(X_test)
        print('-'*32)
        print(f'{name}: {accuracy_score(y_test,y_pred)}')

        bag_class = BaggingClassifier(base_estimator=model, n_estimators=10,random_state=42).fit(X_train,y_train)
        y_pred = bag_class.predict(X_test)

        print(f'Bagging {name}: {accuracy_score(y_test,y_pred)}')

    GBC_class = GradientBoostingClassifier(n_estimators=50,random_state=42).fit(X_train,y_train)
    y_pred = GBC_class.predict(X_test)
    print('='*32)
    print(f'Gradien Boosting Classifier {name}: {accuracy_score(y_test,y_pred)}')
