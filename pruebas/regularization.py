# Importamos las bibliotecas necesarias
from matplotlib import pyplot as plt
import pandas as pd  # Para el manejo de datos estructurados

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    dataset = pd.read_csv('./data/whr2017.csv')
    
    # Separar las características (X) y la variable objetivo (y)
    X = dataset[['gdp', 'family', 'lifexp', 'freedom', 'generosity', 'corruption', 'dystopia']]
    y = dataset[['score']]

    # Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Entrenar los modelos de regresión lineal, Lasso, Ridge y ElasticNet
    model_lineal = LinearRegression().fit(X_train, y_train)
    model_lasso = Lasso(alpha=0.02).fit(X_train, y_train)
    model_ridge = Ridge(alpha=1).fit(X_train, y_train)
    model_elastic = ElasticNet(alpha=0.01).fit(X_train, y_train)
    
    # Realizar predicciones en el conjunto de prueba
    y_predict_lineal  = model_lineal.predict(X_test)
    y_predict_lasso   = model_lasso.predict(X_test)
    y_predict_ridge   = model_ridge.predict(X_test)
    y_predict_elastic = model_elastic.predict(X_test)

    # Calcular el MSE para cada modelo
    lineal_loss = mean_squared_error(y_test, y_predict_lineal)
    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    elastic_loss = mean_squared_error(y_test, y_predict_elastic)

    # Imprimir los valores de MSE para cada modelo
    print('lineal_loss: ', lineal_loss)
    print('lasso_loss:  ', lasso_loss)
    print('ridge_loss:  ', ridge_loss)
    print('elastic_loss:', elastic_loss, '\n')

    # Crear un DataFrame con los coeficientes de cada modelo
    data = pd.DataFrame(columns=['lineal', 'lasso', 'ridge', 'elastic'])
    data['lineal'] = model_lineal.coef_.reshape(-1)
    data['lasso'] = model_lasso.coef_.reshape(-1)
    data['ridge'] = model_ridge.coef_.reshape(-1)
    data['elastic'] = model_elastic.coef_.reshape(-1)
    
    # Establecer los índices del DataFrame como las columnas de entrada X
    data.index = X.columns
    
    # Imprimir los coeficientes de cada modelo
    print(data,'\n')




# lineal_loss:  9.893337283086869e-08
# lasso_loss:   0.049605751139829145
# ridge_loss:   0.005650124499962814
# elastic_loss: 0.00912409728272294

#               lineal     lasso     ridge   elastic
# gdp         1.000128  1.289214  1.072349  1.106542
# family      0.999946  0.919694  0.970486  0.962883
# lifexp      0.999835  0.476864  0.856054  0.803020
# freedom     1.000034  0.732973  0.874002  0.861674
# generosity  1.000260  0.142455  0.732857  0.654667
# corruption  0.999771  0.000000  0.685833  0.554539
# dystopia    0.999938  0.899653  0.962066  0.953720