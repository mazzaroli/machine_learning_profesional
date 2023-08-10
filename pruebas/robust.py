import pandas as pd  
from sklearn.linear_model import RANSACRegressor, HuberRegressor  
from sklearn.svm import SVR  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_squared_error 
import warnings
warnings.simplefilter("ignore")

if __name__ == '__main__':
    # Carga los datos desde un archivo CSV en un DataFrame llamado "dataset".
    dataset = pd.read_csv('./data/felicidad_corrupt.csv')
    # print(dataset.head(5))  # Muestra las primeras 5 filas del DataFrame para visualizar los datos.

    # Divide el conjunto de datos en características (X) y variable objetivo (y).
    X = dataset.drop(['country', 'score'], axis=1)  # Selecciona todas las columnas excepto "country" y "score" como características de entrada (variables independientes).
    y = dataset.score  # Selecciona la columna "score" como la variable objetivo (variable dependiente).

    # Divide el conjunto de datos en conjuntos de entrenamiento y prueba.
    # El 70% de los datos se utilizará para entrenar los modelos, y el 30% restante se utilizará para evaluar el rendimiento.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

    # Crea un diccionario llamado "estimadores" que contiene tres instancias de estimadores diferentes: SVR, RANSACRegressor y HuberRegressor.
    # Cada estimador se inicializa con sus respectivos hiperparámetros.
    estimadores = {
        'SVR': SVR(gamma='auto', C=1.0, epsilon=0.1),
        'RANSAC': RANSACRegressor(),
        'HUBER': HuberRegressor(epsilon=1.35)
    }

    # Se itera sobre cada estimador y se evalúa su rendimiento.
    for name, estimador in estimadores.items():
        estimador.fit(X_train, y_train)  # Se entrena el estimador con el conjunto de entrenamiento.
        predictions = estimador.predict(X_test)  # Se realizan predicciones con el conjunto de prueba.

        print("=" * 32)  # Se imprime una línea de 32 "=" para separar los resultados de diferentes estimadores.
        print(name)  # Se imprime el nombre del estimador actual.
        # Se calcula el error cuadrático medio (Mean Squared Error, MSE) entre las etiquetas de prueba y las predicciones.
        print("mse:", mean_squared_error(y_test, predictions))