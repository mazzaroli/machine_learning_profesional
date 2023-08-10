# Importamos las bibliotecas necesarias
import pandas as pd  # Para el manejo de datos estructurados
import sklearn  # Para machine learning en Python
import matplotlib.pyplot as plt  # Para la visualización de gráficos

# Importamos las clases específicas de scikit-learn
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA  # Para análisis de componentes principales
from sklearn.linear_model import LogisticRegression  # Para regresión logística
from sklearn.preprocessing import StandardScaler  # Para estandarizar características
from sklearn.model_selection import train_test_split  # Para dividir los datos en entrenamiento y prueba

# Iniciamos la ejecución del programa principal
if __name__ == "__main__":
    df_heart = pd.read_csv('./data/heart.csv') # Se lee el archivo CSV 'heart.csv' y se guarda en el DataFrame df_heart
    
    # print(df_heart.head(5)) # Se imprime las primeras 5 filas del DataFrame df_heart

    df_features = df_heart.drop(['target'], axis=1) # Se crea un nuevo DataFrame df_features eliminando la columna 'target' del DataFrame df_heart
    
    df_target = df_heart['target'] # Se crea una Serie df_target con los valores de la columna 'target' del DataFrame df_heart

    df_features = StandardScaler().fit_transform(df_features) # Se estandarizan las características del DataFrame df_features utilizando la clase StandardScaler

    # Se dividen los datos en conjuntos de entrenamiento y prueba utilizando train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df_features,df_target,test_size=0.3,random_state=42)

    # print(X_train.shape)
    # print(y_train.shape)

    #n_components = min(n_muestras, n_features)
    pca = PCA(n_components=4)  
    pca.fit(X_train)  

    # Realizamos el análisis de componentes principales incremental (IPCA)
    ipca = IncrementalPCA(n_components=4, batch_size=10)  
    ipca.fit(X_train)  

    # Realizamos el análisis de componentes principales utilizando KernelPCA (KPCA)
    kpca = KernelPCA(n_components=4, kernel='poly')  
    kpca.fit(X_train)  

    # Graficamos la varianza explicada por cada componente principal utilizando Matplotlib
    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)  
    plt.savefig('a')  

    # Creamos un clasificador de regresión logística
    logistic = LogisticRegression(solver='lbfgs')

    # Transformamos los datos de entrenamiento y prueba utilizando PCA
    df_train = pca.transform(X_train)  
    df_test = pca.transform(X_test)  
    logistic.fit(df_train, y_train)  
    print("SCORE PCA:", logistic.score(df_test, y_test))  

    # Transformamos los datos de entrenamiento y prueba utilizando IPCA
    df_train = ipca.transform(X_train)  
    df_test = ipca.transform(X_test)  
    logistic.fit(df_train, y_train)  
    print("SCORE IPCA:", logistic.score(df_test, y_test))  

    # Transformamos los datos de entrenamiento y prueba utilizando KPCA
    df_train = kpca.transform(X_train)  
    df_test = kpca.transform(X_test)  
    logistic.fit(df_train, y_train)  
    print("SCORE KPCA:", logistic.score(df_test, y_test))

    