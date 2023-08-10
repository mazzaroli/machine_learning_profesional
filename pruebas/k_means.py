import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans,MiniBatchKMeans

import warnings
warnings.simplefilter("ignore")

if __name__ == '__main__':
    df_candy = pd.read_csv('./data/candy.csv')
    print(df_candy.head(5))

    X = df_candy.drop('competitorname',axis=1)
    y = df_candy.competitorname


    cluster_model = {
        "KMeans": KMeans(n_clusters=4),
        "MiniKMeans": MiniBatchKMeans(n_clusters=4,batch_size=8)
    }
    
    for name,model in cluster_model.items():
        model = model.fit(X)
        df_candy[name] = model.predict(X)

    fig,ax = plt.subplots(1,2,figsize=(20,10))
    
    sns.scatterplot(ax=ax[0],data=df_candy, x='winpercent',y='sugarpercent',hue='KMeans', palette='colorblind')
    sns.scatterplot(ax=ax[1],data=df_candy, x='winpercent',y='sugarpercent',hue='MiniKMeans', palette='colorblind')
    sns.pairplot(df_candy[['sugarpercent','pricepercent', 'winpercent','KMeans']], hue='KMeans',palette='colorblind')
    sns.pairplot(df_candy[['sugarpercent','pricepercent', 'winpercent','MiniKMeans']], hue='MiniKMeans',palette='colorblind')
