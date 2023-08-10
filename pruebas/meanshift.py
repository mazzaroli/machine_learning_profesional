import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import MeanShift

if __name__ == '__main__':
    data = pd.read_csv('./data/candy.csv')
    # print(data)
    
    X = data.drop('competitorname', axis=1)

    meanshift = MeanShift().fit(X)
    print(max(meanshift.labels_))

    print('--'*32)

    print(meanshift.cluster_centers_)

    print('--'*32)

    data['meanshift'] = meanshift.labels_
    print(data)
    
    sns.scatterplot(data=data, x='winpercent',y='sugarpercent',hue='meanshift', palette='colorblind')
    sns.pairplot(data[['sugarpercent','pricepercent', 'winpercent','meanshift']], hue='meanshift',palette='colorblind')