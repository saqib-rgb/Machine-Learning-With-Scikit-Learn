#In this example we will create a simple model of classification using scikit-learn
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
churn_df=pd.read_csv('telecom_churn_clean.csv')
X=churn_df[['total_day_charge', 'total_eve_charge']].values
y=churn_df['churn'].values
print(X.shape, y.shape)
knn=KNeighborsClassifier(n_neighbors=15)
knn.fit(X, y)
X_new=np.array([[56.8,17.5],[24.4,24.1],[50.1,10.9]])
predictions=knn.predict(X_new)
print(predictions)


