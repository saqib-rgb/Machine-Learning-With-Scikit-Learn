#In this example we will create a simple model of classification using scikit-learn
# The algorithm used is called KNN(K-NearestNeighbors)
#In this algorithm we chhose a value of k the algorithm makes a scatter plot of the data
#on which it is trained and look around for K values and than decides which classification
#is to be used

#importing libraries
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
#loading the dataset from a csv file
churn_df=pd.read_csv('telecom_churn_clean.csv')
#subsetting data set for training the model and converting them to numpy array as the algo takes only arrays
X=churn_df[['total_day_charge', 'total_eve_charge']].values
y=churn_df['churn'].values
print(X.shape, y.shape)
#setting the value of k
knn=KNeighborsClassifier(n_neighbors=15)
#training the model
knn.fit(X, y)
#input for prediction
X_new=np.array([[56.8,17.5],[24.4,24.1],[50.1,10.9]])
#prediction
predictions=knn.predict(X_new)
print(predictions)


