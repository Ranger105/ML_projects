import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn import neighbors
from sklearn.metrics import accuracy_score

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

engine = create_engine('postgresql://tinku:tinku123@localhost:5432/nischay')

df = pd.read_sql_query('select * from iris_db', con=engine)
print(df)

#input
X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm',
       'PetalWidthCm']]
#output
y = df[['Species']]

#train-test-split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.5)

#model
classifier=neighbors.KNeighborsClassifier()
classifier.fit(X_train, y_train)

#prediction
predictions=classifier.predict(X_test)
print(accuracy_score(y_test,predictions))


# # Perform 6-fold cross validation
# scores = cross_val_score(model, df, y, cv=6)
# print('Cross-validated scores:', scores)
