import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn import neighbors
from sklearn.metrics import accuracy_score
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings("ignore")
engine = create_engine('postgresql://tinku:tinku123@localhost:5432/nischay')
df = pd.read_sql_query('select * from iris_db', con=engine)

#input
X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm',
       'PetalWidthCm']]
#output
encoder = LabelEncoder()
y = df[['Species']]

#train-test-split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.5)

#model
model=neighbors.KNeighborsClassifier()
model.fit(X_train, y_train)

#prediction
predictions=model.predict(X_test)
print(accuracy_score(y_test,predictions)*100)
