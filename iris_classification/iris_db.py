import pandas as pd
from sqlalchemy import create_engine
engine = create_engine('postgresql://tinku:tinku123@localhost/nischay')
#                                  username:password@localhost/database
con = engine.connect()
df = pd.read_csv('Iris.csv', delimiter=',')
df.to_sql('iris_db', con=con)