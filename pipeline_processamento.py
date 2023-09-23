import pandas as pd
import matplotlib.pyplot as plt

data = {'Name': ['Anna', 'Bob', ' Chalie', 'Diana', 'Eric'],
        'Age': [20, 34, 23, None, 33],
        'Gender': ['f', 'm', 'm', 'f', 'm' ],
        'Job': ['Programmer', 'Writer', 'Cook', 'Programmer', 'Teacher']}

df = pd.DataFrame(data)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

#Removendo a coluna 'Name'
df =  df.drop(['Name'], axis=1) 

#Imputando valores faltantes ('Age')
imputer = SimpleImputer(strategy='mean')
df['Age'] = imputer.fit_transform(df[['Age']])

# Genero de forma binária
gender_dct = {'m': 0, 'f': 1}
df['Gender'] = [gender_dct[g] for g in df['Gender']]

# Transformando linhas em colunas (Job)
encoder = OneHotEncoder()
matrix = encoder.fit_transform(df[['Job']]).toarray()

column_names = ["Programmer", "Writer", "Cook", "Teacher"]

for i in range(len(matrix.T)):
    df[column_names[i]] = matrix.T[i]

df = df.drop(['Job'], axis=1)

######### AUTOMATIZANDO O PROCESSO #########

#Removendo nomes ('Names')
from sklearn.base import BaseEstimator, TransformerMixin

class NameDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.drop(['Name'], axis=1)
    
data = {'Name': ['Fiona', 'Geral', ' Hans', 'Isabella', 'Jacob'],
        'Age': [20, 34, None, None, 33],
        'Gender': ['f', 'm', 'm', 'f', 'm' ],
        'Job': ['Writer', 'Programmer', 'Programmer', 'Programmer', 'Teacher']}

df2 = pd.DataFrame(data)

dropper = NameDropper()
dropper.fit_transform(df2)

#Imputando valores de idade ('Age')
class AgeImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        imputer = SimpleImputer(strategy='mean')
        X['Age'] = imputer.fit_transform(X[['Age']])
        return X
#Genero de forma binária ('Gender') e transformando linhas em colunas ('Job')
class FeatureEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        gender_dct = {'m': 0, 'f': 1}
        X['Gender'] = [gender_dct[g] for g in X['Gender']]
    
        encoder = OneHotEncoder()
        matrix = encoder.fit_transform(X[['Job']]).toarray()

        column_names = ["Programmer", "Writer", "Cook", "Teacher"]

        for i in range(len(matrix.T)):
            X[column_names[i]] = matrix.T[i]

        return X.drop(['Job'], axis=1)

dropper = NameDropper()
imp = AgeImputer()
enc = FeatureEncoder()

enc.fit_transform(imp.fit_transform(dropper.fit_transform(df2)))

########## PIPELINE ##########
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('dropper', NameDropper()),
    ('imputer', AgeImputer()),
    ('encoder', FeatureEncoder())
])

print(pipe.fit_transform(df2))