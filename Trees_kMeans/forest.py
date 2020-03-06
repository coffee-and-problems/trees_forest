import warnings
warnings.filterwarnings('ignore')

from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier

data = read_csv('DATA.csv')
x = data[['Возраст', 'Зарплата']].values
y = data['Должник'].values
model = RandomForestClassifier(n_estimators=100, bootstrap = True, max_features = 'sqrt')
model.fit(x, y)
print(model.score(x,y))

import numpy as np
a = np.array([[239],[100]]).reshape(1,2)
print(model.predict(a))