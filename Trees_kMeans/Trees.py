#отключим предупреждения Anaconda, бесят)
import warnings
warnings.filterwarnings('ignore')

from pandas import read_csv
from sklearn import tree
import graphviz

data = read_csv('DATA.csv')
x = data[['Возраст', 'Зарплата']].values
#x = data['Возраст'].values.reshape(-1, 1)
y = data['Должник'].values

model = tree.DecisionTreeClassifier(criterion='entropy', random_state=17)
model = model.fit(x, y)

#сохраняем картиночку дерева
dot_data = tree.export_graphviz(model, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("tree")