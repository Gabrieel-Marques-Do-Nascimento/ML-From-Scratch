"""
/storage/emulated/0/__Projetos__/ML-From-Scratch/machine_learning_from_Scratch/knntest.py
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state =1234)
"""
print(X_train.shape)
print(X_train[0])

print(y_train.shape)
print(y_train)

plt.figure()
plt.scatter(X[:, 0], X[: , 1], c=y, cmap=cmap, edgecolor='k', s=20)
plt.show()
"""
"""
a = [1, 1,1,2,2,3,4,5,6]
from collections import Counter
mos_common = Counter(a).most_common(1)
print(mos_common)
# mostra uma tupla com o valor msis comum e o numero de vezes que ele aparece
print(mos_common[0][0])
#pegando os indices
"""
from knn import KNN

# classificador
clf = KNN(k=5)
# primeiro o metodo de ajuste
clf.fit(X_train, y_train)
# para prever as amostras de teste
predictions = clf.predict(X_test)

# caucular a precisao, ver quais previsoes estao corretas
# soma cada previsao que for igual y_test
# divide pelo numero de amostras do y_test
acc = np.sum(predictions ==y_test) / len(y_test)
# teste 1 == 1.0 estao corretos,   k=3 agora altera para k=5, normslmente sao numeros impars
# teste 2 == 966666666666 , nao foi tao bom mais , esta funcionando corretamente
print(acc)