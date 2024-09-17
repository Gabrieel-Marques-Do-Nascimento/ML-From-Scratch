"""
CONHECIDO COMO APRENDIZADO PREGUICOSO
-----------------------------------------------------
Exemplos reais de aplicação do KNN:
Reconhecimento de padrões de imagem:

O KNN pode ser usado para classificar imagens, por exemplo, na detecção de dígitos escritos à mão (como o famoso
dataset MNIST). O algoritmo classifica cada imagem de um dígito com base nos vizinhos mais próximos no espaço de
características da imagem.

Classificação de textos (Spam/Não Spam):

O KNN pode ser aplicado para classificar e-mails em "spam" ou "não spam" com base em suas características (como a
frequência de palavras, presença de certos termos, etc.).

Previsão de doenças médicas:

No campo médico, o KNN é usado para prever o diagnóstico de doenças com base em atributos do paciente (idade,
níveis de colesterol, etc.) comparando com casos já conhecidos.

Sistemas de recomendação:

Em sistemas de recomendação, como sugerir produtos similares com base nas compras de outros usuários. O KNN pode
comparar a similaridade de usuários ou produtos, recomendando aqueles que estão mais "próximos" no espaço de
características.

/storage/emulated/0/__Projetos__/ML-From-Scratch/machine_learning_from_Scratch/KNN.py
"""
from collections import Counter

import numpy as np


def eucliendean_distance(x1, x2):
    """
    """

    # raisquadrada(
    #        somar( (x1 - x2 ) elevado a 2)
    #  cada componente do vetor
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, k=3):
        """
        k: o numero de visinhos mais proximos.defaut( 3 )
        """
        self.k = k

    def fit(self, X, y):
        """
        metodo de ajuste, isso ajustara as amostras de treinamento e alguns rotulos de treinamento, envolvera uma etapa de trinamento
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
     metodo de previsao, para prever novas amostras
        """
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # caucular as distancias
        # compute distances
        distances = [eucliendean_distance(x, x_train) for x_train in self.X_train]
        # print("distances=", distances)

        # get k nearest samples, labels
        # obitemos a amostra mais procima, e obiter os label
        # matriz das distancias mais procimas
        k_indices = np.argsort(distances)[:self.k]
        # rotulos mais procimos
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # fazemos uma votacao de maturidade, para obiter o rotulo de class label mais comum
        most_common = Counter(k_nearest_labels).most_common(1)
        # metodo= def most_common()
        return most_common[0][0]
