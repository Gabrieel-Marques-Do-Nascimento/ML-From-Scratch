import Transforner_neural_networks  # noqa: F401
import numpy as np


def load_vectors(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        word_vectors = {}
        for i, line in enumerate(f):
            if i == 0:
# A primeira linha geralmente contém metadados, como o número de palavras e a dimensão dos vetores
                continue
            parts = line.rstrip().split(' ')
            word = parts[0]
            vector = np.array([float(v) for v in parts[1:]], dtype=np.float32)
            word_vectors[word] = vector
    return word_vectors



pt = 'C:/Users/GABRI/OneDrive/Área de Trabalho/__PROJETOS/ML-From-Scratch/dados/cc.pt.300.vec/cc.pt.300.vec'
en = 'C:/Users/GABRI/OneDrive/Área de Trabalho/__PROJETOS/ML-From-Scratch/dados/cc.en.300.vec/cc.en.300.vec'

if __name__ == '__main__':
    #
    # # Carregar embeddings em inglês
    # embeddings_en = load_vectors(
    #     'C:/Users/GABRI/OneDrive/Área de Trabalho/__PROJETOS/ML-From-Scratch/dados/cc.en.300.vec/cc.en.300.vec')

    # Carregar embeddings em português
    embeddings_pt = load_vectors(
        '/dados/cc.pt.300.vec/cc.pt.300.vec')

    # Verificando os primeiros vetores carregados
    print(list(embeddings_pt.items())[:5])  # Mostra as 5 primeiras palavras em inglês e seus vetores

#
# def load_bin_vectors(filename, vec_size):
#     word_vectors = {}
#
#     with open(filename, 'rb') as f:
#         while True:
#             # Tentativa de leitura de uma palavra byte por byte
#             word = []
#             while True:
#                 char = f.read(1)
#                 if not char:  # Fim do arquivo
#                     return word_vectors
#                 if char == b' ':
#                     word = b''.join(word).decode('latin-1')  # Decodifica com 'latin-1'
#                     break
#                 word.append(char)
#
#             # Tentativa de ler o vetor associado (300 floats de 32 bits)
#             vector = f.read(vec_size * 4)
#             if not vector:
#                 break
#             vector = np.frombuffer(vector, dtype=np.float32)
#
#             word_vectors[word] = vector
#
#
# # Definir os parâmetros do arquivo
# vocab_size = 2000000  # Exemplo: tamanho do vocabulário
# vec_size = 300  # Dimensão dos vetores
#
# # Carregar embeddings binários
# embeddings_en_bin = load_bin_vectors(
#     'C:/Users/GABRI/OneDrive/Área de Trabalho/__PROJETOS/ML-From-Scratch/dados/cc.en.300.bin/cc.en.300.bin', vec_size)
#
# # Verificar os primeiros vetores
# print(list(embeddings_en_bin.items())[:5])












