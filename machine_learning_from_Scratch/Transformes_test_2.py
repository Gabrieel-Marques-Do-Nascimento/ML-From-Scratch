from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import numpy as np
import Transforner_test_dados as tf_dados

# Defina alguns pares de palavras em inglês-português
pairs = [
    ("house", "casa"),
    ("dog", "cachorro"),
    ("car", "carro")
]

embeddings_en = tf_dados.load_vectors(tf_dados.en)
embeddings_pt = tf_dados.load_vectors(tf_dados.pt)

# Obtenha os vetores correspondentes
X_en = np.array([embeddings_en[eng] for eng, pt in pairs if eng in embeddings_en and pt in embeddings_pt])
X_pt = np.array([embeddings_pt[pt] for eng, pt in pairs if eng in embeddings_en and pt in embeddings_pt])

# Normalize os vetores para evitar escalas diferentes entre os dois idiomas
X_en_norm = normalize(X_en)
X_pt_norm = normalize(X_pt)

# Use SVD para encontrar a transformação linear
svd = TruncatedSVD(n_components=300)
svd.fit(X_en_norm.T @ X_pt_norm)

# Transformação linear
W = svd.components_.T


# Função para traduzir palavras
# Lista de palavras comuns em português para restringir a busca
common_words_pt = ["casa", "cachorro", "carro"] #, "mesa", "porta", "janela", "computador", "telefone", "livro"]

def translate(word, embeddings_en, embeddings_pt, W, subset_vocab=None):
    if word not in embeddings_en:
        return "Palavra não encontrada no vocabulário."

    vec_en = embeddings_en[word]
    vec_pt = vec_en @ W  # Aplicar a transformação

    # Usar um subconjunto do vocabulário se fornecido, senão usar todas as palavras
    if subset_vocab:
        search_vocab = {pt: embeddings_pt[pt] for pt in subset_vocab if pt in embeddings_pt}
    else:
        search_vocab = embeddings_pt

    # Encontrar a palavra mais próxima no espaço em português
    closest_word = min(search_vocab, key=lambda pt: np.linalg.norm(search_vocab[pt] - vec_pt))
    return closest_word

# Testar a tradução com vocabulário limitado
print(translate("house", embeddings_en, embeddings_pt, W, common_words_pt))
