"""
dados do ingles e do Portugues para o Transformers
"""
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k
import spacy




# # Certifique-se de que o modelo do SpaCy está carregado
# nlp = spacy.load("en_core_web_sm")

# # Função de tokenização com SpaCy


# def spacy_tokenizer(text):
#     return [token.text for token in nlp(text)]


# SRC = Field(tokenize=spacy_tokenizer, lower=True,
#             init_token="<sos>",
#             eos_token="<eos>")
# """
# cria campos de texto para tokenizar as frases em ingles e portugues
# """
# TRG = Field(tokenize=spacy_tokenizer, lower=True,
#             init_token="<sos>",
#             eos_token="<eos>")
# """
# cria campos de texto para tokenizar as frases em ingles e portugues
# """


# #
# train_data, valid_data, test_data = Multi30k.splits(
#     exts=(".en",
#           ".pt"),
#     fields=(SRC, TRG), root='data')
# """
# baixa e carregar o dataset Mult30k
# """

# # Constroi o vocabulario com base na frequencia
# SRC.build_vocab(train_data, max_size=10000,
#                 min_freq=2)

# TRG.build_vocab(train_data, max_size=10000,
#                 min_freq=2)
