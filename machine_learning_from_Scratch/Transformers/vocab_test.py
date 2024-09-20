# from torchtext.data import Field, BucketIterator
# from torchtext.datasets import Multi30k
# import spacy


# def tokenizer_eng(text):
#     return [tok.text for tok in space_eng.tokenizer(text)]


# def tokenizer_pt(text):
#     return [tok.text for tok in space_pt.tokenizer(text)]


# space_eng = spacy.load('en')
# space_pt = spacy.load('pt')
# english = Field(sequential=True,
#                 use_vocab=True,
#                 tokenize=tokenizer_eng,
#                 lower=True)
# portugues = Field(sequential=True,
#                   use_vocab=True,
#                   tokenize=tokenizer_pt,
#                   lower=True)


# train_data, valid_data, test_data = Multi30k.splits(
#     exts=('.pt', '.en'),
#     fields=(portugues, english))


# english.build_vocab(train_data, max_size=10000, min_freq=2)
# portugues.build_vocab(train_data, max_size=10000, min_freq=2)

# train_iterador, valid_iterador, test_iterador = BucketIterator.splits(
#         (train_data, valid_data, test_data), batch_size=64,
#         device='cpu')

# for batch in train_iterador:
#     print(batch)

import spacy

pt = spacy.load('pt_core_news_sm')
