"""

"""
# torch e usado para calculos e de aprendizado de maquina profundo
import torch
# nn e usado para camadas de rede neurais'
from torch import nn


class Module(nn.Module):
    """
`docstring traduzida`: `nn.Module`

    Classe base para todos os módulos de rede neural.
Seus modelos também devem subclassificar esta classe.

Módulos também podem conter outros Módulos, permitindo aninhá-los em uma 
estrutura de árvore. Você pode atribuir os submódulos como atributos regulares:

 import torch.nn as nn\n
    import torch.nn.functional as F

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))


Submódulos atribuídos dessa forma serão registrados e terão seus parâmetros 
convertidos também quando você chamar, etc.

nota Conforme o exemplo acima, uma chamada __init__() para a classe pai deve 
ser feita antes da atribuição na filha.

ivar training : Boolean representa se este módulo está em modo de treinamento 
ou avaliação.
vartype training : bool
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class SelfAttention(Module):
    """
implementa o mecanismo de `atenção`, onde entradas sao processadas com valores,
chaves e consultas para calcular pesos de `atenção`

    Args:
        Module (_type_): `Inicializa o estado interno do módulo, compartilhado
        por nn.Module e ScriptModule.`

Attributes:
   embed_size (int): tamanho da incorporação, dimensão da encorporação.
   heads (int): quantidade de partes que a incorporação sera dividida.
   head_dim (int): divide o embedding em múltiplas partes (heads) para
   facilitar o processamento paralelo.

    """

    def __init__(self, embed_size, heads):
        """
requer o tamanho da incorporação `embed_size` e também as cabeças
`heads` que representa a quantidade de partes que a incorporação sera dividida

        Args:
            embed_size (int): tamanho da incorporação
            heads (int): quantidade de partes que a incorporação sera dividida
        """
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        # >>> divide o embedding em múltiplas partes (heads) para
        # >>> facilitar o processamento paralelo.
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads ==
                embed_size), "Embed size neds  to be div by heats"

        # `Camadas lineares`: sao aplicadas para calcular as chaves,
        #  valores e consultas

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        """
        `nn.Linear`: multiplicação de uma matriz pelos dados de entrada,
        seguida por uma soma de um vetor de viés (bias),
         a menos que você opte por desativá-lo
         
         explicação completa em `linearTutor.py`
         """
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[
            1], query.shape[1]

        # >>> split embedding into self.heads pieces
        # >>> reorganiza as entradas para separar o numero de cabeças(heads)
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        # >>> multiplicação de tenores: einsum calcula a energia entre as
        # >>> consultas e chaves.
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape(N, query_len, heads, heads_dim)
        # keys shape(N, key_len, heads, heads_dim)
        # energy shape(N,  heads, heads_dim, key_len)

        # >>> se uma mascara e fornecida, ela define certos valores como
        # >>> negativos, ignorando esses cálculos de atenção
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))
        # >>> softmax: Normaliza as pontuações para obter pesos de atenção
        #                                     embed_size =256 / 0.5
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # print(attention.shape)
        # print(values.shape)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
    # after shape: (N, value_len, heads, heads_dim) then flatten last two di

        out = self.fc_out(out)
        # >>> a atenção e multiplicada pelos valores e depois concatenada em um
        # >>> tensor final, que e processado pela camada final `fc_out`

        return out


class TransFormerBlock(Module):
    """
    .. `Bloco Transformer` ::
    implementa uma cama do `Transformer`, consistindo de atenção,
    normalização e um feed-forward network.

    Args:
        Module (_type_): _description_
    """

    def __init__(self, embed_size, heads, dropout, forward_expansion):

        super(TransFormerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        # >>> Normaliza as saidas para estabilidade
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        # >>> uma rede neural densa aplicada apos a atencao para
        # >>> processamento adicional
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(Module):
    """
:CODIFICADOR( ENCODER):
converte as sequencias de entrada em embeddings,
add informações de posição e aplica varias camadas Transformer.

:EMBEDDING: 
Associa palavras e posições em vetores de alta dimensão.

:CAMADAS:
aplicação repetida de bloco Transformer para maior processamento

    Args:
        Module (_type_): _description_
    """

    def __init__(
            self,
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_langth
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device

        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embendding = nn.Embedding(max_langth, embed_size)

        self.layers = nn.ModuleList(
            [
                TransFormerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)]  # 6 vezes
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(
            N, seq_length).to(self.device)

        out = self.dropout(self.word_embedding(
            x) + self.position_embendding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(Module):
    """
    :BLOCO DECODIFICADOR:
    implementa um bloco do decodificador, que também usa a atenção e camadas 
    Transformers para gerar a sequencia de saida

    Args:
        Module (_type_): _description_
    """

    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransFormerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(Module):
    """
    :DECODIFICADOR: 
    similar ao `ENCODER`, mas processa a sequencia de destino (target) para
    gerar a Saida prevista.

    :CAMADA FINAL (fc_out):
    a Saida do decodificador r transformada em previsões de probabilidade para
    cada palavra no vocabulário.

    Args:
        Module (_type_): _description_
    """

    def __init__(self,
                 trg_vocab_size,
                 embed_size,
                 num_layers,
                 heads,
                 forward_expansion,
                 dropout,
                 device,
                 max_length
                 ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [DecoderBlock(
                embed_size, heads, forward_expansion, dropout, device
            )
                for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(
            N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(
            x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)
        return out


class Transformer(Module):
    """
    :MODELO TRANSFORMER: junta o codificador (`encoder`) e o
    decodificador (`decoder`)

    Args:
        Module (_type_): _description_
    """

    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 src_pad_idx,
                 trg_pad_idx,
                 embed_size=256,
                 # embed_size (int, optional): representa o tamanho da
                 # entrada do user.
                 # Defaults to 256.
                 num_layers=6,
                 forward_expansion=4,
                 heads=8,
                 dropout=0,
                 device="cpu",  # 'cuda'
                 max_length=100
                 ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out


if __name__ == "__main__":  # ("cuda" if torch.cuda.is_available() else "cpu")
    # Vocabulario simples para simular o processo
    src_vocab = {
        "i": 1,
        "want": 2,
        "to": 3,
        "learn": 4,
        "english": 5,
        "how": 6,
        "aree": 7,
        "yiu": 8,
        "doing": 9,
    }
    trg_vocab = {
        'eu': 1,
        'quero': 2,
        'para': 3,
        'aprender': 4,
        'portugues': 5,
        'como': 6,
        'voce': 7,
        'esta': 8,
        'fazendo': 9

    }
    # +1 para o token de padding
    src_vocab_size = len(src_vocab) + 1
    trg_vocab_size = len(trg_vocab) + 1




import torch

# Vocabulário simples para simular o processo
src_vocab = {"i": 1, "want": 2, "to": 3, "learn": 4, "english": 5, "how": 6, "are": 7, "you": 8, "doing": 9}
trg_vocab = {"eu": 1, "quero": 2, "aprender": 3, "português": 4, "como": 5, "você": 6, "está": 7, "fazendo": 8}

src_vocab_size = len(src_vocab) + 1  # +1 para token de padding
trg_vocab_size = len(trg_vocab) + 1

# Mapeamento inverso do vocabulário para obter as palavras de volta a partir dos índices
index_to_word_trg = {v: k for k, v in trg_vocab.items()}

# Função para converter texto em tensores de índices
def text_to_tensor(text, vocab):
    return torch.tensor([[vocab[word] for word in text.split()]])

# Função para converter tensores de saída em texto
def tensor_to_text(tensor, index_to_word):
    return " ".join([index_to_word[idx.item()] for idx in tensor])

# Modelo Transformer criado anteriormente
modal = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx=0, trg_pad_idx=0).to('cpu')

# Texto para traduzir do inglês para o português
english_text = "i want to learn english"
src_tensor = text_to_tensor(english_text, src_vocab)

# Simula uma tradução (sem treinar de fato o modelo, apenas para fins de demonstração)
output_tensor = modal(src_tensor, src_tensor)  # Tradução fictícia

# Para a demo, vamos simular a saída do modelo
output_tensor = torch.tensor([[1, 2, 3, 4]])  # Simulando que o modelo retornou a tradução "eu quero aprender português"

# Converte a saída do modelo para texto
translated_text = tensor_to_text(output_tensor[0], index_to_word_trg)

# Exibe a tradução
print(f"Texto em inglês: {english_text}")
print(f"Tradução em português: {translated_text}")