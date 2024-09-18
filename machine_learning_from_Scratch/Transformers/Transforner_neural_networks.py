"""
na imagem, do lado esquerdo temos o `encoder`, e o decoder no lado `direito`.

.. ENCODER::
`codificador`: de baixo temos entradas digamos que algum texto fonte para tradução, vamos criar alguns `embeddings`, que sao enviados para o bloco, o bloco e enviado para um `mult-head-attention` que e um bloco menor dentro do bloco transformador, o nucleo essencial do transformador ,sera enviado para o `mult-head-attention` 3 entradas diferentes, chamados `chaves` e `consultas`, por diante paca por um normalizador depois por `Feed-forwaed`, e vai para outro `normalizador` as setas na imagem sao conexoes entre as partes do bloco.

.. DECODER::
`decodificador` a saida do `ENCODER` e enviado como entrada (valores,mm
chaves e consultas) para o `mult-head-attention`
vamos chamalo de bloco decodificador, mm

.. description::
o `attention` vai dividir a entrada pelo numero de cabeças(`heads`)

`exemp`:  256 / 8 == 32 dimensões
que serão enviados em 3 através de camadas lineares, sera enviado a entrada
dividida a Saida vai para o produto escalar


.. CODIFICANDO::
vamos começar pela parte mais complicada que e `auto atenção`
"""

import torch
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
`Inicializa o estado interno do módulo, compartilhado por nn.Module e 
ScriptModule.`

    Args:
        Module (_type_): _description_
    

Descrição detalhada (opcional):
Esta classe faz explicação do que a classe faz.

Attributes:
    embed_size (int): tamanho da incorporação, dimensão da encorporação.
    heads (int): quantidade de partes que a incorporação sera dividida.
    head_dim (int): tamanho de cada parte da incorporação, dimensão de cada parte
    """
    def __init__(self, embed_size, heads):
        """_requer o tamanho da incorporação `embed_size` e também as cabeças `heads` que representa a quantidade de partes que a incorporação sera dividida
  
        Args:
            embed_size (int): tamanho da incorporação
            heads (int): quantidade de partes que a incorporação sera dividida
        """
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        # afirmar 
        assert (self.head_dim * heads ==
                embed_size), "Embed size neds  to be div by heats"
        # O tamanho do embed precisa ser dividido pelos heats 
       # pega a dimencao do head vsi mapealo para a dimencao  head  bias como false
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # heads * head_dim tem que ser igual a embed_size
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        """ 
        recebe a query,chaves ,valores e mascara
        """
        #primeira coisa e obter o numero de exemplos de treinamento, 
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape(N, query_len, heads, heads_dim)
        # keys shape(N, key_len, heads, heads_dim)
        # energy shape(N,  heads, heads_dim, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))

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
        return out


class TransFormerBlock(Module):

    def __init__(self, embed_size, heads, dropout, forward_expansion):
        
        super(TransFormerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

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
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 src_pad_idx,
                 trg_pad_idx,
                 embed_size=256,
                 # embed_size (int, optional): representa o tamanho da entrada do user.
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
    device = torch.device("cpu")
    x = torch.tensor(
        [[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device

    )
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [
        1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10

    modal = Transformer(
        src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(
        device
    )
    out = modal(x, trg[:, :-1])
    print(out.shape)
