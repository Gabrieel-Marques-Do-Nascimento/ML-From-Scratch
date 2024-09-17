f = """
na imagem, do lado esquerdo temos o `encoder`, e o decoder no lado `direito`.

.. ENCODER::
`codificador` : de baixo temos entradas digamos que algum texto fonte para traducao, vamos criar alguns `embeddings`, que sao enviados para o bloco, o bloco e enviado para um `mult-head-attention` que e um bloco menor dentro do bloco transformador, o nucleo essencial do transformador ,sera enviado para o `mult-head-attention` 3 entradas diferentes, chamados `chaves` e `consultas`, por diante paca por um normalizador depois por `Feed-forwaed`, e vai para outro `normalizador` as setas na imagem sao conexoes entre as partes do bloco.

.. DECODER::
`decodificador` a saida do `ENCODER` e enviado como entrada (valores,
 chaves e consultas) para o `mult-head-attention`
vamos chamalo de bloco decodificador, mm

.. description::
o `attention` vai dividir a entrada pelo numero de cabeças(`heads`)

`exemp`:  256 / 8 == 32 dimensões
que serão enviados em 3 através de camadas lineares, sera enviado a entrada
dividida a Saida vai para o produto escalar
"""

for c in range(len(f)):
    if c == 643:
        print(f[c:])
