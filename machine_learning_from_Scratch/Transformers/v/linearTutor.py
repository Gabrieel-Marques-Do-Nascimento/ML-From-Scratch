"""
`O que é nn.Linear?`
O nn.Linear no PyTorch faz uma multiplicação de uma matriz pelos seus dados d
e entrada. Essa operação transforma os seus dados de entrada,
mudando eles de uma dimensão (tamanho) para outra.
Vamos imaginar que você tem um vetor (ou lista de números) de tamanho 3, como:
>>> x = [1, 2, 3]

E você quer transformar esse vetor em outro de mesmo tamanho, usando uma
matriz de multiplicação. O nn.Linear cria essa matriz para você
automaticamente e faz a multiplicação para gerar a saída.

`O que significa self.head_dim?`
self.head_dim é o tamanho do seu vetor de entrada e também o tamanho da saída.
Se self.head_dim for 3, significa que a camada vai pegar um vetor de tamanho 3
e devolver um vetor de tamanho 3, aplicando uma transformação linear.
`O que o código faz?`
Ele pega um vetor de tamanho self.head_dim.
Ele multiplica esse vetor por uma matriz de pesos (que o nn.Linear cria).
Ele gera um novo vetor transformado, também de tamanho self.head_dim.
Como você está usando bias=False, ele não vai somar nenhum valor extra ao
final, apenas vai fazer a multiplicação.

`Exemplo mais visual`:

Se self.head_dim = 2, imagine que o vetor de entrada seja:
>>>     [ 1 ]
>>> x = [ 2 ]
`x` e uma matrix 2X1

O nn.Linear vai multiplicar isso por uma matriz de pesos
>>>      [ 0.5 -1 ]
>>> W =  [ 1    2 ]
`W` e uma matrix 2X2
O resultado vai ser um novo vetor:

>>>              [ 0.5 -1 ]   [1]    [0.5 -2]    [-1.5]
>>> y = W * x =  [ 1    2 ] * [2] =  [ 1 + 4] =  [  5 ]

`Fazendo a multiplicação`:
A multiplicação de matriz segue a regra de multiplicação de cada linha da
matriz pelos elementos do vetor.

Primeira linha de 𝑊 multiplicada por x:
>>> ( 0.5 × 1 )+( −1 × 2 ) = 0.5− 2 = −1.5

Segunda linha de 𝑊 multiplicada por x:
>>> ( 1 × 1 ) + ( 2 × 2 ) = 1 + 4 = 5
Assim, nn.Linear pega o vetor original e transforma ele em outro vetor.
Resultado
O vetor y resultante será:
>>>      [-1.5]
>>> y =  [  5 ]

`saida esperada:

>>>  tensor([ 1.2143, -0.5317], grad_fn=<SqueezeBackward4>)

`Isso faz mais sentido agora?
"""
# from linearTutor import embed_size as emb

from torch import nn
import torch


embed_size = 256
heads = 8
head_dim = embed_size // heads  # 32

head_dim = 3
keys = nn.Linear(head_dim, head_dim, bias=False)


# Criando uma camada Linear: vai transformar um vetor de tamanho head_dim para
# outro de mesmo tamanho
linear_layer = nn.Linear(head_dim, head_dim, bias=False)

# Exemplo de vetor de entrada (tamanho head_dim = 3)
input_vector = torch.tensor([1.0, 2.0, 3.0])
""" >>> "return" tensor([1., 2., 3.])  """

# Fazendo a transformação linear (multiplicação de matriz)
output_vector = linear_layer(input_vector)
""">>> "return" tensor([ 0.0141, -0.2811, -0.4417], grad_fn=<SqueezeBackward4>)
"""

# Mostrando o vetor de entrada e o resultado da multiplicação (vetor de saída)

# print("Vetor de entrada:", input_vector)
# print("Vetor de saída (após nn.Linear):", output_vector)
print("Pesos da matriz W:", linear_layer.weight)
# Pesos da matriz W: Parameter containing:
# tensor([[ 0.0954, -0.3409,  0.2750],
#         [-0.1823,  0.0496,  0.4370],
#         [ 0.2369,  0.4463, -0.3980]], requires_grad=True)ary_


def linear(head_dim=3, input_vector=[1.0, 2.0, 3.0]):
    import numpy as np

    # Definindo o tamanho do vetor (head_dim)
    head_dim = 3

    # Criando um vetor de entrada
    input_vector = np.array(input_vector)  # [1.0, 2.0, 3.0]

    # Criando uma matriz de pesos aleatória de tamanho (head_dim x head_dim)
    W2 = np.random.rand(head_dim, head_dim)

    # Fazendo a multiplicação da matriz pelos dados de entrada
    output_vector2 = np.dot(W2, input_vector)

    # Mostrando o vetor de entrada, a matriz de pesos e o resultado da
    # multiplicação

    # print("Vetor de entrada:", input_vector)
    # print("Matriz de pesos W:\n", W)
    print("Vetor 2 de saída (após a multiplicação):", output_vector2)


# linear()
