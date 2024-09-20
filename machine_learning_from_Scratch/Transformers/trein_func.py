from torch.nn.utils.rnn
import torch


def train(model, iterador, optimizer, criterion, clip, device):
    """
    `funcao de treinamento`

    Args:
        model (_type_): _description_
        iterador (_type_): _description_
        optimizer (_type_): _description_
        criterion (_type_): _description_
        clip (_type_): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_
    """
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterador):
        src = batch.src.to(device)
        trg = batch.trg.to(device)

        # remove o ultimo token trg para ser entrada so decoder
        trg_input = trg[:, :-1]
        # forward pass
        optimizer.zero_grad()
        output = model(src, trg_input)
        # iguinora o token do inicio para o calculo da perda
        output = output.reshape(-1, output.shape[2])
        trg = trg[:, 1].reshape(-1)
        # calculo da perda
        loss = criterion(output, trg)
        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.steo()
        epoch_loss += loss.item
    return epoch_loss / len(iterador)

def epoch_time(start_time, end_time):
    """ retorna o valor inteiro  de minutos e segundos

    Args:
        start_time (_type_): _description_
        end_time (_type_): _description_

    Returns:
        _type_: _description_
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time -  (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def evaluate(model, iterador, criterion, device):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i , batch in enumerate(iterador):
            src =  batch.src.to(device)
            trg = batch.trg.to(device)

            trg_input = trg[:, :-1]
            output = model(src, trg_input)

            output = output.reshape(-1,  output.shape[2])

            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
        return epoch_loss / len(iterador)

if __name__ == "__main__":
    pass