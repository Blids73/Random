#######################################################################################################################
#    IMPORTING MODULE

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from torch.utils.tensorboard import SummaryWriter

#######################################################################################################################

# CREATING MODEL

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Linear(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):


        embedding = self.dropout(self.embedding(x))


        outputs, (hidden, cell) = self.rnn(embedding)


        return hidden, cell


class Decoder(nn.Module):
    def __init__(
        self, input_size, embedding_size, hidden_size, output_size, num_layers, p
    ):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Linear(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):

        #x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))


        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))


        predictions = self.fc(outputs)

        #predictions = predictions.squeeze(0)

        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        #batch_size = source.shape[0]#[1]
        target_len = target.shape[0]
        #target_vocab_size = 1 #len(english.vocab)

        #outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        outputs = torch.zeros(target_len).to(device)

        #AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

        hidden, cell = self.encoder(source)


        x = target[0]

        for t in range(1, target_len):
            # Use previous hidden, cell as context from encoder at start
            output, hidden, cell = self.decoder(x, hidden, cell)

            # Store next output prediction
            outputs[t] = output


            best_guess = output #.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs

#######################################################################################################################

#       HYPERPARAMETERS



# Training hyperparameters
num_epochs = 100
learning_rate = 0.001
batch_size = 1 #64 I THINK THIS IS PADDING AAAAAAAAAAAAAAAAAAAAAAA

# Model hyperparameters
load_model = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size_encoder = 7
input_size_decoder = 7
output_size = 7
encoder_embedding_size = 7
decoder_embedding_size = 7
hidden_size = 1024
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5


writer = SummaryWriter(f"runs/loss_plot")
step = 0

#######################################################################################################################

#           CONSTRUCT MODEL

encoder_net = Encoder(
    input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout
).to(device)

decoder_net = Decoder(
    input_size_decoder,
    decoder_embedding_size,
    hidden_size,
    output_size,
    num_layers,
    dec_dropout,
).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)

#######################################################################################################################

#           DEFINE ERROR LOSS

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

#######################################################################################################################

#           LOADING AND SAVING MODEL

def save_checkpoint( state, filename= 'my_checkpoint.pth.tar'):
    print('=> Saving checkpoint')
    torch.save(state, filename)

def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

#######################################################################################################################

#               DATA
lista_input =  [1, 7, 0, 6, 2, 5, 6]
lista_target = [2,6,1,0,4,7,0,3,6,5]
arr_inp = torch.from_numpy(np.array(lista_input))
arr_trg = torch.from_numpy(np.array(lista_target))
train_iterator = [[arr_inp,arr_trg]]


#######################################################################################################################

#               TRAINING


for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    # save_checkpoint(checkpoint)

    model.train()

    for batch_idx, element in enumerate(train_iterator):

        inp_data = element[0].to(device)
        target = element[1].to(device)

        # Forward prop
        output = model(inp_data, target)


        output = output.reshape(-1)#, output.shape[2])
        target = target.reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)


        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)


        optimizer.step()


        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1

print('finished')