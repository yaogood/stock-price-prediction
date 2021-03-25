import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderLSTM(nn.Module):
    def __init__(self, num_features, input_size, hidden_size, n_layers=1, dropout=0.2):
        super(EncoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = self.hidden_size
        self.n_layers = n_layers
        
        self.encoder_in_cnn = nn.Conv1d(in_channels=num_features, out_channels=input_size, kernel_size=1)
        self.drop_out = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, dropout=dropout, batch_first=True)

    def forward(self, source, hidden):
        # input shape: [b, seq_len, num_features]
        batch_size = source.size(0)
        encoder_seq_len = source.size(1)
        num_features = source.size(2)

        src = self.encoder_in_cnn(source.view(-1, num_features, 1))
        src = src.squeeze().view(batch_size, encoder_seq_len, self.input_size)
        src = self.drop_out(src)

        outputs, hidden = self.lstm(src, hidden)

        return outputs, hidden

    def init_hidden(self, batch_size=1, device='cpu'):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device))


class LuongDecoderLSTM(nn.Module):
    def __init__(self, num_features, input_size, hidden_size, n_layers=1, dropout=0.2):
        super(LuongDecoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.attention = Attention(hidden_size,"concat")

        self.embedding_sos = nn.Embedding(1, input_size)
        
        self.decoder_in_cnn = nn.Conv1d(in_channels=num_features, out_channels=input_size, kernel_size=1)
        self.drop_out = nn.Dropout(dropout)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, n_layers, dropout=dropout, batch_first=True)
        self.decoder_out_cnn = nn.Conv1d(in_channels=self.hidden_size*2, out_channels=1, kernel_size=1)
        
    def forward(self, target, hidden, encoder_outputs):
        batch_size = target.size(0)
        decoder_seq_len = target.size(1)
        num_features = target.size(2)

        sos = self.embedding_sos(torch.tensor(np.zeros([batch_size], dtype=int))) # [b, input_size]
        sos = torch.unsqueeze(sos, 1) # [b, 1, input_size]
        tgt = self.decoder_in_cnn(target.view(-1, num_features, 1))      # [b*seq_len, input_size, 1]
        tgt = tgt.squeeze().view(batch_size, decoder_seq_len, self.input_size) # [b, seq_len, input_size]
        tgt = torch.cat(sos, tgt, dim=1)
        tgt = self.drop_out(tgt) # [b, seq_len+1, input_size]

        # Passing previous output word (embedded) and hidden state into LSTM cell
        lstm_outs, hidden = self.lstm(tgt, hidden)

        outputs = self.decoder_out_cnn()

        return outputs, hidden,
        
        # # Calculating Alignment Scores - see Attention class for the forward pass function
        # alignment_scores = self.attention(lstm_out, encoder_outputs)
        # # Softmaxing alignment scores to obtain Attention weights
        # attn_weights = F.softmax(alignment_scores.view(1,-1), dim=1)
        # # Multiplying Attention weights with encoder outputs to get context vector
        # context_vector = torch.bmm(attn_weights.unsqueeze(0),encoder_outputs)
        
        # # Concatenating output from LSTM with context vector
        # output = torch.cat((lstm_out, context_vector), -1)
        # # Pass concatenated vector through Linear layer acting as a Classifier
        # output = F.log_softmax(self.classifier(output[0]), dim=1)
        
        # return output, hidden, attn_weights
    

class Attention(nn.Module):
    def __init__(self, hidden_size, method="dot"):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        
        # Defining the layers/weights required depending on alignment scoring method
        if method == "general":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        
        elif method == "concat":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
            self.weight = nn.Parameter(torch.FloatTensor(1, hidden_size))
    
    def forward(self, decoder_hidden, encoder_outputs):
        if self.method == "dot":
        # For the dot scoring method, no weights or linear layers are involved
            return encoder_outputs.bmm(decoder_hidden.view(1,-1,1)).squeeze(-1)      
        elif self.method == "general":
        # For general scoring, decoder hidden state is passed through linear layers to introduce a weight matrix
            out = self.fc(decoder_hidden)
            return encoder_outputs.bmm(out.view(1,-1,1)).squeeze(-1)   
        elif self.method == "concat":
        # For concat scoring, decoder hidden state and encoder outputs are concatenated first
            out = torch.tanh(self.fc(decoder_hidden+encoder_outputs))
            return out.bmm(self.weight.unsqueeze(-1)).squeeze(-1)



class MyTransformer(nn.Module):
    def __init__(self, num_features, encoder_nlayers, encoder_nhead, d_model, nhid, decoder_nlayers, decoder_nhead, dropout):
        super(MyTransformer, self).__init__()
        self.d_model = d_model
        self.encoder_pos_encoding = PositionalEncoder(d_model)
        self.decoder_pos_encoding = PositionalEncoder(d_model)

        self.encoder_in_cnn = nn.Conv1d(in_channels=num_features, out_channels=d_model, kernel_size=1)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, encoder_nhead, nhid, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, encoder_nlayers)

        self.decoder_in_cnn = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=1)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model, decoder_nhead, nhid, dropout=dropout)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, decoder_nlayers) 
        self.decoder_out_cnn = nn.Conv1d(in_channels=d_model, out_channels=1, kernel_size=1)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, source, target, tgt_mask):
        '''
        shapes:
            src: [seq_len, b_size, feature_numbers], [30, b, 19] -> [30, b, 512]
            tgt: [tgt_seq_len, b_size, feature_numbers], [5, b, 1] -> [5, b, 512]
            tgt_mask: [tgt_seq_len, tgt_seq_len], [5, 5]

            return: [tgt_seq_len, b_size, 1], [5, b, 512] -> [5, b, 1]
        '''
        encoder_seq_len = source.size(0)
        batch_size = source.size(1)
        num_features = source.size(2)
        src = self.encoder_in_cnn(source.view(-1, num_features, 1))
        src = src.squeeze().view(encoder_seq_len, batch_size, self.d_model)

        src = self.encoder_pos_encoding(src)
        memory = self.encoder(src)

        decoder_seq_len = target.size(0)
        num_tgt = target.size(2)
        tgt = self.decoder_in_cnn(target.view(-1, num_tgt, 1))
        tgt = tgt.squeeze().view(decoder_seq_len, batch_size, self.d_model)

        tgt = self.decoder_pos_encoding(tgt)
        out = self.decoder(tgt, memory, tgt_mask) # out shape: [5, b, 512]

        out = self.decoder_out_cnn(out.view(-1, self.d_model, 1)) # [5*b, 1, 1]
        
        return out


class PositionalEncoder(nn.Module):
     # create constant 'pe' matrix with values dependant on pos and i
    def __init__(self, d_model, max_seq_len = 30, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model):
                if i % 2 == 0:
                    pe[pos, i] = math.sin(pos / (10000 ** (2*i/d_model)))
                else:
                    pe[pos, 1] = math.cos(pos / (10000 ** (2*i/d_model)))
                
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x): # [seq_len, b, d_model]
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# import matplotlib.pyplot as plt
# model = MyTransformer(1, 1, 16, 16, 1, 1, 0.5)
# mask = model.generate_square_subsequent_mask(5)

# plt.figure(figsize=(5,5))
# plt.imshow(mask)
# plt.show()

# fig, ax = plt.subplots()
# im = ax.imshow(mask, cmap=plt.get_cmap('hot'), interpolation='nearest',
#                vmin=0, vmax=1)
# fig.colorbar(im)
# plt.show()