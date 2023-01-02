import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import PackedSequence
import random
import math

SOS_TOKEN = 0
EOS_TOKEN = 29
class pBLSTM(torch.nn.Module):

    '''
    Pyramidal BiLSTM
    Read the write up/paper and understand the concepts and then write your implementation here.

    At each step,
    1. Pad your input if it is packed (Unpack it)
    2. Reduce the input length dimension by concatenating feature dimension
        (Tip: Write down the shapes and understand)
        (i) How should  you deal with odd/even length input? 
        (ii) How should you deal with input length array (x_lens) after truncating the input?
    3. Pack your input
    4. Pass it into LSTM layer

    To make our implementation modular, we pass 1 layer at a time.
    '''
    
    def __init__(self, input_size, hidden_size, dropout):
        super(pBLSTM, self).__init__()
        # TODO: Initialize a single layer bidirectional LSTM with the given input_size and hidden_size
        self.blstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True,
                            dropout=dropout, bidirectional=True)

    def forward(self, x, x_lens): # x_packed is a PackedSequence

        # TODO: Pad Packed Sequence
        if isinstance(x, PackedSequence):
            x, x_lens = pad_packed_sequence(x, batch_first=True, total_length=max(x_lens.cpu()))
        
        # Call self.trunc_reshape() which downsamples the time steps of x and increases the feature dimensions as mentioned above
        # self.trunc_reshape will return 2 outputs. What are they? Think about what quantites are changing.
        x, x_lens = self.trunc_reshape(x, x_lens)
        # TODO: Pack Padded Sequence. What output(s) would you get?
        x_packed = pack_padded_sequence(x, x_lens.cpu(), batch_first=True, enforce_sorted=False)
        # TODO: Pass the sequence through bLSTM
        x, (hidden, cell) = self.blstm(x_packed)

        # What do you return?
        return x, x_lens

    def trunc_reshape(self, x, x_lens): 
        # TODO: If you have odd number of timesteps, how can you handle it? (Hint: You can exclude them)
        if x.shape[1]%2 != 0:
            x = x[:, :-1, :]
            x_lens = x_lens - 1
        # TODO: Reshape x. When reshaping x, you have to reduce number of timesteps by a downsampling factor while increasing number of features by the same factor
        x = x.reshape(x.shape[0], x.shape[1]//2, 2*x.shape[2])
        # TODO: Reduce lengths by the same downsampling factor
        x_lens = torch.div(x_lens, 2, rounding_mode='floor')

        return x, x_lens

# Encoder
class Listener(torch.nn.Module):
    '''
    The Encoder takes utterances as inputs and returns latent feature representations
    '''
    def __init__(self, input_size, encoder_hidden_size=256, dropout=0):
        super(Listener, self).__init__()
        # The first LSTM at the very bottom
        self.base_lstm = nn.LSTM(input_size, 256, batch_first=True, 
                                 dropout=dropout, bidirectional=True)#TODO: Fill this up

        # self.pBLSTMs = nn.Sequential() # How many pBLSTMs are required?
            # TODO: Fill this up with pBLSTMs - What should the input_size be? 
            # Hint: You are downsampling timesteps by a factor of 2, 
            # upsampling features by a factor of 2 and the LSTM is bidirectional)
            # Optional: Dropout/Locked Dropout after each pBLSTM (Not needed for early submission)
        self.pBLSTMs = nn.ModuleList([pBLSTM(1024, encoder_hidden_size, dropout) for i in range(3)])
         
    def forward(self, x, x_lens):
        # Where are x and x_lens coming from? The dataloader
        
        # TODO: Pack Padded Sequence
        x_packed = pack_padded_sequence(x, x_lens.cpu(), batch_first=True, enforce_sorted=False)
        # TODO: Pass it through the first LSTM layer (no truncation)
        x_packed, (hidden, cell) = self.base_lstm(x_packed)
        # TODO: Pass Sequence through the pyramidal Bi-LSTM layer
        encoder_outputs, encoder_lens = x_packed, x_lens
        for f in self.pBLSTMs:
            encoder_outputs, encoder_lens = f(encoder_outputs, encoder_lens)
        # TODO: Pad Packed Sequence
        encoder_outputs, encoder_lens = pad_packed_sequence(encoder_outputs, batch_first=True, total_length=max(encoder_lens.cpu()))

        # Remember the number of output(s) each function returns
        return encoder_outputs, encoder_lens

class Attention(torch.nn.Module):
    '''
    Attention is calculated using the key, value (from encoder hidden states) and query from decoder.
    Here are different ways to compute attention and context:

    After obtaining the raw weights, compute and return attention weights and context as follows.:

    masked_raw_weights  = mask(raw_weights) # mask out padded elements with big negative number (e.g. -1e9 or -inf in FP16)
    attention           = softmax(masked_raw_weights)
    context             = bmm(attention, value)
    
    At the end, you can pass context through a linear layer too.

    '''
    
    def __init__(self, encoder_hidden_size, decoder_output_size, projection_size, device):
        super(Attention, self).__init__()
        self.device = device
        self.projection_size = projection_size

        self.key_projection     = nn.Linear(encoder_hidden_size*2, projection_size)# TODO: Define an nn.Linear layer which projects the encoder_hidden_state to keys
        self.value_projection   = nn.Linear(encoder_hidden_size*2, projection_size)# TODO: Define an nn.Linear layer which projects the encoder_hidden_state to value
        self.query_projection   = nn.Linear(decoder_output_size, projection_size)# TODO: Define an nn.Linear layer which projects the decoder_output_state to query
        # Optional : Define an nn.Linear layer which projects the context vector

        self.softmax            = nn.Softmax(dim=1)# TODO: Define a softmax layer. Think about the dimension which you need to apply 
        # Tip: What is the shape of energy? And what are those?

    # As you know, in the attention mechanism, the key, value and mask are calculated only once.
    # This function is used to calculate them and set them to self
    def set_key_value_mask(self, encoder_outputs, encoder_lens):
    
        batch, encoder_max_seq_len, _ = encoder_outputs.shape

        self.key      = self.key_projection(encoder_outputs)# TODO: Project encoder_outputs using key_projection to get keys
        self.value    = self.value_projection(encoder_outputs)# TODO: Project encoder_outputs using value_projection to get values

        # encoder_max_seq_len is of shape (batch_size, ) which consists of the lengths encoder output sequences in that batch
        # The raw_weights are of shape (batch_size, timesteps)

        # TODO: To remove the influence of padding in the raw_weights, we want to create a boolean mask of shape (batch_size, timesteps) 
        # The mask is False for all indicies before padding begins, True for all indices after.
        # TODO You want to use a comparison between encoder_max_seq_len and encoder_lens to create this mask. 
        self.padding_mask     =  torch.ones([batch, encoder_max_seq_len]).to(self.device)

        for i in range(encoder_lens.shape[0]):
            if encoder_lens[i] < encoder_max_seq_len:
                self.padding_mask[i, encoder_lens[i]:] = 0

        # (Hint: Broadcasting gives you a one liner)
        
    def forward(self, decoder_output_embedding):
        # key   : (batch_size, timesteps, projection_size)
        # value : (batch_size, timesteps, projection_size)
        # query : (batch_size, projection_size)

        self.query         = self.query_projection(decoder_output_embedding)# TODO: Project the query using query_projection

        # Hint: Take a look at torch.bmm for the products below 

        raw_weights        = torch.bmm(self.query.unsqueeze(1), torch.transpose(self.key, 1, 2)).squeeze(dim=1)# TODO: Calculate raw_weights which is the product of query and key, and is of shape (batch_size, timesteps)
        raw_weights        = raw_weights / math.sqrt(self.query.shape[-1])# TODO: Divide raw_weights by the square root of the projection size   
        masked_raw_weights = raw_weights.masked_fill_(self.padding_mask==0, -1e4)# TODO: Mask the raw_weights with self.padding_mask. 
        # Take a look at pytorch's masked_fill_ function (You want the fill value to be a big negative number for the softmax to make it close to 0)

        attention_weights  = self.softmax(masked_raw_weights)# TODO: Calculate the attention weights, which is the softmax of raw_weights
        context            = torch.bmm(attention_weights.unsqueeze(1), self.value).squeeze(dim=1)# TODO: Calculate the context - it is a product between attention_weights and value

        # Hint: You might need to use squeeze/unsqueeze to make sure that your operations work with bmm

        return context, attention_weights # Return the context, attention_weights

# Decoder
class Speller(torch.nn.Module):

    def __init__(self, embed_size, decoder_hidden_size, decoder_output_size, vocab_size, device, attention_module=None):
        super().__init__()
        self.device             = device

        # TODO: Initialize the Embedding Layer (Use the nn.Embedding Layer from torch), make sure you set the correct padding_idx  
        self.embedding          = nn.Embedding(num_embeddings=vocab_size, 
                                               embedding_dim=embed_size, # embedding vector dimension(임의로 정할 수 있음 128등)
                                               padding_idx=EOS_TOKEN) # <pad> index


        self.lstm_cells         = torch.nn.Sequential(
                                # Create Two LSTM Cells as per LAS Architecture
                                # What should the input_size of the first LSTM Cell? 
                                # Hint: It takes in a combination of the character embedding and context from attention
                                nn.LSTMCell(embed_size+attention_module.projection_size, decoder_hidden_size),
                                nn.LSTMCell(decoder_hidden_size, decoder_output_size)
                                )
    
                                # We are using LSTMCells because process individual time steps inputs and not the whole sequence.
                                # Think why we need this in terms of the query

        # TODO: Initialize the classification layer to generate your probability distribution over all characters
        self.char_prob          = nn.Linear(decoder_output_size*2, vocab_size) 

        self.char_prob.weight   = self.embedding.weight # Weight tying

        self.attention          = attention_module

    
    def forward(self, encoder_outputs, encoder_lens, y = None, tf_rate = 1): 

        '''
        Args: 
            embedding: Attention embeddings 
            hidden_list: List of Hidden States for the LSTM Cells
        ''' 

        batch_size, encoder_max_seq_len, _ = encoder_outputs.shape

        if self.training:
            timesteps     = y.shape[1] # The number of timesteps is the sequence of length of your transcript during training
            label_embed   = self.embedding(y) # Embeddings of the transcript, when we want to use teacher forcing
        else:
            timesteps     = 600 # 600 is a design choice that we recommend, however you are free to experiment.
        

        # INITS
        predictions     = []

        # Initialize the first character input to your decoder, SOS
        char            = torch.full((batch_size,), fill_value=SOS_TOKEN, dtype=torch.long).to(self.device) 

        # Initialize a list to keep track of LSTM Cell Hidden and Cell Memory States, to None
        hidden_states   = [None]*len(self.lstm_cells) 

        attention_plot          = []
        attention_weights       = torch.zeros(batch_size, encoder_max_seq_len).to(self.device) # Attention Weights are zero if not using Attend Module

        # Set Attention Key, Value, Padding Mask just once
        if self.attention != None:
            self.attention.set_key_value_mask(encoder_outputs, encoder_lens)

        context = self.attention.key[:,0,:]# TODO: Initialize context (You have a few choices, refer to the writeup )

        for t in range(timesteps):
            
            char_embed = self.embedding(char) #TODO: Generate the embedding for the character at timestep t

            if self.training and t > 0:
                # TODO: We want to decide which embedding to use as input for the decoder during training
                # We can use the embedding of the transcript character or the embedding of decoded/predicted character, from the previous timestep 
                # Using the embedding of the transcript character is teacher forcing, it is very important for faster convergence
                # Use a comparison between a random probability and your teacher forcing rate, to decide which embedding to use
                
                if random.random() <= tf_rate:
                    char_embed = label_embed[:,t-1,:]
      
            # TODO: What do we want to concatenate as input to the decoder? (Use torch.cat)
            decoder_input_embedding = torch.cat((char_embed, context), dim=1)

            
            # Loop over your lstm cells
            # Each lstm cell takes in an embedding 
            for i in range(len(self.lstm_cells)):
                # An LSTM Cell returns (h,c) -> h = hidden state, c = cell memory state
                # Using 2 LSTM Cells is akin to a 2 layer LSTM looped through t timesteps 
                # The second LSTM Cell takes in the output hidden state of the first LSTM Cell (from the current timestep) as Input, along with the hidden and cell states of the cell from the previous timestep
                hidden_states[i] = self.lstm_cells[i](decoder_input_embedding, hidden_states[i]) 
                decoder_input_embedding = hidden_states[i][0]

            # The output embedding from the decoder is the hidden state of the last LSTM Cell
            decoder_output_embedding = hidden_states[-1][0]

            # We compute attention from the output of the last LSTM Cell
            if self.attention != None:
                context, attention_weights = self.attention(decoder_output_embedding) # The returned query is the projected query

            attention_plot.append(attention_weights[0].detach().cpu())

            # TODO: Concatenate the projected query with context for the output embedding
            # Hint: How can you get the projected query from attention
            # If you are not using attention, what will you use instead of query?
            output_embedding     = torch.cat((decoder_output_embedding, context), dim=1)

            char_prob            = self.char_prob(output_embedding)
            
            # Append the character probability distribution to the list of predictions 
            predictions.append(char_prob)

            char = torch.argmax(char_prob, dim=1)# TODO: Get the predicted character for the next timestep from the probability distribution 
            # (Hint: Use Greedy Decoding for starters)

        attention_plot  = torch.stack(attention_plot, dim=1)# TODO: Stack list of attetion_plots 
        predictions     = torch.stack(predictions, dim=1)# TODO: Stack list of predictions 

        return predictions, attention_plot

# Sequence to Sequence Model
class LAS(torch.nn.Module):
    def __init__(self, input_size, encoder_hidden_size, 
                 vocab_size, embed_size,
                 decoder_hidden_size, decoder_output_size,
                 device,
                 projection_size=128):
        
        super(LAS, self).__init__()

        self.encoder        = Listener(input_size, encoder_hidden_size, 0) # TODO: Initialize Encoder
        attention_module    = Attention(encoder_hidden_size, decoder_output_size, projection_size, device)# TODO: Initialize Attention
        self.decoder        = Speller(embed_size, decoder_hidden_size, decoder_output_size, vocab_size, device, attention_module)# TODO: Initialize Decoder, make sure you pass the attention module 

    def forward(self, x, x_lens, y=None, tf_rate=1):

        encoder_outputs, encoder_lens = self.encoder(x, x_lens) # from Listener
        predictions, attention_plot = self.decoder(encoder_outputs, encoder_lens, y, tf_rate)
        
        return predictions, attention_plot