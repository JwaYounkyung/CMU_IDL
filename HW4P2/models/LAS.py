import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import PackedSequence

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
        self.input_size = input_size
        self.hidden_size = hidden_size

        # TODO: Initialize a single layer bidirectional LSTM with the given input_size and hidden_size
        self.blstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True,
                            dropout=dropout, bidirectional=True)

    def forward(self, x, x_lens): # x_packed is a PackedSequence

        # TODO: Pad Packed Sequence
        if isinstance(x, PackedSequence):
            x, x_lens = pad_packed_sequence(x, batch_first=True, total_length=max(x_lens))
        
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
        x_lens = x_lens//2

        return x, x_lens

# Encoder
class Listener(torch.nn.Module):
    '''
    The Encoder takes utterances as inputs and returns latent feature representations
    '''
    def __init__(self, input_size, encoder_hidden_size=256, dropout=0):
        super(Listener, self).__init__()

        self.encoder_hidden_size = encoder_hidden_size
        # The first LSTM at the very bottom
        self.base_lstm = nn.LSTM(input_size, 256, batch_first=True, 
                                 dropout=dropout, bidirectional=True)#TODO: Fill this up

        # self.pBLSTMs = nn.Sequential() # How many pBLSTMs are required?
            # TODO: Fill this up with pBLSTMs - What should the input_size be? 
            # Hint: You are downsampling timesteps by a factor of 2, 
            # upsampling features by a factor of 2 and the LSTM is bidirectional)
            # Optional: Dropout/Locked Dropout after each pBLSTM (Not needed for early submission)
        self.pBLSTMs = nn.ModuleList([pBLSTM(1024, self.encoder_hidden_size, dropout) for i in range(3)])
        self.pBLSTM = pBLSTM(1024, self.encoder_hidden_size, dropout)
         
    def forward(self, x, x_lens):
        # Where are x and x_lens coming from? The dataloader
        
        # TODO: Pack Padded Sequence
        x_packed = pack_padded_sequence(x, x_lens.cpu(), batch_first=True, enforce_sorted=False)
        # TODO: Pass it through the first LSTM layer (no truncation)
        x_packed, (hidden, cell) = self.base_lstm(x_packed)
        # TODO: Pad Packed Sequence
        encoder_outputs, encoder_lens = pad_packed_sequence(x_packed, batch_first=True, total_length=x.shape[1])
        # TODO: Pass Sequence through the pyramidal Bi-LSTM layer
        for f in self.pBLSTMs:
            encoder_outputs, encoder_lens = f(encoder_outputs, encoder_lens)
        # nn.Sequential에 argument 추가하는 방법 알아내기

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
    
    def __init__(self, encoder_hidden_size, decoder_output_size, projection_size):
        super(Attention, self).__init__()

        self.key_projection     = 0# TODO: Define an nn.Linear layer which projects the encoder_hidden_state to keys
        self.value_projection   = 0# TODO: Define an nn.Linear layer which projects the encoder_hidden_state to value
        self.query_projection   = 0# TODO: Define an nn.Linear layer which projects the decoder_output_state to query
        # Optional : Define an nn.Linear layer which projects the context vector

        self.softmax            = 0# TODO: Define a softmax layer. Think about the dimension which you need to apply 
        # Tip: What is the shape of energy? And what are those?

    # As you know, in the attention mechanism, the key, value and mask are calculated only once.
    # This function is used to calculate them and set them to self
    def set_key_value_mask(self, encoder_outputs, encoder_lens):
    
        _, encoder_max_seq_len, _ = encoder_outputs.shape

        self.key      = 0# TODO: Project encoder_outputs using key_projection to get keys
        self.value    = 0# TODO: Project encoder_outputs using value_projection to get values

        # encoder_max_seq_len is of shape (batch_size, ) which consists of the lengths encoder output sequences in that batch
        # The raw_weights are of shape (batch_size, timesteps)

        # TODO: To remove the influence of padding in the raw_weights, we want to create a boolean mask of shape (batch_size, timesteps) 
        # The mask is False for all indicies before padding begins, True for all indices after.
        self.padding_mask     =  0# TODO: You want to use a comparison between encoder_max_seq_len and encoder_lens to create this mask. 
        # (Hint: Broadcasting gives you a one liner)
        
    def forward(self, decoder_output_embedding):
        # key   : (batch_size, timesteps, projection_size)
        # value : (batch_size, timesteps, projection_size)
        # query : (batch_size, projection_size)

        self.query         = 0# TODO: Project the query using query_projection

        # Hint: Take a look at torch.bmm for the products below 

        raw_weights        = 0# TODO: Calculate raw_weights which is the product of query and key, and is of shape (batch_size, timesteps)
        masked_raw_weights = 0# TODO: Mask the raw_weights with self.padding_mask. 
        # Take a look at pytorch's masked_fill_ function (You want the fill value to be a big negative number for the softmax to make it close to 0)

        attention_weights  = 0# TODO: Calculate the attention weights, which is the softmax of raw_weights
        context            = 0# TODO: Calculate the context - it is a product between attention_weights and value

        # Hint: You might need to use squeeze/unsqueeze to make sure that your operations work with bmm

        return context, attention_weights # Return the context, attention_weights

# Decoder
class Speller(torch.nn.Module):

    def __init__(self, embed_size, decoder_hidden_size, decoder_output_size, vocab_size, attention_module=None):
        super().__init__()

        self.vocab_size         = vocab_size

        self.embedding          = 0# TODO: Initialize the Embedding Layer (Use the nn.Embedding Layer from torch), make sure you set the correct padding_idx  

        self.lstm_cells         = torch.nn.Sequential(
                                # Create Two LSTM Cells as per LAS Architecture
                                # What should the input_size of the first LSTM Cell? 
                                # Hint: It takes in a combination of the character embedding and context from attention
                                )
    
                                # We are using LSTMCells because process individual time steps inputs and not the whole sequence.
                                # Think why we need this in terms of the query

        self.char_prob          = 0# TODO: Initialize the classification layer to generate your probability distribution over all characters

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
        char            = torch.full((batch_size,), fill_value=SOS_TOKEN, dtype= torch.long).to(DEVICE) 

        # Initialize a list to keep track of LSTM Cell Hidden and Cell Memory States, to None
        hidden_states   = [None]*len(self.decoder.lstm_cells) 

        attention_plot          = []
        context                 = 0# TODO: Initialize context (You have a few choices, refer to the writeup )
        attention_weights       = torch.zeros(batch_size, encoder_max_seq_len) # Attention Weights are zero if not using Attend Module

        # Set Attention Key, Value, Padding Mask just once
        if self.attention != None:
            self.attention.set_key_value_mask(encoder_outputs, encoder_lens)


        for t in range(timesteps):
            
            char_embed = 0#TODO: Generate the embedding for the character at timestep t

            if self.training and t > 0:
                # TODO: We want to decide which embedding to use as input for the decoder during training
                # We can use the embedding of the transcript character or the embedding of decoded/predicted character, from the previous timestep 
                # Using the embedding of the transcript character is teacher forcing, it is very important for faster convergence
                # Use a comparison between a random probability and your teacher forcing rate, to decide which embedding to use

                char_embed = 0# TODO
      
            decoder_input_embedding = 0# TODO: What do we want to concatenate as input to the decoder? (Use torch.cat)
            
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

            output_embedding     = 0# TODO: Concatenate the projected query with context for the output embedding
            # Hint: How can you get the projected query from attention
            # If you are not using attention, what will you use instead of query?

            char_prob            = self.char_prob(output_embedding)
            
            # Append the character probability distribution to the list of predictions 
            predictions.append(char_prob)

            char = 0# TODO: Get the predicted character for the next timestep from the probability distribution 
            # (Hint: Use Greedy Decoding for starters)

        attention_plot  = 0# TODO: Stack list of attetion_plots 
        predictions     = 0# TODO: Stack list of predictions 

        return predictions, attention_plot

# Sequence to Sequence Model
class LAS(torch.nn.Module):
    def __init__(self, input_size, encoder_hidden_size, 
                 vocab_size, embed_size,
                 decoder_hidden_size, decoder_output_size,
                 projection_size=128):
        
        super(LAS, self).__init__()

        self.encoder        = Listener(input_size, encoder_hidden_size, 0) # TODO: Initialize Encoder
        attention_module    = Attention(encoder_hidden_size, decoder_output_size, projection_size)# TODO: Initialize Attention
        self.decoder        = Speller(embed_size, decoder_hidden_size, decoder_output_size, vocab_size, attention_module)# TODO: Initialize Decoder, make sure you pass the attention module 

    def forward(self, x, x_lens, y=None, tf_rate=1):

        encoder_outputs, encoder_lens = self.encoder(x, x_lens) # from Listener
        predictions, attention_plot = self.decoder(encoder_outputs, encoder_lens, y, tf_rate)
        
        return predictions, attention_plot