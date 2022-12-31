import random
import numpy as np
import torch

import seaborn as sns
import matplotlib.pyplot as plt

import Levenshtein

SOS_TOKEN = 0
EOS_TOKEN = 29

def set_random_seed(seed_num=1):
	random.seed(seed_num)
	np.random.seed(seed_num)
	torch.manual_seed(seed_num)
	torch.cuda.manual_seed(seed_num)
	torch.cuda.manual_seed_all(seed_num)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def plot_attention(attention): 
    # Function for plotting attention
    # You need to get a diagonal plot
    plt.clf()
    sns.heatmap(attention, cmap='GnBu')
    plt.show()

# We have given you this utility function which takes a sequence of indices and converts them to a list of characters
def indices_to_chars(indices, vocab):
    tokens = []
    for i in indices: # This loops through all the indices
        if vocab[int(i)] == SOS_TOKEN: # If SOS is encountered, dont add it to the final list
            continue
        elif vocab[int(i)] == EOS_TOKEN: # If EOS is encountered, stop the decoding process
            break
        else:
            tokens.append(vocab[i])
    return tokens

# To make your life more easier, we have given the Levenshtein distantce / Edit distance calculation code
def calc_edit_distance(predictions, y, ly, vocab, print_example=False):

    dist                = 0
    batch_size, seq_len = predictions.shape

    for batch_idx in range(batch_size): 

        y_sliced    = indices_to_chars(y[batch_idx,0:ly[batch_idx]], vocab)
        pred_sliced = indices_to_chars(predictions[batch_idx], vocab)

        # Strings - When you are using characters from the AudioDataset
        y_string    = ''.join(y_sliced)
        pred_string = ''.join(pred_sliced)
        
        dist        += Levenshtein.distance(pred_string, y_string)
        # Comment the above and uncomment below for toy dataset 
        # dist      += Levenshtein.distance(y_sliced, pred_sliced)

    if print_example: 
        # Print y_sliced and pred_sliced if you are using the toy dataset
        print("Ground Truth : ", y_string)
        print("Prediction   : ", pred_string)
        
    dist/=batch_size
    return dist