import torch
import numpy as np
from torchsummaryX import summary
import sklearn
import gc
import zipfile
import pandas as pd
from tqdm.auto import tqdm
import os
import datetime
import wandb
from collections import Counter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)

### PHONEME LIST
PHONEMES = [
            'SIL',   'AA',    'AE',    'AH',    'AO',    'AW',    'AY',  
            'B',     'CH',    'D',     'DH',    'EH',    'ER',    'EY',
            'F',     'G',     'HH',    'IH',    'IY',    'JH',    'K',
            'L',     'M',     'N',     'NG',    'OW',    'OY',    'P',
            'R',     'S',     'SH',    'T',     'TH',    'UH',    'UW',
            'V',     'W',     'Y',     'Z',     'ZH',    '<sos>', '<eos>']

# Dataset class to load train and validation data

class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, context, offset=0, partition= "train-clean-100", limit=-1): # Feel free to add more arguments

        self.context = context
        self.offset = offset
        self.data_path = data_path

        self.mfcc_dir = self.data_path + partition + '/mfcc/'  # TODO: MFCC directory - use partition to acces train/dev directories from kaggle data
        self.transcript_dir = self.data_path + partition + '/transcript/' # TODO: Transcripts directory - use partition to acces train/dev directories from kaggle data

        mfcc_names = sorted(os.listdir(self.mfcc_dir))[:100] # TODO: List files in sefl.mfcc_dir_dir using os.listdir in sorted order, optionally subset using limit to slice the number of files you load
        transcript_names = sorted(os.listdir(self.transcript_dir))[:100] # TODO: List files in self.transcript_dir using os.listdir in sorted order, optionally subset using limit to slice the number of files you load

        assert len(mfcc_names) == len(transcript_names) # Making sure that we have the same no. of mfcc and transcripts

        self.mfccs, self.transcripts = [], []

        # TODO:
        # Iterate through mfccs and transcripts
        for i in range(0, len(mfcc_names)):
        #   Load a single mfcc
            mfcc = np.load(self.mfcc_dir+mfcc_names[i])
        #   Optionally do Cepstral Normalization of mfcc
        #   Load the corresponding transcript
            transcript = np.load(self.transcript_dir+transcript_names[i])
            transcript = transcript[1:-1] # Remove [SOS] and [EOS] from the transcript (Is there an efficient way to do this 
                                          # without traversing through the transcript?)
        #   Append each mfcc to self.mfcc, transcript to self.transcript
            self.mfccs.append(mfcc)
            self.transcripts.append(transcript)

        

        # NOTE:
        # Each mfcc is of shape T1 x 15, T2 x 15, ...
        # Each transcript is of shape (T1+2) x 15, (T2+2) x 15 before removing [SOS] and [EOS]

        # TODO: Concatenate all mfccs in self.mfccs such that the final shape is T x 15 (Where T = T1 + T2 + ...) 
        self.mfccs = np.concatenate(self.mfccs)

        # TODO: Concatenate all transcripts in self.transcripts such that the final shape is (T,) meaning, each time step has one phoneme output
        self.transcripts = np.concatenate(self.transcripts)
        # Hint: Use numpy to concatenate

        # Take some time to think about what we have done. self.mfcc is an array of the format (Frames x Features). Our goal is to recognize phonemes of each frame
        # From hw0, you will be knowing what context is. We can introduce context by padding zeros on top and bottom of self.mfcc
        self.mfccs = np.pad(self.mfccs, [(self.context, self.context), (0, 0)], mode='constant') # TODO

        # These are the available phonemes in the transcript
        self.phonemes = [
            'SIL',   'AA',    'AE',    'AH',    'AO',    'AW',    'AY',  
            'B',     'CH',    'D',     'DH',    'EH',    'ER',    'EY',
            'F',     'G',     'HH',    'IH',    'IY',    'JH',    'K',
            'L',     'M',     'N',     'NG',    'OW',    'OY',    'P',
            'R',     'S',     'SH',    'T',     'TH',    'UH',    'UW',
            'V',     'W',     'Y',     'Z',     'ZH',    '<sos>', '<eos>']
        self.phonemes_dic = {k: v for v, k in enumerate(self.phonemes)}
        # But the neural network cannot predict strings as such. Instead we map these phonemes to integers

        self.transcripts = [self.phonemes_dic[i] for i in self.transcripts] # TODO: Map the phonemes to their corresponding list indexes in self.phonemes
        # Now, if an element in self.transcript is 0, it means that it is 'SIL' (as per the above example)
        a = 0
        for i in self.transcripts:
            if i==0: 
                a = a+1
        
        res = Counter(self.transcripts)
        tar_ele = res.most_common()[-1][0]
        print(tar_ele)    
        print(self.phonemes[tar_ele])  
        

        # Length of the dataset is now the length of concatenated mfccs/transcripts
        self.length = len(self.mfccs)

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        
        frames = self.mfccs[ind:ind+2*self.context+1] # TODO: Based on context and offset, return a frame at given index with context frames to the left, and right.
        # After slicing, you get an array of shape 2*context+1 x 15. But our MLP needs 1d data and not 2d.
        frames = frames.flatten() # TODO: Flatten to get 1d data

        frames = torch.FloatTensor(frames) # Convert to tensors
        phoneme = torch.tensor(self.transcripts[ind])       

        return frames, phoneme

class AudioTestDataset(torch.utils.data.Dataset):
    # TODO: Create a test dataset class similar to the previous class but you dont have transcripts for this
    # Imp: Read the mfccs in sorted order, do NOT shuffle the data here or in your dataloader.
    def __init__(self, data_path, context, offset=0, partition= "test-clean", limit=-1): # Feel free to add more arguments

        self.context = context
        self.offset = offset
        self.data_path = data_path

        self.mfcc_dir = self.data_path + partition + '/mfcc/'  # TODO: MFCC directory - use partition to acces train/dev directories from kaggle data

        mfcc_names = sorted(os.listdir(self.mfcc_dir)) # TODO: List files in sefl.mfcc_dir_dir using os.listdir in sorted order, optionally subset using limit to slice the number of files you load

        self.mfccs= []

        # TODO:
        # Iterate through mfccs and transcripts
        for i in range(0, len(mfcc_names)):
        #   Load a single mfcc
            mfcc = np.load(self.mfcc_dir+mfcc_names[i])
        #   Optionally do Cepstral Normalization of mfcc
        #   Load the corresponding transcript
        #   Append each mfcc to self.mfcc, transcript to self.transcript
            self.mfccs.append(mfcc)
        

        # NOTE:
        # Each mfcc is of shape T1 x 15, T2 x 15, ...
        # Each transcript is of shape (T1+2) x 15, (T2+2) x 15 before removing [SOS] and [EOS]

        # TODO: Concatenate all mfccs in self.mfccs such that the final shape is T x 15 (Where T = T1 + T2 + ...) 
        self.mfccs = np.concatenate(self.mfccs)
        # Hint: Use numpy to concatenate

        # Take some time to think about what we have done. self.mfcc is an array of the format (Frames x Features). Our goal is to recognize phonemes of each frame
        # From hw0, you will be knowing what context is. We can introduce context by padding zeros on top and bottom of self.mfcc
        self.mfccs = np.pad(self.mfccs, [(self.context, self.context), (0, 0)], mode='constant') # TODO

        # Length of the dataset is now the length of concatenated mfccs/transcripts
        self.length = len(self.mfccs) - 2*self.context

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        
        frames = self.mfccs[ind:ind+2*self.context+1] # TODO: Based on context and offset, return a frame at given index with context frames to the left, and right.
        # After slicing, you get an array of shape 2*context+1 x 15. But our MLP needs 1d data and not 2d.
        frames = frames.flatten() # TODO: Flatten to get 1d data

        frames = torch.FloatTensor(frames) # Convert to tensors

        return frames


config = {
    'epochs': 10,
    'batch_size' : 1024,
    'context' : 10,
    'learning_rate' : 0.001,
    'architecture' : 'very-low-cutoff'
    # Add more as you need them - e.g dropout values, weight decay, scheduler parameters
}

train_data = AudioDataset('data/',config['context'],offset=0,partition= "train-clean-100") #TODO: Create a dataset object using the AudioDataset class for the training data 

val_data = AudioDataset('data/',config['context'],offset=0,partition= "dev-clean") # TODO: Create a dataset object using the AudioDataset class for the validation data 

test_data = AudioTestDataset('data/',config['context'],offset=0,partition= "test-clean") # TODO: Create a dataset object using the AudioTestDataset class for the test data 

# Define dataloaders for train, val and test datasets
# Dataloaders will yield a batch of frames and phonemes of given batch_size at every iteration
train_loader = torch.utils.data.DataLoader(train_data, num_workers= 4,
                                           batch_size=config['batch_size'], pin_memory= True,
                                           shuffle= True)

val_loader = torch.utils.data.DataLoader(val_data, num_workers= 2,
                                         batch_size=config['batch_size'], pin_memory= True,
                                         shuffle= False)

test_loader = torch.utils.data.DataLoader(test_data, num_workers= 2, 
                                          batch_size=config['batch_size'], pin_memory= True, 
                                          shuffle= False)


print("Batch size: ", config['batch_size'])
print("Context: ", config['context'])
print("Input size: ", (2*config['context']+1)*15)
print("Output symbols: ", len(PHONEMES))

print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
print("Validation dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))
print("Test dataset samples = {}, batches = {}".format(test_data.__len__(), len(test_loader)))

# Testing code to check if your data loaders are working
for i, data in enumerate(test_loader):
    frames = data    

