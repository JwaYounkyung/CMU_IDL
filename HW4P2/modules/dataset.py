import torch
import os
import numpy as np
from torch.nn.utils.rnn import pad_sequence

SOS_TOKEN = 0
EOS_TOKEN = 29

# Dataset class for the Toy dataset
class ToyDataset(torch.utils.data.Dataset):

    def __init__(self, partition, X, Y):

    
        if partition == "train":
            self.mfccs = X[:, :, :15]
            self.transcripts = Y

        elif partition == "valid":
            self.mfccs = X[:, :, :15]
            self.transcripts = Y

        assert len(self.mfccs) == len(self.transcripts)

        self.length = len(self.mfccs)

    def __len__(self):

        return self.length

    def __getitem__(self, i):

        x = torch.tensor(self.mfccs[i])
        y = torch.tensor(self.transcripts[i])

        return x, y

    def collate_fn(self, batch):

        x_batch, y_batch = list(zip(*batch))

        x_lens      = [x.shape[0] for x in x_batch] 
        y_lens      = [y.shape[0] for y in y_batch] 

        x_batch_pad = torch.nn.utils.rnn.pad_sequence(x_batch, batch_first=True, padding_value= EOS_TOKEN)
        y_batch_pad = torch.nn.utils.rnn.pad_sequence(y_batch, batch_first=True, padding_value= EOS_TOKEN) 
        
        return x_batch_pad, y_batch_pad, torch.tensor(x_lens), torch.tensor(y_lens)

# TODO: Create a dataset class which is exactly the same as HW3P2. You are free to reuse it. 
# The only change is that the transcript mapping is different for this HW.
# Note: We also want to retain SOS and EOS tokens in the transcript this time.
class AudioDataset(torch.utils.data.Dataset):

    # For this homework, we give you full flexibility to design your data set class.
    # Hint: The data from HW1 is very similar to this HW

    #TODO
    def __init__(self, data_path, PHONEMES, partitions, transform=None): 
        '''
        Initializes the dataset.
        '''
        # Load the directory and all files in them
        self.data_path = data_path
        self.transform = transform

        self.mfcc_files, self.transcript_files = [], []
        for partition in partitions:
            mfccs, transcripts = self.file_list(partition)
            self.mfcc_files.extend(mfccs)
            self.transcript_files.extend(transcripts)

        # 수정
        # self.mfcc_files = self.mfcc_files[:10]
        # self.transcript_files = self.transcript_files[:10]
        assert len(self.mfcc_files) == len(self.transcript_files) 

        self.PHONEMES = PHONEMES
        self.length = len(self.mfcc_files)
        self.phonems_ind = {k: v for v, k in enumerate(self.PHONEMES)}

        #TODO
        self.mfccs, self.transcripts = [], []
        for i in range(0, self.length):
            mfcc = np.load(self.mfcc_files[i])
            # 수정 
            # Cepstral Normalization of mfcc (train, test both)
            transcript = np.load(self.transcript_files[i])
            transcript = [self.phonems_ind[phonem] for phonem in transcript]
            self.mfccs.append(mfcc)
            self.transcripts.append(transcript)

    def file_list(self, partition):
        mfcc_dir = self.data_path + partition + '/mfcc/'  #TODO
        transcript_dir = self.data_path + partition + '/transcript/raw/' #TODO

        mfcc_files = sorted(os.listdir(mfcc_dir)) #TODO
        transcript_files = sorted(os.listdir(transcript_dir)) #TODO

        mfcc_files = [mfcc_dir + file for file in mfcc_files]
        transcript_files = [transcript_dir + file for file in transcript_files]

        return mfcc_files, transcript_files

    def __len__(self):
        '''
        TODO: What do we return here?
        '''
        return self.length

    def __getitem__(self, ind):
        '''
        TODO: RETURN THE MFCC COEFFICIENTS AND ITS CORRESPONDING LABELS
        '''
        mfcc = torch.FloatTensor(self.mfccs[ind]) # TODO
        mfcc = (mfcc - mfcc.mean(axis=0))/mfcc.std(axis=0)

        transcript = torch.tensor(self.transcripts[ind]) # TODO

        # 수정
        # You may choose to apply transformations here or in collate
          # Pros of transforming in get item:
            # - More variability in comparison to collate
          # Conss of transforming in get item
            # - More time consuming (Remember get item is called every single time)
        
        return mfcc, transcript


    def collate_fn(self,batch):
        '''
        TODO:
        1.  Extract the features and labels from 'batch'
        2.  We will additionally need to pad both features and labels,
            look at pytorch's docs for pad_sequence
        3.  This is a good place to perform transforms, if you so wish. 
            Performing them on batches will speed the process up a bit.
        4.  Return batch of features, labels, lenghts of features, 
            and lengths of labels.
        '''
        # batch -> [(mfcc, transcript), (mfcc, transcript), ......]

        # batch of input mfcc coefficients
        batch_mfcc = [batch[i][0] for i in range(len(batch))] # TODO
        
        # batch of output phonemes
        batch_transcript = [batch[i][1] for i in range(len(batch))] # TODO

        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True, padding_value=EOS_TOKEN) # TODO
        lengths_mfcc = [batch_mfcc[i].shape[0] for i in range(len(batch))] # TODO 

        batch_transcript_pad = pad_sequence(batch_transcript, batch_first=True, padding_value=EOS_TOKEN) # TODO
        lengths_transcript = [batch_transcript[i].shape[0] for i in range(len(batch))] # TODO

        # 수정
        # You may apply some transformation, Time and Frequency masking, here in the collate function;
        # Food for thought -> Why are we applying the transformation here and not in the __getitem__?
        #                  -> Would we apply transformation on the validation set as well?
        #                  -> Is the order of axes / dimensions as expected for the transform functions?
        if self.transform:
            batch_mfcc_pad = self.transform(batch_mfcc_pad)
        
        return batch_mfcc_pad, batch_transcript_pad, torch.tensor(lengths_mfcc), torch.tensor(lengths_transcript)

# TODO: Similarly, create a test dataset class
class AudioDatasetTest(torch.utils.data.Dataset):
    # remove transcript
    def __init__(self, data_path, PHONEMES, partition="test-clean", transform=None): 
        '''
        Initializes the dataset.
        '''
        # Load the directory and all files in them
        self.data_path = data_path
        self.transform = transform
        self.mfcc_dir = self.data_path + partition + '/mfcc/'  #TODO
        self.mfcc_files = sorted(os.listdir(self.mfcc_dir)) #TODO

        # 수정
        # self.mfcc_files = self.mfcc_files[:10]

        self.PHONEMES = PHONEMES
        self.length = len(self.mfcc_files)
        self.phonems_ind = {k: v for v, k in enumerate(self.PHONEMES)}

        #TODO
        self.mfccs = []
        for i in range(0, self.length):
            mfcc = np.load(self.mfcc_dir+self.mfcc_files[i])
            # 수정 
            # Cepstral Normalization of mfcc
            self.mfccs.append(mfcc)

    def __len__(self):
        '''
        TODO: What do we return here?
        '''
        return self.length

    def __getitem__(self, ind):
        '''
        TODO: RETURN THE MFCC COEFFICIENTS AND ITS CORRESPONDING LABELS
        '''
        mfcc = torch.FloatTensor(self.mfccs[ind]) # TODO
        mfcc = (mfcc - mfcc.mean(axis=0))/mfcc.std(axis=0)

        return mfcc


    def collate_fn(self,batch):
        '''
        TODO
        '''
        batch_mfcc = [batch[i] for i in range(len(batch))] # TODO
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True, padding_value=EOS_TOKEN) # TODO
        lengths_mfcc = [batch_mfcc[i].shape[0] for i in range(len(batch))] # TODO 
        # 수정
        if self.transform:
            batch_mfcc_pad = self.transform(batch_mfcc_pad)

        return batch_mfcc_pad, torch.tensor(lengths_mfcc)
