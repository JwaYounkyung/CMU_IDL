import torch
import os
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class AudioDataset(torch.utils.data.Dataset):

    # For this homework, we give you full flexibility to design your data set class.
    # Hint: The data from HW1 is very similar to this HW

    #TODO
    def __init__(self, data_path, PHONEMES, partition1, partition2, transforms=None): 
        '''
        Initializes the dataset.
        '''
        # Load the directory and all files in them
        self.data_path = data_path

        mfcc_files1, transcript_files1 = self.file_list(partition1)
        mfcc_files2, transcript_files2 = self.file_list(partition2)

        self.mfcc_files = mfcc_files1 + mfcc_files2
        self.transcript_files = transcript_files1 + transcript_files2

        # 수정
        # self.mfcc_files = self.mfcc_files[:100]
        # self.transcript_files = self.transcript_files[:100]
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
            transcript = transcript[1:-1] # Remove [SOS] and [EOS]
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

        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True, padding_value=0.0) # TODO
        lengths_mfcc = [batch_mfcc[i].shape[0] for i in range(len(batch))] # TODO 

        batch_transcript_pad = pad_sequence(batch_transcript, batch_first=True, padding_value=0) # TODO
        lengths_transcript = [batch_transcript[i].shape[0] for i in range(len(batch))] # TODO

        # 수정
        # You may apply some transformation, Time and Frequency masking, here in the collate function;
        # Food for thought -> Why are we applying the transformation here and not in the __getitem__?
        #                  -> Would we apply transformation on the validation set as well?
        #                  -> Is the order of axes / dimensions as expected for the transform functions?
        
        return batch_mfcc_pad, batch_transcript_pad, torch.tensor(lengths_mfcc), torch.tensor(lengths_transcript)

# Test Dataloader
#TODO
class AudioDatasetTest(torch.utils.data.Dataset):
    # Train에서 transcript 부분만 제거
    def __init__(self, data_path, PHONEMES, partition="test-clean", transforms=None): 
        '''
        Initializes the dataset.
        '''
        # Load the directory and all files in them
        self.data_path = data_path
        self.mfcc_dir = self.data_path + partition + '/mfcc/'  #TODO
        self.mfcc_files = sorted(os.listdir(self.mfcc_dir)) #TODO

        # 수정
        # self.mfcc_files = self.mfcc_files[:100]

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
        # 수정
        return mfcc


    def collate_fn(self,batch):
        '''
        TODO
        '''
        batch_mfcc = [batch[i] for i in range(len(batch))] # TODO
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True, padding_value=0.0) # TODO
        lengths_mfcc = [batch_mfcc[i].shape[0] for i in range(len(batch))] # TODO 
        # 수정
        
        return batch_mfcc_pad, torch.tensor(lengths_mfcc)
