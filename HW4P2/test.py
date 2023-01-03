import os
import pandas as pd
import numpy as np
import Levenshtein

import torch
import torchaudio

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
import gc
from torchsummaryX import summary
import wandb
from glob import glob

import torch.distributed as dist
import torch.utils.data as torchdata
from torch.nn.parallel import DistributedDataParallel as DDP

from modules.utils import set_random_seed, plot_attention, calc_edit_distance, indices_to_chars
from modules.dataset import ToyDataset, AudioDataset, AudioDatasetTest
from models.LAS import Listener, Attention, LAS

import argparse

import warnings
warnings.filterwarnings('ignore')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(__file__)
    parser.add_argument('--distributed', type=bool, default=False)
    parser.add_argument(
        '--model_dir', default=os.path.join(os.path.dirname(__file__), 'weights/LAS'))
    parser.add_argument('--exp_name', default='LAS')
    parser.add_argument('--toy', default=False)
    args = parser.parse_args(argv)
    return args
args = parse_args()

set_random_seed(seed_num=1)

# multi gpu
distributed = args.distributed
local_rank, gpu_ids = 0, [0, 1, 2, 3]
batch_size = 16

if distributed:
    gpu_no = len(gpu_ids)
else:
    gpu_no = 1

try:
    local_rank = int(os.environ["LOCAL_RANK"])
except:
    pass

device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
DEVICE = device

if distributed and device != 'cpu':
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    print('local_rank', local_rank)

if device != 'cpu' and distributed:
    batch_size *= len(gpu_ids)

# Global config dict.
config = {
    "batch_size"      : batch_size,
    "num_workers"     : 24, # mac 0

    "encoder_hidden_size" : 512,
    "locked_dropout"      : 0.35,
    "dropout"         : 0.2,

    "epochs"          : 70,
    "lr"              : 1e-3,
    "weight_decay"    : 5e-6,
    "step_size"       : 3,
    "scheduler_gamma" : 0.5,
}

if args.toy:
    # Load the toy dataset
    X_train = np.load("data/f0176_mfccs_train.npy") # (1600, 176, 26)
    X_valid = np.load("data/f0176_mfccs_dev.npy")   # (1600, 176, 26)
    Y_train = np.load("data/f0176_hw3p2_train.npy") # (1600, 23)
    Y_valid = np.load("data/f0176_hw3p2_dev.npy")   # (1600, 23)

    # This is how you actually need to find out the different trancripts in a dataset. 
    # Can you think whats going on here? Why are we using a np.unique?
    VOCAB_MAP           = dict(zip(np.unique(Y_valid), range(len(np.unique(Y_valid))))) 
    VOCAB_MAP["[PAD]"]  = len(VOCAB_MAP)
    VOCAB               = list(VOCAB_MAP.keys()) # 43 unique characters

    SOS_TOKEN = VOCAB_MAP["[SOS]"]
    EOS_TOKEN = VOCAB_MAP["[EOS]"]
    PAD_TOKEN = VOCAB_MAP["[PAD]"]

    Y_train = [np.array([VOCAB_MAP[p] for p in seq]) for seq in Y_train]
    Y_valid = [np.array([VOCAB_MAP[p] for p in seq]) for seq in Y_valid]

else:
    # These are the various characters in the transcripts of the datasetW
    VOCAB = ['<sos>',   
            'A',   'B',    'C',    'D',    
            'E',   'F',    'G',    'H',    
            'I',   'J',    'K',    'L',       
            'M',   'N',    'O',    'P',    
            'Q',   'R',    'S',    'T', 
            'U',   'V',    'W',    'X', 
            'Y',   'Z',    "'",    ' ', 
            '<eos>'] # 30 unique characters

    VOCAB_MAP = {VOCAB[i]:i for i in range(0, len(VOCAB))}

    SOS_TOKEN = VOCAB_MAP["<sos>"]
    EOS_TOKEN = VOCAB_MAP["<eos>"]

# TODO: Create the datasets and dataloaders
# All these things are similar to HW3P2
# You can reuse the same code
root = 'data/hw4p2/' 
gc.collect() 

# Time Masking and Frequency Masking are 2 types of transformation, you may choose to apply
transform = None #torchaudio.transforms.SlidingWindowCmn()
if args.toy:
    train_data = ToyDataset("train", X_train, Y_train)
    val_data = ToyDataset("valid", X_valid, Y_valid)
else:
    train_data = AudioDataset(root, VOCAB, ["train-clean-100"], transform=transform)
    val_data = AudioDataset(root, VOCAB, ["dev-clean"], transform=transform)
test_data = AudioDatasetTest(root, VOCAB, "test-clean", transform=transform)



train_sampler = None
if distributed:
    train_sampler = torchdata.distributed.DistributedSampler(train_data)

train_loader = torch.utils.data.DataLoader(train_data, num_workers= config['num_workers'],
                                           batch_size=config['batch_size'], pin_memory=True,
                                           shuffle=(train_sampler is None), sampler=train_sampler,    
                                           collate_fn=train_data.collate_fn)

val_loader = torch.utils.data.DataLoader(val_data, num_workers=config['num_workers'],
                                         batch_size=config['batch_size'], pin_memory=True,
                                         shuffle=False, collate_fn=val_data.collate_fn)

test_loader = torch.utils.data.DataLoader(test_data, num_workers=config['num_workers'], 
                                          batch_size=config['batch_size'], pin_memory=True, 
                                          shuffle=False, collate_fn=test_data.collate_fn)

print("Batch size: ", config['batch_size'])
print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
print("Val dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))
print("Test dataset samples = {}, batches = {}".format(test_data.__len__(), len(test_loader)))

# The sanity check for shapes also are similar
# sanity check
for data in train_loader:
    x, y, lx, ly = data
    # print(x.shape, y.shape, lx.shape, ly.shape)
    # [batch_size, 1658, 15] [batch_size, 246] [batch_size] [batch_size]
    break 
# for data in test_loader:
#     x_test, lx_test = data
#     print(x_test.shape, lx_test.shape)
#     # [batch_size, 1047, 15] [batch_size]
#     break 
input_size = 15

# Encoder Check
# encoder_hidden_size = 256
# encoder = Listener(input_size, encoder_hidden_size, 0)# TODO: Initialize Listener
# print(encoder)
# # summary(encoder, x.to(DEVICE), lx)
# del encoder


# Baseline LAS has the following configuration:
# Encoder bLSTM/pbLSTM Hidden Dimension of 512 (256 per direction)
# Decoder Embedding Layer Dimension of 256
# Decoder Hidden Dimension of 512 
# Decoder Output Dimension of 128
# Attention Projection Size of 128
# Feel Free to Experiment with this 

model = LAS(
    # Initialize your model 
    # Read the paper and think about what dimensions should be used
    # You can experiment on these as well, but they are not requried for the early submission
    # Remember that if you are using weight tying, some sizes need to be the same
    input_size, encoder_hidden_size=config['encoder_hidden_size'],
    vocab_size=len(VOCAB), embed_size=256,
    decoder_hidden_size=512, decoder_output_size=128, projection_size=128,
    locked_dropout=config['locked_dropout'], dropout=config['dropout'],
    device=DEVICE,
)

model = model.to(DEVICE)
if distributed:
    model = DDP(model, device_ids=[local_rank])
# print(model)

summary(model, 
        x=x.to(DEVICE), 
        x_lens=lx, 
        y=y.to(DEVICE))


optimizer   = torch.optim.AdamW(model.parameters(), lr=config['lr'], amsgrad=True, weight_decay=config['weight_decay'])
criterion   = torch.nn.CrossEntropyLoss(reduction='none') # Why are we using reduction = 'none' ? 
scaler      = torch.cuda.amp.GradScaler()
# Optional: Create a custom class for a Teacher Force Schedule 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=2e-8, 
                patience=config['step_size'], factor=config['scheduler_gamma'])


# Optional: Load your best model Checkpoint here
path = os.path.join(args.model_dir, 'checkpoint' + '.pth')
checkpoint = torch.load(path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])


# TODO: Create a testing function similar to validation 
def make_string(predictions, vocab):
    pred_strs = []
    for i in range(predictions.shape[0]):
        pred_sliced = indices_to_chars(predictions[i], vocab)
        pred_string = "".join(pred_sliced)
        pred_strs.append(pred_string)

    return pred_strs

def predict(model, test_loader):

    model.eval()
    batch_bar = tqdm(total=len(test_loader), dynamic_ncols=True, position=0, leave=False, desc='Test', ncols=5)
    test_results = []
  
    for i, (x, lx) in enumerate(test_loader):
        x, lx = x.to(DEVICE), lx.to(DEVICE)

        with torch.no_grad():
            predictions, attentions = model(x, lx)
        
        greedy_predictions   = torch.argmax(predictions, dim=2)
        pred_str = make_string(greedy_predictions, VOCAB)
        test_results.extend(pred_str)
        
        batch_bar.update()

        del x, lx
      
    batch_bar.close()
    return test_results

predictions = predict(model, test_loader)
# TODO: Create a file with all predictions 
df = pd.read_csv('data/hw4p2/test-clean/transcript/random_submission.csv')
df.label = predictions
df.rename(columns={"index": "id"}, inplace=True)

df.to_csv('HW4P2/results/submission' + args.exp_name + '.csv', index = False)
# TODO: Submit to Kaggle
#!kaggle competitions submit -c 11-785-f22-hw3p2 -f results/submission_early.csv