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
local_rank, gpu_ids = 0, [0, 1, 2, 3, 4, 5, 6, 7]
batch_size = 128

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

    "architecture"    : "LAS",
    "tf_rate"         : 1.0,
    "dropout"         : 0.0,

    "epochs"          : 70,
    "lr"              : 1e-3,
    "weight_decay"    : 5e-6,
    "step_size"       : 5,
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
# for data in train_loader:
#     x, y, lx, ly = data
#     print(x.shape, y.shape, lx.shape, ly.shape)
#     # [batch_size, 1658, 15] [batch_size, 246] [batch_size] [batch_size]
#     break 
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
    input_size, encoder_hidden_size=256,
    vocab_size=len(VOCAB), embed_size=256,
    decoder_hidden_size=512, decoder_output_size=128,
    device=DEVICE,
    projection_size=128
)

model = model.to(DEVICE)
if distributed:
    model = DDP(model, device_ids=[local_rank])
# print(model)

# summary(model, 
#         x=x.to(DEVICE), 
#         x_lens=lx, 
#         y=y.to(DEVICE))


optimizer   = torch.optim.AdamW(model.parameters(), lr=config['lr'], amsgrad=True, weight_decay=config['weight_decay'])
criterion   = torch.nn.CrossEntropyLoss(reduction='none') # Why are we using reduction = 'none' ? 
scaler      = torch.cuda.amp.GradScaler()
# Optional: Create a custom class for a Teacher Force Schedule 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=2e-8, 
                patience=config['step_size'], factor=config['scheduler_gamma'])

def train(model, dataloader, criterion, optimizer, teacher_forcing_rate):

    model.train()
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5)

    running_loss        = 0.0
    running_perplexity  = 0.0
    
    for i, (x, y, lx, ly) in enumerate(dataloader):
        optimizer.zero_grad()

        x, y, lx, ly = x.to(DEVICE), y.to(DEVICE), lx.to(DEVICE), ly.to(DEVICE)

        with torch.cuda.amp.autocast():
            predictions, attention_plot = model(x, lx, y=y, tf_rate=teacher_forcing_rate)

            # Predictions are of Shape (batch_size, timesteps, vocab_size). 
            # Transcripts are of shape (batch_size, timesteps) Which means that you have batch_size amount of batches with timestep number of tokens.
            # So in total, you have batch_size*timesteps amount of characters.
            # Similarly, in predictions, you have batch_size*timesteps amount of probability distributions.
            # How do you need to modify transcipts and predictions so that you can calculate the CrossEntropyLoss? Hint: Use Reshape/View and read the docs
            loss        =  criterion(predictions.view(-1, len(VOCAB)), y.view(-1)) # TODO: Cross Entropy Loss

            # TODO: Create a boolean mask using the lengths of your transcript that remove the influence of padding indices 
            # (in transcripts) in the loss 
            mask        = torch.ones_like(y) # TODO: Create a mask of the same shape as y
            for i in range(ly.shape[0]):
                if ly[i] < y.shape[1]:
                    mask[i, ly[i]:] = 0
            
            mask = mask.view(-1)
            masked_loss = mask*loss # Product between the mask and the loss, divided by the mask's sum. Hint: You may want to reshape the mask too 
            masked_loss = masked_loss.mean()

            perplexity  = torch.exp(masked_loss) # Perplexity is defined the exponential of the loss

            running_loss        += masked_loss.item()
            running_perplexity  += perplexity.item()

        # Backward on the masked loss
        scaler.scale(masked_loss).backward()
        # Optional: Use torch.nn.utils.clip_grad_norm to clip gradients to prevent them from exploding, if necessary
        # If using with mixed precision, unscale the Optimizer First before doing gradient clipping
        scaler.step(optimizer)
        scaler.update()

        batch_bar.set_postfix(
            loss="{:.04f}".format(running_loss/(i+1)),
            perplexity="{:.04f}".format(running_perplexity/(i+1)),
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])),
            tf_rate='{:.02f}'.format(teacher_forcing_rate))
        batch_bar.update()

        del x, y, lx, ly
        torch.cuda.empty_cache()
    
    running_loss /= len(dataloader)
    running_perplexity /= len(dataloader)
    batch_bar.close()

    return running_loss, running_perplexity, attention_plot


def validate(model, dataloader):

    model.eval()

    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc="Val", ncols=5)

    running_lev_dist = 0.0

    for i, (x, y, lx, ly) in enumerate(dataloader):
        x, y, lx, ly = x.to(DEVICE), y.to(DEVICE), lx.to(DEVICE), ly.to(DEVICE)

        with torch.no_grad():
            predictions, attentions = model(x, lx, y=None)
            loss        =  criterion(predictions.view(-1, len(VOCAB)), y.view(-1)) # TODO: Cross Entropy Loss

            # TODO: Create a boolean mask using the lengths of your transcript that remove the influence of padding indices 
            # (in transcripts) in the loss 
            mask        = torch.ones_like(y) # TODO: Create a mask of the same shape as y
            for i in range(ly.shape[0]):
                if ly[i] < y.shape[1]:
                    mask[i, ly[i]:] = 0
            
            mask = mask.view(-1)
            masked_loss = mask*loss # Product between the mask and the loss, divided by the mask's sum. Hint: You may want to reshape the mask too 
            masked_loss = masked_loss.mean()

            running_loss        += masked_loss.item()

        # Greedy Decoding
        greedy_predictions   = torch.argmax(predictions, dim=2) # TODO: How do you get the most likely character from each distribution in the batch?

        # Calculate Levenshtein Distance
        running_lev_dist    += calc_edit_distance(greedy_predictions, y, ly, VOCAB, print_example = False) # You can use print_example = True for one specific index i in your batches if you want

        batch_bar.set_postfix(
            dist="{:.04f}".format(running_lev_dist/(i+1)))
        batch_bar.update()

        del x, y, lx, ly
        torch.cuda.empty_cache()
    
    running_loss /= len(dataloader)
    running_lev_dist /= len(dataloader)
    batch_bar.close()

    return running_loss, running_lev_dist


# Login to Wandb
# Initialize your Wandb Run Here
# Optional: Save your model architecture in a txt file, and save the file to Wandb
def wandb_init():
    wandb.login(key="0699a3c4c17f76e3d85a803c4d7039edb8c3a3d9") # enter your wandb key here
    run = wandb.init(
        name = "LAS", ## Wandb creates random run names if you skip this field
        reinit = True, ### Allows reinitalizing runs when you re-run this cell
        # run_id = ### Insert specific run id here if you want to resume a previous run
        # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
        project = "hw4p2", ### Project should be created in your wandb account 
        config = config ### Wandb Config for your run
    )
    return run

if local_rank == 0:
    run = wandb_init()


torch.cuda.empty_cache()
gc.collect()

best_lev_dist = float("inf")
for epoch in range(0, config['epochs']):
    curr_lr = float(optimizer.param_groups[0]['lr'])
    # Call train and validate
    train_loss, train_perplexity, attention_plot = train(model, train_loader, criterion, optimizer, config['tf_rate'])
    val_loss, val_lev_dist = validate(model, val_loader)
    
    # Print your metrics
    print("\nEpoch {}/{}: \nTrain Loss {:.04f}\t  Train Perplexity {:.04f}".format(
          epoch + 1, config['epochs'], train_loss, train_perplexity))

    print("Val Loss {:.04f}\t Val Levenshtein Distance {:.04f}".format(
          val_loss, val_lev_dist))
    
    # Plot Attention 
    plot_attention(attention_plot)

    # Log metrics to Wandb
    if local_rank == 0:
        wandb.log({"train_loss":train_loss, "validation_loss": val_loss,
                   "validation_dist":val_lev_dist, "learning_rate": curr_lr,
                   "best_dist": best_lev_dist})

    # Optional: Scheduler Step / Teacher Force Schedule Step
    if val_lev_dist <= best_lev_dist:
        best_lev_dist = val_lev_dist
        # Save your model checkpoint here
        os.makedirs(args.model_dir, exist_ok=True)
        path = os.path.join(args.model_dir, 'checkpoint' + '.pth')
        print("Saving model")
        torch.save({'model_state_dict':model.state_dict(),
                    'val_loss': val_lev_dist, 
                    'epoch': epoch}, path)

    scheduler.step(val_lev_dist)

if local_rank == 0:
    run.finish()

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
        torch.cuda.empty_cache()
      
    batch_bar.close()
    return test_results

predictions = predict(model, test_loader)
# TODO: Create a file with all predictions 
df = pd.read_csv('data/hw4p2/test-clean/transcript/random_submission.csv')
df.label = predictions
df.rename(columns={"index": "id"})

df.to_csv('HW4P2/results/submission' + args.exp_name + '.csv', index = False)
# TODO: Submit to Kaggle
#!kaggle competitions submit -c 11-785-f22-hw3p2 -f results/submission_early.csv