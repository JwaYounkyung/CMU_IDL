# !kaggle competitions download -c 11-785-f22-hw3p2
# !unzip -q 11-785-f22-hw3p2.zip

# %% Import libraries
import wandb
import torch
from torchsummaryX import summary
import pandas as pd
import gc

import pandas as pd
from tqdm import tqdm
import os
import datetime

import Levenshtein
import ctcdecode
from ctcdecode import CTCBeamDecoder

import warnings
from modules.dataset import AudioDataset, AudioDatasetTest
from models.basic import Network
from modules.utils import set_random_seed, calculate_levenshtein

import torch.distributed as dist
import torch.utils.data as torchdata
from torch.nn.parallel import DistributedDataParallel as DDP

# %% Config
set_random_seed(seed_num=1)
warnings.filterwarnings('ignore')

# %% Hyperparameters
local_rank, gpu_ids = 0, [0, 1, 2, 3]
batch_size = 32

# os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2,3"
distributed = True
if distributed:
    gpu_no = len(gpu_ids)
else:
    gpu_no = 1

try:
    local_rank = int(os.environ["LOCAL_RANK"])
except:
    pass

device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
if distributed and device != 'cpu':
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    print('local_rank', local_rank)

if device != 'cpu':
    batch_size *= len(gpu_ids)

args = {
    'model_dir': os.path.join(os.path.dirname(__file__), 'weights/lstm'),
    'exp_name': 'lstm',
}
    
config = {
    "batch_size": batch_size, # 수정
    "num_workers": 24, # 수정 mac 0
    
    "architecture" : "lstm",
    "embedding_size1": 64,
    "embedding_size2": 128,
    "hidden_size" : 128,
    "num_layers" : 5,
    "dropout" : 0.3,
    "bidirectional" : True,

    "beam_width_train" : 2,
    "beam_width_test" : 50,
    "lr" : 4e-3,
    "epochs" : 100,
    "weight_decay" : 1e-5,
    "step_size" : 7,
    "scheduler_gamma" : 0.8,
    } 


# %% ARPABET PHONEME MAPPING
# DO NOT CHANGE
CMUdict_ARPAbet = {
    "" : " ", # BLANK TOKEN
    "[SIL]": "-", "NG": "G", "F" : "f", "M" : "m", "AE": "@", 
    "R"    : "r", "UW": "u", "N" : "n", "IY": "i", "AW": "W", 
    "V"    : "v", "UH": "U", "OW": "o", "AA": "a", "ER": "R", 
    "HH"   : "h", "Z" : "z", "K" : "k", "CH": "C", "W" : "w", 
    "EY"   : "e", "ZH": "Z", "T" : "t", "EH": "E", "Y" : "y", 
    "AH"   : "A", "B" : "b", "P" : "p", "TH": "T", "DH": "D", 
    "AO"   : "c", "G" : "g", "L" : "l", "JH": "j", "OY": "O", 
    "SH"   : "S", "D" : "d", "AY": "Y", "S" : "s", "IH": "I",
    "[SOS]": "[SOS]", "[EOS]": "[EOS]"}

CMUdict = list(CMUdict_ARPAbet.keys()) # 43
ARPAbet = list(CMUdict_ARPAbet.values())

PHONEMES = CMUdict
mapping = CMUdict_ARPAbet
LABELS = ARPAbet

# %% Hyperparameters

# 수정 
transforms = [] # Time Masking and Frequency Masking are 2 types of transformation, you may choose to apply
# You may pass this as a parameter to the dataset class above

root = 'data/' 
gc.collect() 

# %% Data Load
# 수정 
# train-clean-360도 같이 써야함
train_data = AudioDataset(root, PHONEMES, ["train-clean-360", "train-clean-100"], transforms=None) #TODO
val_data = AudioDataset(root, PHONEMES, ["dev-clean"], transforms=None) #TODO
test_data = AudioDatasetTest(root, PHONEMES, "test-clean", transforms=None) #TODO

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

# sanity check
for data in train_loader:
    x, y, lx, ly = data
    print(x.shape, y.shape, lx.shape, ly.shape)
    break 
for data in test_loader:
    x_test, lx_test = data
    print(x_test.shape, lx_test.shape)
    break 
input_size = x.shape[2]

# %% Model Config
OUT_SIZE = len(LABELS)

model = Network(input_size, config["embedding_size1"], config["embedding_size2"], config["hidden_size"], config["num_layers"], 
                config["dropout"], config["bidirectional"], OUT_SIZE)
model.to(device)
if distributed:
    model = DDP(model, device_ids=[local_rank])
# summary(model, x.to(device), lx)

# %%
criterion = torch.nn.CTCLoss(blank=0) 
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.999), eps=1e-08, 
                              weight_decay=config["weight_decay"]) 
# 수정
# ReduceLRonPlateau
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['scheduler_gamma']) #TODO
# Mixed Precision, if you need it
scaler = torch.cuda.amp.GradScaler()

# %% 
decoder = CTCBeamDecoder(
    LABELS,
    model_path=None,
    alpha=0,
    beta=0,
    cutoff_top_n=40,
    cutoff_prob=1.0,
    beam_width=2,#config["beam_width_test"],
    num_processes=4,
    blank_id=0,
    log_probs_input=True
)#TODO 

# Sanity Check Levenshtein Distance: 450 < d < 800
with torch.no_grad():
  for i, data in enumerate(train_loader):
      
      #TODO: 
      # Follow the following steps, and 
      # Add some print statements here for sanity checking
      
      #1. What values are you returning from the collate function
      #2. Move the features and target to <DEVICE>
      #3. Print the shapes of each to get a fair understanding 
      #4. Pass the inputs to the model
            # Think of the following before you implement:
            # 4.1 What will be the input to your model?
            # 4.2 What would the model output?
            # 4.3 Print the shapes of the output to get a fair understanding 

      # Calculate loss: https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
      # Calculating the loss is not straightforward. Check the input format of each parameter
      
      x, y, lx, ly = data
      x, y, lx, ly = x.to(device), y.to(device), lx.to(device), ly.to(device)

    #   distance = calculate_levenshtein(x, y, lx, ly, decoder, LABELS, debug = False)
    #   print(f"lev-distance: {distance}")
      x = x.permute(1, 0, 2)

      loss = criterion(x, y, lx, ly)
      print(f"loss: {loss}")



      break # one iteration is enough


# %% Train

# Checkpointing parameters
last_epoch_completed = 0
start = last_epoch_completed
end = config['epochs']
best_val_dist = float("inf") # if you're restarting from some checkpoint, use what you saw there.
dist_freq = 1

def train_step(model, train_loader, optimizer, criterion, scaler):

    model.train()
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5) 
    
    train_loss = 0
    for i, data in enumerate(train_loader):

        optimizer.zero_grad()
        x, y, lx, ly = data
        x, y, lx, ly = x.to(device), y.to(device), lx.to(device), ly.to(device)
        
        with torch.cuda.amp.autocast(): # Mixed Precision 
            outputs, outputs_length = model(x, lx)
            outputs = outputs.permute(1, 0, 2)
            loss = criterion(outputs, y, outputs_length, ly)

        train_loss += float(loss.item())

        batch_bar.set_postfix(
            loss = f"{train_loss/ (i+1):.4f}",
            lr = f"{optimizer.param_groups[0]['lr']}"
        )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update() 
        batch_bar.update()
    
    batch_bar.close()
    train_loss = float(train_loss / len(train_loader)) # TODO

    return train_loss 

# %% Validation
def evaluate(model, val_loader, criterion, epoch):
    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, leave=False, position=0, desc='Val', ncols=5)
    
    val_dist = 0
    val_loss = 0
    for i, data in enumerate(val_loader):

        x, y, lx, ly = data
        x, y, lx, ly = x.to(device), y.to(device), lx.to(device), ly.to(device)
        
        with torch.no_grad():
            outputs, outputs_length = model(x, lx)
            outputs = outputs.permute(1, 0, 2)
            loss = criterion(outputs, y, outputs_length, ly)
        
        # if epoch == end - 1:
        #     distance = calculate_levenshtein(x, y, lx, ly, decoder, LABELS, debug = False)
        #     val_dist += distance
        
        val_loss += float(loss.item())

        batch_bar.set_postfix(
            loss = f"{val_loss/ (i+1):.4f}",
        )
        batch_bar.update()
    
    batch_bar.close()
    val_loss = float(val_loss / len(val_loader)) # TODO
    val_dist = float(val_dist / len(val_loader)) # TODO

    return val_loss, val_dist

# %% wandb 
def wandb_init():
    wandb.login(key="0699a3c4c17f76e3d85a803c4d7039edb8c3a3d9")
    run = wandb.init(
        name = "lstm", ## Wandb creates random run names if you skip this field
        reinit = True, ### Allows reinitalizing runs when you re-run this cell
        # run_id = ### Insert specific run id here if you want to resume a previous run
        # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
        project = "hw3p2-ablations", ### Project should be created in your wandb account 
        config = config ### Wandb Config for your run
    )
    return run

if local_rank == 1:
    run = wandb_init()

# %% Train Loop
torch.cuda.empty_cache()
gc.collect()

#TODO: Please complete the training loop

best_val_loss = float("inf")
for epoch in range(config["epochs"]):
    curr_lr = float(optimizer.param_groups[0]['lr'])

    train_loss = train_step(model, train_loader, optimizer, criterion, scaler)
    print("\nEpoch {}/{}: \nTrain Loss {:.04f}\t Learning Rate {:.04f}".format(
        epoch + 1,
        config['epochs'],
        train_loss,
        curr_lr))
    
    val_loss, val_dist = evaluate(model, val_loader, criterion, epoch)
    print("Val Loss {:.04f}\t Val Dist {:.04f}".format(val_loss, val_dist))

    if local_rank == 1:
        wandb.log({"train_loss":train_loss, "validation_loss": val_loss,
                "validation_Dist":val_dist, "learning_Rate": curr_lr})
    
    scheduler.step()
    # 수정
    # HINT: Calculating levenshtein distance takes a long time. Do you need to do it every epoch?
    # Does the training step even need it? 
    if val_loss < best_val_loss:
      os.makedirs(args['model_dir'], exist_ok=True)
      path = os.path.join(args['model_dir'], 'checkpoint' + '.pth')
      print("Saving model")
      torch.save({'model_state_dict':model.state_dict(),
                  'optimizer_state_dict':optimizer.state_dict(),
                  'scheduler_state_dict':scheduler.state_dict(),
                  'val_loss': val_loss, 
                  'epoch': epoch}, path)
      best_val_loss = val_loss

      if local_rank == 1:
        wandb.save('checkpoint.pth')

if local_rank == 1:
    run.finish()

# %% Inference
# Follow the steps below:
# 2. Get prediction string by decoding the results of the beam decoder

decoder_test = CTCBeamDecoder(
    LABELS,
    model_path=None,
    alpha=0,
    beta=0,
    cutoff_top_n=40,
    cutoff_prob=1.0,
    beam_width=config["beam_width_test"],
    num_processes=4,
    blank_id=0,
    log_probs_input=True
)#TODO 

def make_output(h, lh, decoder, LABELS):
    # concatenate my predictions into strings
    beam_results, beam_scores, timesteps, out_len = decoder.decode(h, seq_lens=lh)
    batch_size = beam_results.shape[0] #What is the batch size
    
    dist = 0
    preds = []
    # y [4, 1476, 43] -> [4, ?] with decoder
    for i in range(batch_size): # Loop through each element in the batch
        h_sliced = beam_results[i][0][:out_len[i][0]] #TODO: Obtain the beam results
        h_string = "".join([LABELS[n] for n in h_sliced]) #TODO: Convert the beam results to phonemes
        preds.append(h_string)
        # calculate_levenshtein_distance(h_string, y[i]) #TODO: Calculate the levenshtein distance
    
    return preds

def predict(model, test_loader, decoder, LABELS):

    model.eval()
    batch_bar = tqdm(total=len(test_loader), dynamic_ncols=True, position=0, leave=False, desc='Test')
    test_results = []
  
    for i, data in enumerate(test_loader):
        x, lx = data
        x, lx = x.to(device), lx.to(device)

        with torch.no_grad():
            outputs, outputs_length = model(x, lx)
        
        preds = make_output(outputs, outputs_length, decoder, LABELS)
        outputs = torch.argmax(outputs, axis=1).detach().cpu().numpy().tolist()
        test_results.extend(preds)
        
        batch_bar.update()
      
    batch_bar.close()
    return test_results

# %% Make predictions
#TODO:

path = os.path.join(args['model_dir'], 'checkpoint' + '.pth')
checkpoint = torch.load(path,  map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

predictions = predict(model, test_loader, decoder_test, LABELS)

df = pd.read_csv('data/test-clean/transcript/random_submission.csv')
df.label = predictions

df.to_csv('results/submission' + args['exp_name'] + '.csv', index = False)
#!kaggle competitions submit -c 11-785-f22-hw3p2 -f results/submission_early.csv