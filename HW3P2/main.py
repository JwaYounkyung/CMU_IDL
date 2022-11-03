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

# %% Config
set_random_seed(seed_num=1)
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)

# %% Hyperparameters
model_dir = os.path.join(os.path.dirname(__file__), 'weights/lstm')
config = {
    "batch_size": 4, # 수정
    "num_workers": 0, # 수정
    
    "architecture" : "lstm",
    "embedding_size": 64,
    "hidden_size" : 128,
    "num_layers" : 2,
    "dropout" : 0.25,
    "bidirectional" : True,

    "beam_width_train" : 2,
    "beam_width_test" : 20,
    "lr" : 2e-3,
    "epochs" : 30,
    "weight_decay" : 1e-5,
    "step_size" : 10,
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
train_data = AudioDataset(root, PHONEMES, "train-clean-100", transforms=None) #TODO
val_data = AudioDataset(root, PHONEMES, "dev-clean", transforms=None) #TODO
test_data = AudioDatasetTest(root, PHONEMES, "test-clean", transforms=None) #TODO

train_loader = torch.utils.data.DataLoader(train_data, num_workers= config['num_workers'],
                                           batch_size=config['batch_size'], pin_memory=True,
                                           shuffle=True, collate_fn=train_data.collate_fn)

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
print(OUT_SIZE)

torch.cuda.empty_cache()
model = Network(input_size, config["embedding_size"], config["hidden_size"], config["num_layers"], 
                config["dropout"], config["bidirectional"], OUT_SIZE).to(device)
summary(model, x.to(device), lx)

# %%
criterion = torch.nn.CTCLoss(blank=0) 
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.999), eps=1e-08, 
                              weight_decay=config["weight_decay"]) 
# 수정
# ReduceLRonPlateau
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['scheduler_gamma']) #TODO
# Mixed Precision, if you need it
scaler = torch.cuda.amp.GradScaler()

# %% CTC Beam Decoder
#TODO
# Declare the decoder. Use the CTC Beam Decoder to decode phonemes
'''
Create an object the ctc beam decoder here
  # The following would change based on the problem statement:
    1. LABELS
    2. BEAM_WIDTH
    2. BLANK_ID
'''
# CTC Beam Decoder Doc: https://github.com/parlance/ctcdecode
decoder = CTCBeamDecoder(
    LABELS,
    model_path=None,
    alpha=0,
    beta=0,
    cutoff_top_n=40,
    cutoff_prob=1.0,
    beam_width=config['beam_width_train'],
    num_processes=4,
    blank_id=0,
    log_probs_input=False
)#TODO 
# %% 
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
      x = x.permute(1, 0, 2)

      loss = criterion(x, y, lx, ly)
      print(f"loss: {loss}")

    #   distance = calculate_levenshtein(x, y, lx, ly, decoder, LABELS, debug = False)
    #   print(f"lev-distance: {distance}")

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
            x = x.permute(1, 0, 2)
            # 수정
            # y [4, 1476, 43] -> [4, ?] with decoder
            # outputs_length도 다시 계산해야하나? (이것도 decoder에 있는 듯)
            # beam_results, beam_scores, timesteps, out_lens = decoder.decode(output)
            loss = criterion(x, outputs, lx, outputs_length)

        total_loss += float(loss.item())

        batch_bar.set_postfix(
            loss = f"{train_loss/ (i+1):.4f}",
            lr = f"{optimizer.param_groups[0]['lr']}"
        )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update() 
        batch_bar.update()
    
    batch_bar.close()
    total_loss = float(total_loss / len(train_loader)) # TODO

    return train_loss 

# %% Validation
def evaluate(data_loader, model):
    model.eval()
    batch_bar = tqdm(total=len(data_loader), dynamic_ncols=True, leave=False, position=0, desc='Val', ncols=5)
    
    val_dist = 0
    val_loss = 0
    for i, data in enumerate(train_loader):

        x, y, lx, ly = data
        x, y, lx, ly = x.to(device), y.to(device), lx.to(device), ly.to(device)
        
        with torch.no_grad():
            outputs, outputs_length = model(x, lx)
            x = x.permute(1, 0, 2)
            # 수정 
            loss = criterion(x, outputs, lx, outputs_length)

        val_loss += float(loss.item())

        batch_bar.set_postfix(
            loss = f"{val_loss/ (i+1):.4f}",
        )
        batch_bar.update()
    
    batch_bar.close()
    val_loss = float(val_loss / len(data_loader)) # TODO

    return loss, val_dist

# %% wandb 
wandb.login(key="0699a3c4c17f76e3d85a803c4d7039edb8c3a3d9")
run = wandb.init(
    name = "early-submission", ## Wandb creates random run names if you skip this field
    reinit = True, ### Allows reinitalizing runs when you re-run this cell
    # run_id = ### Insert specific run id here if you want to resume a previous run
    # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
    project = "hw3p2-ablations", ### Project should be created in your wandb account 
    config = config ### Wandb Config for your run
)

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
    
    val_loss, val_dist = evaluate(model, val_loader, criterion)
    print("Val Dist {:.04f}%\t Val Loss {:.04f}".format(val_dist, val_loss))

    wandb.log({"train_loss":train_loss, "validation_loss": val_loss,
               "validation_Dist":val_dist, "learning_Rate": curr_lr})
    
    scheduler.step()
    # 수정
    # HINT: Calculating levenshtein distance takes a long time. Do you need to do it every epoch?
    # Does the training step even need it? 
    if val_loss < best_val_loss:
      os.makedirs(model_dir, exist_ok=True)
      path = os.path.join(model_dir, 'checkpoint' + '.pth')
      print("Saving model")
      torch.save({'model_state_dict':model.state_dict(),
                  'optimizer_state_dict':optimizer.state_dict(),
                  'scheduler_state_dict':scheduler.state_dict(),
                  'val_loss': val_loss, 
                  'epoch': epoch}, path)
      best_val_loss = val_loss
      wandb.save('checkpoint.pth')

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
    log_probs_input=False
)#TODO 

# 수정
def make_output(h, lh, decoder, LABELS):
  
    beam_results, beam_scores, timesteps, out_seq_len = decoder_test.decode() #TODO
    batch_size = 0 #What is the batch size

    dist = 0
    preds = []
    for i in range(batch_size): # Loop through each element in the batch

        h_sliced = 0#TODO: Obtain the beam results
        h_string = 0#TODO: Convert the beam results to phonemes
        preds.append(h_string)
        # calculate_levenshtein_distance(h_string, y[i]) #TODO: Calculate the levenshtein distance
    
    return preds

# %% Make predictions
#TODO:

torch.cuda.empty_cache()
path = os.path.join(model_dir, 'checkpoint' + '.pth')
checkpoint = torch.load(path,  map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# 수정
predictions = make_output(test_loader, model, decoder, LABELS)

df = pd.read_csv('/data/test-clean/transcript/random_submission.csv')
df.label = predictions

df.to_csv('results/submission_early.csv', index = False)
#!kaggle competitions submit -c <competition> -f submission.csv -m "I made it!"