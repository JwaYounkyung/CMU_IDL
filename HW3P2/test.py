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

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)

# %% Hyperparameters
model_dir = os.path.join(os.path.dirname(__file__), 'weights/lstm')
gpu_ids = [0, 1, 2, 3]
batch_size = 32
if device != 'cpu':
    batch_size *= len(gpu_ids)
    
config = {
    "batch_size": batch_size, # 수정
    "num_workers": 24, # 수정
    
    "architecture" : "lstm",
    "embedding_size": 64,
    "hidden_size" : 128,
    "num_layers" : 2,
    "dropout" : 0.25,
    "bidirectional" : True,

    "beam_width_train" : 2,
    "beam_width_test" : 50,
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
test_data = AudioDatasetTest(root, PHONEMES, "test-clean", transforms=None) #TODO

test_loader = torch.utils.data.DataLoader(test_data, num_workers=config['num_workers'], 
                                          batch_size=config['batch_size'], pin_memory=True, 
                                          shuffle=False, collate_fn=test_data.collate_fn)

print("Batch size: ", config['batch_size'])
print("Test dataset samples = {}, batches = {}".format(test_data.__len__(), len(test_loader)))

# sanity check
for data in test_loader:
    x_test, lx_test = data
    print(x_test.shape, lx_test.shape)
    break 
input_size = x_test.shape[2]

# %% Model Config
OUT_SIZE = len(LABELS)
print(OUT_SIZE)

model = Network(input_size, config["embedding_size"], config["hidden_size"], config["num_layers"], 
                config["dropout"], config["bidirectional"], OUT_SIZE)
# model = torch.nn.DataParallel(model, device_ids=gpu_ids)
model.to(device)
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

path = os.path.join(model_dir, 'checkpoint' + '.pth')
checkpoint = torch.load(path,  map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

predictions = predict(model, test_loader, decoder_test, LABELS)

df = pd.read_csv('data/test-clean/transcript/random_submission.csv')
df.label = predictions

df.to_csv('results/submission_early.csv', index = False)
#!kaggle competitions submit -c 11-785-f22-hw3p2 -f results/submission_early.csv