import torch
from torchsummary import summary
import torchvision #This library is used for image-based operations (Augmentations)
import os
import gc
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import glob
import wandb

import argparse

from models import basic
from backbones import get_model

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# !kaggle competitions download -c 11-785-f22-hw2p2-classification
# !unzip -qo '11-785-f22-hw2p2-classification.zip' -d '/content/data'

# !kaggle competitions download -c 11-785-f22-hw2p2-verification
# !unzip -qo '11-785-f22-hw2p2-verification.zip' -d '/content/data'

def parse_args(argv=None):
    parser = argparse.ArgumentParser(__file__)
    parser.add_argument(
        '--model_dir', default=os.path.join(os.path.dirname(__file__), 'data/r50'))

    # parser.add_argument('--batch_size', type=int, default=1024)
    # parser.add_argument('--lr', type=float, default=0.01)
    # parser.add_argument('--epoch', type=int, default=90)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gpu_ids', nargs="+", default=[2, 3])

    args = parser.parse_args(argv)
    return args

args = parse_args()

config = {
    'batch_size': 32*2, # Increase this if your GPU can handle it
    'lr': 0.1,
    'epochs': 50, # 10 epochs is recommended ONLY for the early submission - you will have to train for much longer typically.
    # Include other parameters as needed.
}

#torch.cuda.set_device(args.local_rank)
DATA_DIR = 'data/11-785-f22-hw2p2-classification/'# TODO: Path where you have downloaded the data
TRAIN_DIR = os.path.join(DATA_DIR, "classification/train") 
VAL_DIR = os.path.join(DATA_DIR, "classification/dev")
TEST_DIR = os.path.join(DATA_DIR, "classification/test")

# Data mean and std for normalization
'''
data_transformer = torchvision.transforms.Compose([
                    torchvision.transforms.GaussianBlur(kernel_size=7),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor()])
train_dataset = torchvision.datasets.ImageFolder(TRAIN_DIR, transform = data_transformer)

meanRGB = []
stdRGB = []

for x, _ in train_dataset:
    meanRGB.append(np.mean(x.numpy(), axis=(1,2)))
    stdRGB.append(np.std(x.numpy(), axis=(1,2)))        

meanR = np.mean([m[0] for m in meanRGB])
meanG = np.mean([m[1] for m in meanRGB])
meanB = np.mean([m[2] for m in meanRGB])

stdR = np.mean([s[0] for s in stdRGB])
stdG = np.mean([s[1] for s in stdRGB])
stdB = np.mean([s[2] for s in stdRGB])

print(f"{meanR}, {meanG}, {meanB}") # 0.5130248665809631, 0.4033524692058563, 0.35215428471565247
print(f"{stdR}, {stdG}, {stdB}") # 0.2638624310493469, 0.22904270887374878, 0.2152242362499237
'''
# Transforms using torchvision - Refer https://pytorch.org/vision/stable/transforms.html
train_transforms = torchvision.transforms.Compose([ 
    # Implementing the right transforms/augmentation methods is key to improving performance.
                    # torchvision.transforms.RandomApply(torch.nn.ModuleList([
                    # torchvision.transforms.ColorJitter(brightness=(0.5,2), contrast=(0.5,2), saturation=(0.5,2))]), p=0.3),
                    torchvision.transforms.GaussianBlur(kernel_size=7),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.5130248665809631, 0.4033524692058563, 0.35215428471565247], std=[0.2638624310493469, 0.22904270887374878, 0.2152242362499237]), 
                    ])
# Most torchvision transforms are done on PIL images. So you convert it into a tensor at the end with ToTensor()
# But there are some transforms which are performed after ToTensor() : e.g - Normalization
# Normalization Tip - Do not blindly use normalization that is not suitable for this dataset

val_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


train_dataset = torchvision.datasets.ImageFolder(TRAIN_DIR, transform = train_transforms)
val_dataset = torchvision.datasets.ImageFolder(VAL_DIR, transform = val_transforms)
# You should NOT have data augmentation on the validation set. Why?


# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = config['batch_size'], 
                                           shuffle = True,num_workers = args.num_workers, pin_memory = True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = config['batch_size'], 
                                         shuffle = False, num_workers = 2)

# You can do this with ImageFolder as well, but it requires some tweaking
class ClassificationTestDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, transforms):
        self.data_dir   = data_dir
        self.transforms = transforms

        # This one-liner basically generates a sorted list of full paths to each image in the test directory
        self.img_paths  = list(map(lambda fname: os.path.join(self.data_dir, fname), sorted(os.listdir(self.data_dir))))

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        return self.transforms(Image.open(self.img_paths[idx]))

test_dataset = ClassificationTestDataset(TEST_DIR, transforms = val_transforms) #Why are we using val_transforms for Test Data?
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = config['batch_size'], shuffle = False,
                         drop_last = False, num_workers = 2)

print("Number of classes: ", len(train_dataset.classes))
print("No. of train images: ", train_dataset.__len__())
print("No. of val images: ", val_dataset.__len__())
print("No. of test images: ", test_dataset.__len__())
print("Shape of image: ", train_dataset[0][0].shape)
print("Batch size: ", config['batch_size'])
print("Train batches: ", train_loader.__len__())
print("Val batches: ", val_loader.__len__())
print("Test batches: ", test_loader.__len__())
            
# %% model
#model = basic.Network()
model = get_model("r50", dropout=0.0, fp16=True, num_features=len(train_dataset.classes))

# DataParallel
#model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
model.to(device)
summary(model, (3, 224, 224))

# %% define loss and optimizer
criterion = torch.nn.CrossEntropyLoss()# TODO: What loss do you need for a multi class classification problem?
optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=1e-4)

# TODO: Implement a scheduler (Optional but Highly Recommended)
# You can try ReduceLRonPlateau, StepLR, MultistepLR, CosineAnnealing, etc.
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.99)
scaler = torch.cuda.amp.GradScaler() # Good news. We have FP16 (Mixed precision training) implemented for you
# It is useful only in the case of compatible GPUs such as T4/V100


def train(model, dataloader, optimizer, criterion):
    
    model.train()

    # Progress Bar 
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5) 
    
    num_correct = 0
    total_loss = 0

    for i, (images, labels) in enumerate(dataloader):
        
        optimizer.zero_grad() # Zero gradients

        images, labels = images.to(device), labels.to(device)
        
        with torch.cuda.amp.autocast(): # This implements mixed precision. Thats it! 
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Update no. of correct predictions & loss as we iterate
        num_correct += int((torch.argmax(outputs, axis=1) == labels).sum())
        total_loss += float(loss.item())

        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * num_correct / (config['batch_size']*(i + 1))),
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            num_correct=num_correct,
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))
        
        scaler.scale(loss).backward() # This is a replacement for loss.backward()
        scaler.step(optimizer) # This is a replacement for optimizer.step()
        scaler.update() 

        # TODO? Depending on your choice of scheduler,
        # You may want to call some schdulers inside the train function. What are these?
      
        batch_bar.update() # Update tqdm bar

    scheduler.step()
    batch_bar.close() # You need this to close the tqdm bar

    acc = 100 * num_correct / (config['batch_size']* len(dataloader))
    total_loss = float(total_loss / len(dataloader))

    return acc, total_loss


def validate(model, dataloader, criterion):
  
    model.eval()
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='Val', ncols=5)

    num_correct = 0.0
    total_loss = 0.0

    for i, (images, labels) in enumerate(dataloader):
        
        # Move images to device
        images, labels = images.to(device), labels.to(device)
        
        # Get model outputs
        with torch.inference_mode():
            outputs = model(images)
            loss = criterion(outputs, labels)

        num_correct += int((torch.argmax(outputs, axis=1) == labels).sum())
        total_loss += float(loss.item())

        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * num_correct / (config['batch_size']*(i + 1))),
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            num_correct=num_correct)

        batch_bar.update()
        
    batch_bar.close()
    acc = 100 * num_correct / (config['batch_size']* len(dataloader))
    total_loss = float(total_loss / len(dataloader))
    return acc, total_loss

gc.collect() # These commands help you when you face CUDA OOM error
torch.cuda.empty_cache()
#'''
wandb.login(key="0699a3c4c17f76e3d85a803c4d7039edb8c3a3d9") #API Key is in your wandb account, under settings (wandb.ai/settings)

# Create your wandb run
run = wandb.init(
    name = "r50", ## Wandb creates random run names if you skip this field
    reinit = True, ### Allows reinitalizing runs when you re-run this cell
    # run_id = ### Insert specific run id here if you want to resume a previous run
    # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
    project = "hw2p2-ablations", ### Project should be created in your wandb account 
    config = config ### Wandb Config for your run
)

best_valacc = 0.0


for epoch in range(config['epochs']):

    curr_lr = float(optimizer.param_groups[0]['lr'])

    train_acc, train_loss = train(model, train_loader, optimizer, criterion)
    
    print("\nEpoch {}/{}: \nTrain Acc {:.04f}%\t Train Loss {:.04f}\t Learning Rate {:.04f}".format(
        epoch + 1,
        config['epochs'],
        train_acc,
        train_loss,
        curr_lr))
    
    val_acc, val_loss = validate(model, val_loader, criterion)
    
    print("Val Acc {:.04f}%\t Val Loss {:.04f}".format(val_acc, val_loss))

    wandb.log({"train_loss":train_loss, 'train_Acc': train_acc, 'validation_Acc':val_acc, 
               'validation_loss': val_loss, "learning_Rate": curr_lr})
    
    # If you are using a scheduler in your train function within your iteration loop, you may want to log
    # your learning rate differently 

    # #Save model in drive location if val_acc is better than best recorded val_acc
    if val_acc >= best_valacc:
      os.makedirs(args.model_dir, exist_ok=True)
      path = os.path.join(args.model_dir, 'checkpoint' + '.pth')
      print("Saving model")
      torch.save({'model_state_dict':model.state_dict(),
                  'optimizer_state_dict':optimizer.state_dict(),
                  #'scheduler_state_dict':scheduler.state_dict(),
                  'val_acc': val_acc, 
                  'epoch': epoch}, path)
      best_valacc = val_acc
      wandb.save('checkpoint.pth')
      # You may find it interesting to exlplore Wandb Artifcats to version your models
run.finish()
#'''

def test(model,dataloader):

  model.eval()
  batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='Test')
  test_results = []
  
  for i, (images) in enumerate(dataloader):
      # TODO: Finish predicting on the test set.
      images = images.to(device)

      with torch.inference_mode():
        outputs = model(images)

      outputs = torch.argmax(outputs, axis=1).detach().cpu().numpy().tolist()
      test_results.extend(outputs)
      
      batch_bar.update()
      
  batch_bar.close()
  return test_results

path = os.path.join(args.model_dir, 'checkpoint' + '.pth')
checkpoint = torch.load(path,  map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
#'''
test_results = test(model, test_loader)
with open("results/classification_submission.csv", "w+") as f:
    f.write("id,label\n")
    for i in range(len(test_dataset)):
        f.write("{},{}\n".format(str(i).zfill(6) + ".jpg", test_results[i]))
#'''    

known_regex = "data/verification/known/*/*"
known_paths = [i.split('/')[-2] for i in sorted(glob.glob(known_regex))] 
# This obtains the list of known identities from the known folder

unknown_regex = "data/verification/unknown_test/*" #Change the directory accordingly for the test set

# We load the images from known and unknown folders
unknown_images = [Image.open(p) for p in tqdm(sorted(glob.glob(unknown_regex)))]
known_images = [Image.open(p) for p in tqdm(sorted(glob.glob(known_regex)))]

# Why do you need only ToTensor() here?
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()])

unknown_images = torch.stack([transforms(x) for x in unknown_images])
known_images  = torch.stack([transforms(y) for y in known_images ])
#Print your shapes here to understand what we have done
print("Unknown Images: ", unknown_images.shape)
print("Known Images: ", known_images.shape)

# You can use other similarity metrics like Euclidean Distance if you wish
similarity_metric = torch.nn.CosineSimilarity(dim= 1, eps= 1e-6) 


def eval_verification(unknown_images, known_images, model, similarity, batch_size= config['batch_size'], mode='val'): 

    unknown_feats, known_feats = [], []

    batch_bar = tqdm(total=len(unknown_images)//batch_size, dynamic_ncols=True, position=0, leave=False, desc=mode)
    model.eval()

    # We load the images as batches for memory optimization and avoiding CUDA OOM errors
    for i in range(0, unknown_images.shape[0], batch_size):
        unknown_batch = unknown_images[i:i+batch_size] # Slice a given portion upto batch_size
        
        with torch.no_grad():
            unknown_feat = model(unknown_batch.float().to(device), return_feats=True) #Get features from model         
        unknown_feats.append(unknown_feat)
        batch_bar.update()
    
    batch_bar.close()
    
    batch_bar = tqdm(total=len(known_images)//batch_size, dynamic_ncols=True, position=0, leave=False, desc=mode)
    
    for i in range(0, known_images.shape[0], batch_size):
        known_batch = known_images[i:i+batch_size] 
        with torch.no_grad():
              known_feat = model(known_batch.float().to(device), return_feats=True)
          
        known_feats.append(known_feat)
        batch_bar.update()

    batch_bar.close()

    # Concatenate all the batches
    unknown_feats = torch.cat(unknown_feats, dim=0)
    known_feats = torch.cat(known_feats, dim=0)

    similarity_values = torch.stack([similarity(unknown_feats, known_feature) for known_feature in known_feats])
    # Print the inner list comprehension in a separate cell - what is really happening?

    predictions = similarity_values.argmax(0).squeeze().cpu().numpy() #Why are we doing an argmax here?

    # Map argmax indices to identity strings
    pred_id_strings = [known_paths[i] for i in predictions]
    
    if mode == 'val':
      true_ids = pd.read_csv('data/verification/dev_identities.csv')['label'].tolist()
      accuracy = accuracy_score(pred_id_strings, true_ids)
      print("Verification Accuracy = {}".format(accuracy))
    
    return pred_id_strings

pred_id_strings = eval_verification(unknown_images, known_images, model, similarity_metric, config['batch_size'], mode='test')

with open("results/verification_submission.csv", "w+") as f:
    f.write("id,label\n")
    for i in range(len(pred_id_strings)):
        f.write("{},{}\n".format(i, pred_id_strings[i]))
