# CMU 11-785 Fall 2022 Homework 2 P2

## Dataset

You can download the data using this command.

```bash
kaggle competitions download -c 11-785-f22-hw2p2-classification
unzip -qo '11-785-f22-hw2p2-classification.zip' -d './data'

kaggle competitions download -c 11-785-f22-hw2p2-verification
unzip -qo '11-785-f22-hw2p2-verification.zip' -d './data'
```

Eventually, the directory structure should look like this:

* this repo
  * data
    * 11-785-f22-hw2p2-classification 
    * verification
  * models
    * backbones
    * basic.py
  * main.py
  * README.md

## Implementation

To train and test the model, use

```bash
python main.py --wandb_key your_key --gpu_ids what_you_have
```
By default, test results are stored under `results/`.

## Results
For the experiments, I used 3 kinds of structure.
* basic
* ResNet 50
* ResNet 100

The highest validation accuracy is 79.72456661 using ResNet50_experiment3.
I mentioned the details on the HW2P2_results.csv.
You can see the graph in Experiments.png


