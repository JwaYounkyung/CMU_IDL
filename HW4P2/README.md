# CMU 11-785 Fall 2022 Homework 4 P2

## Dataset

You can download the data using this command.

```bash
kaggle competitions download -c 11-785-f22-hw4p2
unzip -q 11-785-f22-hw4p2.zip
```
### Number of data
- train-clean-100 : 28539
- dev-clean : 2703
- test-clean : 2620

### Directory
Eventually, the directory structure should look like this:

* this repo
  * data
    * hw4p2
      * dev-clean
      * test-clean
      * train-clean-100
      * phonetices.py
  * models
    * LAS.py
  * modules
    * dataset.py
    * utils.py
  * results
  * weights
  * main.py
  * README.md

## Implementation

To train and test the model, use

```bash
python main.py 
```
By default, test results are stored under `results/`.

# Shape

## Listener
- x [4, 1520, 15] x_lens [1117, 1498, 1520, 1075]
- packed       [5210, 15]
- packed lstm  [5210, 512]
- unpacked     [4, 1520, 512]
- pBLSTMs [650, 512] lens [139, 187, 190, 134]

## pBLSTM 
1
- x [4, 1520, 512] x_lens [1117, 1498, 1520, 1075]
- trunc x [4, 760, 1024] x_lens [558, 749, 760, 537] # 버림
- packed [2604, 1024]
- return [2604, 512]
2
- packed [2604, 512] x_lens [558, 749, 760, 537]
- x [4, 760, 512]
- trunc x [4, 380, 1024] x_lens [279, 374, 380, 268]
- packed [1301, 1024]
- return [1301, 512]
3
- packed [1301, 512] x_lens [279, 374, 380, 268]
- x [4, 380, 512]
- trunc x [4, 190, 1024] x_lens [139, 187, 190, 134]
- packed [650, 1024]
- return [650, 512]

## Attention