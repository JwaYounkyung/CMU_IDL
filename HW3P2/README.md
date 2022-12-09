# CMU 11-785 Fall 2022 Homework 3 P2

## Dataset

You can download the data using this command.

```bash
kaggle competitions download -c 11-785-f22-hw3p2
unzip -q 11-785-f22-hw3p2.zip
```
### Number of data
- train-clean-360 : 104014
- train-clean-100 : 28539
- dev-clean : 2703
- test-clean : 2620

### Directory
Eventually, the directory structure should look like this:

* this repo
  * data
    * dev-clean
    * test-clean
    * train-clean-100
    * train-clean-360
    * phonetices.py
  * models
    * basic.py
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