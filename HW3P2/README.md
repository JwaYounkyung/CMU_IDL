### Shape
x [4, 1476, 15]
y [4, 129]
lx [4]
ly [4]
outputs [4, 1476, 43] -> [4, ?] using CTC Beam

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

### Data
- train-clean-360 : 104014
- train-clean-100 : 28539
- dev-clean : 2703
- test-clean : 2620

### Model 1 
- input [4, 1520, 15], lx [1117, 1498, 1520, 1075]
- packed [5210, 15]
- packed lstm [5210, 256]
- unpack [4, 1520, 256]
- fc [4, 1520, 43]

### Model 2 embedding 2
- input [4, 1520, 15], lx [1117, 1498, 1520, 1075]
- embedding1 [4, 1520, 64]
- embedding2 [4, 1520, 128]
- packed [5210, 128]
- packed lstm [5210, 256]
- unpack [4, 1520, 256]
- fc [4, 1520, 43]

### Decoder 
- outputs [B, T, 43]
- beam_results [B, 2, 1592]
- out_len [B, 2]
- top beam for the batch : beam_results[i][0][:out_len[i][0]]

### 질문
- collate_fn를 사용하면 getitem을 안하는 건가? nono 
    A. collate_fn이 getitem에서 1개씩 데이터를 불러다가 batch를 만듬    
- batch 1일때 돌아가는지 확인
    A. oo
- pad는 batch 단위로 하면 되는 건가? max_length가 그때 그때 다른데
- lstm에 dropout 들어가는데 그 밖에도 dropout 함?
- validation에서도 dist를 안구하는 건가?

# 할 일
1. multi gpu 돌아가게 만들기
torchrun --nproc_per_node=4 main.py
2. embedding with cnn and dropout 
3. train-clean-360 data 추가  
4. resnet 50?
5. transforms
6. calculate_levenshtein
7. dropout 높이기
8. normalize

 right embedding layer, recurrent layer, and classification layer - coupled with good regularization will let you cross all cutoffs.

 # Possible Problem
 1. If maximum length of the batch is too large, it cause the lack of the memory