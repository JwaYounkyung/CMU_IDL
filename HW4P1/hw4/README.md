- dataset (579,)
- concate (2075677,)
- seq_len (691893, 3)
- batchs (2, 3) 345947개

Test용
- dataset (2,)
- seq_len (2602, 3)
- batchs (2, 3) 1301개


# Shape

## Model
- x      [2, 3]
- emb    [2, 3, 64]
- rnn    [2, 3, 128]
- fc     [2, 3, 33278]

## Prediction
- inp    [128, 20]
- out    [128, 20, 33278]

## Generation
- inp    [32, 20]
- predictions [32, 33278]
- armax [32, 1]


Q
1. seq_len로 데이터 자를 때 마지막 부분 어케 처리함? - last_batch discard
2. data 불러 올때마다 shuffle 하는거 맞아? - iter에서 for문만 계속 불러짐
