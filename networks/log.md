# 03-22 角度1[100 epochs]无预训练------------
Training batch 230/236Epoch 99 result:
Avg loss (train): 0.7125
Avg acc (train): 4.9025


# 03-22 角度1[100 epochs]------------
问题探究：
    1、训练、验证集划分是否应该随机
Training batch 230/236Epoch 99 result:
Avg loss (train): 0.1710
Avg acc (train): 7.5085
----------

# 03-22 角度1[20 epochs]------------
Training batch 230/236Epoch 19 result:
Avg loss (train): 0.4472
Avg acc (train): 6.3517

Test batch 230/236Evaluation completed in 7m 33s
Avg loss (test): 0.3940
Avg acc (test): 6.5424
----------

# 【100 epochs】--------------------------------------------------------------
问题探究：
    1、验证集过大，增大训练/验证比例
    2、训练集上精度也很低，可能训练过程有误
---------- 验证集
Test batch 350/358Evaluation completed in 7m 43s
Avg loss (test): 3.6938
Avg acc (test): 3.8966
----------
---------- 训练集
Test batch 1240/1244Evaluation completed in 26m 14s
Avg loss (test): 0.2169
Avg acc (test): 7.2484
----------

# 【2 epochs】----------------------------------------------------------------
Epoch 0 result:
Avg loss (train): 0.8169
Avg acc (train): 4.5161
Avg loss (val): 1.3110
Avg acc (val): 3.4385
----------
Epoch 1/2
----------
Training batch 620/622.0
Validation batch 300/358
Epoch 1 result:
Avg loss (train): 0.7865
Avg acc (train): 4.5531
Avg loss (val): 1.2316
Avg acc (val): 4.7654
----------
Training completed in 112m 30s
Best acc: 4.7654