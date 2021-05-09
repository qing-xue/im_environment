# 20-05-09 Test Resnet, 三分类模型
Training:Epoch[000/100] Iteration[060/622] Loss: 0.8149 Acc:50.83%
Training:Epoch[000/100] Iteration[070/622] Loss: 0.7927 Acc:51.88%
...
Training:Epoch[006/100] Iteration[190/622] Loss: 0.5552 Acc:65.59%
Training:Epoch[006/100] Iteration[200/622] Loss: 0.7404 Acc:65.50%
Training:Epoch[006/100] Iteration[210/622] Loss: 0.7220 Acc:65.27%

# 仅采用角度1的图片

* 是否使用预训练模型
  
    | model          | epochs | Train acc | Val acc | 
    | ----           |  ----  | ----      |  ----   | 
    | _20_a1_retrain |  20    | xxxxx     | 0.4832  | 
    | _100_a1        |  100   | xxxxx     | 0.4562  | 

* 回归模型 V.S 分类模型（都为预训练）
  
    | model          | epochs | Train acc | Val acc | 
    | ----           |  ----  | ----      |  ----   | 
    | _30_a1_re      |  30    | xxxxx     | 0.5387  | 
    | _20_a1_retrain |  100   | 0.6169    | 0.4832  | 

# 采用3个角度的图片

* 原始污染程度3分类模型
  
    | model          | epochs | Train acc | Val acc | 
    | ----           |  ----  | ----      |  ----   | 
    | _20            |  20    | xxxxx     | 0.2642  | 
    | _100           |  100   | xxxxx     | 0.4876  | 

# 问题探究
1. 验证集过大，增大训练/验证比例
2. 训练、验证集划分是否应该随机
