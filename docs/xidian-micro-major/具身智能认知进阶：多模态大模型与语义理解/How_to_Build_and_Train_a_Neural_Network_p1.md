数据集

MINIST数据集 CIFFAR-10数据集

训练集 验证集 测试集

使用训练集训练模型 -> 使用验证集评估模型 -> 根据在验证集上获得的效果调整模型 -> 选择最佳效果模型（若效果不好继续重复训练） -> 使用测试集确认模型的效果


损失函数

Trail -> Evaluate -> Adjust 

做题 -> 批改 -> 订正

训练代码对应的学习过程——forward -> Loss -> Backward -> Step

前向传播 -> 计算损失 -> 反向传播 -> 参数更新

```Python
outputs = model(inputs)
loss = criterion(outputs, target)
optimizer.zero_gard()
loss.backward()
optimizer.step()
```

`zero_gard`:清空上一轮梯度，避免梯度累加

`backward`：计算每个参数应该如何变化

`step`：真正修改网格权重

什么是损失函数？——Lossfunction=模型学习的“评分标准"

Prediction -> 比较 -> GroundTruth -> 误差大小 -> loss 

单个样本误差vs整体训练目标

不同任务如何选择损失函数？损失函数要和任务目标一致

为什么MINIST使用交叉熵？

损失函数的设计会影响训练效果

训练日志：Loss下降不代表一定泛化良好：训练正常 train loss下降 val loss也下降 accuracy逐步上升;可能过拟合 trian loss继续下降 val loss不降反升 测试集效果变差;可能欠拟合 train loss很高 val loss也很高 模型表达能力不足



过拟合情况：训练数据集单一，样本不足；训练数据中噪声过大；模型过于复杂

解决方法：数据增强，扩大数据集；数据特征预处理；正则化，L1，L2，Dropout；简化模型，奥卡姆剃刀原则


欠拟合情况：特征维度过少，模型过于简单，导致拟合的函数无法满足训练集，误差较大

解决方法：增加特征维度；增加模型复杂度；精简训练数据与特征


优化器 批量梯度下降

偏导数决定爬坡的方向，学习率决定步伐大小

缺陷：梯度为0，无论怎么更新权重，loss都不再改变，此时梯度下降法失效

随机梯度下降


