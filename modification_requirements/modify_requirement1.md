EnhancedKeyFrameSelector目前存在如下的问题：

1.content_aware_sampling这个模块设计不合理

​	新的设计：content_scores用分割图像面积进行约束，这样content_scores直接反应的是肿瘤的面积大小；采用非极大值抑制得到最好的K帧，窗口数默认为5

2.attention_guided_sampling的输入有问题

​	现在的输入attention_scores是x在H，W维度上平局池化得到的，这样损失太多了，而且没有任何信息，应该先通过几层卷积层提取特征，再计算分数

3.fuse_strategies里面是选择得分最高的帧，但实际上每种结果的选择帧可能并不完全重合，使用一个简单的网络进行选取，这样也方便后面梯度回传

4.模型需要具备对于分割图像面积的预测能力，用Brats 3种Mask的面积作为label，用一个神经网络去进行预测，并且设置参数可以是否冻结这个网络。

5.对于Mask的预测，比如模型提取出来的是（B，C，H，W，K）这K帧，那么算预测Mask面积也是拿这K帧计算

6.calculate_diversity_loss是用selected_frames展成特征向量计算相似度的，这样很不合理，用Med-CLIP的编码特征来计算相似度衡量，请自行下载预训练好的Med-CLIP

VoCo计算目前存在如下的问题：

VoCo的计算：对于（B，C，H，W，K）选出来的K帧，对于每一帧（B，C，H，W，1），通过固定crop成crop^2个（B,C,H//crop,W//crop,1），然后随机crop一个块，与这些计算面积，作为子监督，现在的32，32，32的VoCo切片不对

认真分析并修改我提到的上述问题，修改代码，直到代码能够跑通，看到val损失工作正常，并一条条确认我提出的问题完全解决，现在显卡0被使用了，你可以使用显卡2或者3测试，你可以用conda activate ume激活之前配置好的环境