# Purple_Clay_Pot
Purple clay teapot shape classification system based on deep learning.

# 步骤

1. 数据聚类。执行cluster.py，使用K-Means算法进行初始聚类，将数据集划分为5类，分别是cluster_0、cluster_1、cluster_2、cluster_3、cluster_4，；
2. 数据增强。执行augmentation.py，使用水平翻转和旋转、颜色抖动和仿射变换将数据集扩大到400多张；
3. 数据集划分。执行divide.py，将数据集按照70%，15%，15%的比例划分为训练集、验证集和测试集；
4. 模型训练。执行train.py，采用3个卷积层，每层后接最大池化层，然后展开一个全连接层的CNN架构，最后将训练参数保存在final_model.pth中；
5. 预测执行。执行predict.py，对输入图像进行预处理和模型加载后，输出置信度最大的类别和对应的置信度；
