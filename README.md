# Purple_Clay_Pot
Purple clay teapot shape classification system based on deep learning.

# 步骤

1. 数据集划分。执行cluster.py，使用kmeans聚类算法将数据集划分为5类，分别是cluster_0、cluster_1、cluster_2、cluster_3、cluster_4，；
2. 数据增强。执行augmentation.py，使用水平翻转和旋转、颜色抖动和仿射变换将数据集扩大到400多张；
