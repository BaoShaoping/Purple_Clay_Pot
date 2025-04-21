# Purple_Clay_Pot
Purple clay teapot shape classification system based on deep learning.

# 一、框架设计

本项目采用轻量级卷积神经网络实现紫砂壶器型分类，主要流程包含：
- 数据聚类与预处理
- 数据增强与划分
- 卷积神经网络构建
- 模型训练与验证
- 预测结果可视化
# 二、数据准备
1. 数据聚类
采用K-Means算法进行初始数据划分，生成5个类别目录：cluster_0至cluster_4
关键参数：
python
n_clusters = 5  # 预设类别数
max_iter = 300  # 最大迭代次数
2. 数据增强
增强策略：
python
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomRotation(15)
transforms.ColorJitter(0.2, 0.2, 0.2)
数据集扩展至400+样本

3. 数据划分
目录结构：
dataset/
├── train/   # 训练集（70%）
├── val/     # 验证集（15%）
└── test/    # 测试集（15%）

# 三、模型构建
1. 网络架构
组件	参数说明
输入尺寸	128×128×3
卷积层	3层（通道数16→32→64）
池化层	最大池化（2×2窗口）
全连接层	128单元+5分类输出
激活函数	ReLU
Dropout率	0.5
2. PyTorch实现
python
class SimpleZishaCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            # ...中间层省略...
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*16*16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
3. 训练配置
参数	设置值
损失函数	CrossEntropyLoss
优化器	Adam(lr=0.001)
Batch Size	16
Epoch	20
学习率衰减	StepLR(step=10, gamma=0.1)
模型保存	final_model.pth
# 四、实验结果
分类性能（测试集）
评价指标	数值
准确率	89.2%
Macro-F1	0.874
