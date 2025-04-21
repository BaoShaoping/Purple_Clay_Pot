import os
from PIL import Image
from torchvision import transforms


# 定义增强函数：基础增强
def basic_transform(image):
    # 基础变换：随机水平翻转和旋转
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15)
    ])
    return transform(image)


# 定义增强函数：高级增强
def advanced_transform(image):
    # 高级变换：颜色抖动和仿射变换
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1))
    ])
    return transform(image)


# 定义图像增强处理函数
def augment_images(input_dir, output_dir, target_per_class=80):
    # 获取输入目录下的所有类别
    classes = os.listdir(input_dir)

    # 遍历每个类别
    for cls in classes:
        cls_path = os.path.join(input_dir, cls)
        output_cls_path = os.path.join(output_dir, cls)
        os.makedirs(output_cls_path, exist_ok=True)  # 创建输出目录

        # 获取该类别下的所有图片
        images = [f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.png'))]

        # 对每张图片进行处理
        for img_name in images:
            img_path = os.path.join(cls_path, img_name)
            base_image = Image.open(img_path)

            # 保存原始图像
            base_image.save(os.path.join(output_cls_path, f"orig_{img_name}"))

            # 阶段一：生成 3 个基础增强样本
            for i in range(3):
                transformed = basic_transform(base_image)
                transformed.save(os.path.join(output_cls_path, f"basic{i}_{img_name}"))

            # 阶段二：生成 2 个高级增强样本
            for i in range(2):
                transformed = advanced_transform(base_image)
                transformed.save(os.path.join(output_cls_path, f"adv{i}_{img_name}"))


augment_images("./clustered_data", "./augmented_data", target_per_class=80)
