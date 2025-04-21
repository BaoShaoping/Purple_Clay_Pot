import os
from PIL import Image
from torchvision import transforms

def basic_transform(image):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15)
    ])
    return transform(image)

def advanced_transform(image):
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1))
    ])
    return transform(image)

def augment_images(input_dir, output_dir, target_per_class=80):
    classes = os.listdir(input_dir)
    for cls in classes:
        cls_path = os.path.join(input_dir, cls)
        output_cls_path = os.path.join(output_dir, cls)
        os.makedirs(output_cls_path, exist_ok=True)
        images = [f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.png'))]
        for img_name in images:
            img_path = os.path.join(cls_path, img_name)
            base_image = Image.open(img_path)
            base_image.save(os.path.join(output_cls_path, f"orig_{img_name}"))
            for i in range(3):
                transformed = basic_transform(base_image)
                transformed.save(os.path.join(output_cls_path, f"basic{i}_{img_name}"))
            for i in range(2):
                transformed = advanced_transform(base_image)
                transformed.save(os.path.join(output_cls_path, f"adv{i}_{img_name}"))

augment_images("./clustered_data", "./augmented_data", target_per_class=80)
