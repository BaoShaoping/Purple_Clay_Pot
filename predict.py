import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class SimpleZishaCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class ZishaClassifier:
    def __init__(self, model_path, class_names, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.class_names = class_names
        self.model = SimpleZishaCNN(num_classes=len(class_names))
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model = self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def classify(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"图像加载失败: {str(e)}")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(img_tensor)
            prob = torch.softmax(output, dim=1)
        pred_prob, pred_idx = torch.max(prob, 1)
        return img, self.class_names[pred_idx.item()], pred_prob.item()

    def print_result(self, image_path):
        _, class_name, confidence = self.classify(image_path)
        print(f"预测类别: {class_name}")
        print(f"置信度: {confidence:.2%}")

if __name__ == "__main__":
    CLASS_NAMES = ["cluster_0", "cluster_1", "cluster_2", "cluster_3", "cluster_4"]
    MODEL_PATH = "./final_model.pth"
    TEST_IMAGE = "split_dataset/test/cluster_2/orig_IMG_20240628_141114.jpg"
    predictor = ZishaClassifier(MODEL_PATH, CLASS_NAMES)
    predictor.print_result(TEST_IMAGE)
