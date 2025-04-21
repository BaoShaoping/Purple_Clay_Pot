import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
import shutil
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch
from torchvision import models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = models.resnet18(pretrained=True)
model.fc = torch.nn.Identity()
model = model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset_path = r'C:\Users\STAR\Desktop\grind\mentor\deng\dataset'
cluster_path = os.path.join(".", "clustered_data")
os.makedirs(cluster_path, exist_ok=True)

features = []
img_paths = []
valid_count = 0
print("Starting feature extraction...")
for i, img_name in enumerate(os.listdir(dataset_path)):
    try:
        img_path = os.path.join(dataset_path, img_name)
        img = Image.open(img_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(img_tensor).cpu().numpy().flatten()
        features.append(feat)
        img_paths.append(img_path)
        valid_count += 1
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1} images, {valid_count} valid so far...")
    except Exception as e:
        print(f"Error processing {img_name}: {str(e)}")
        continue

features = np.array(features)
print(f"Feature extraction complete. Total valid images: {valid_count}")
print(f"Feature shape: {features.shape}")

if features.shape[0] > 1:
    pca = PCA(n_components=50, random_state=0)
    features_reduced = pca.fit_transform(features)
    print(f"PCA explained variance ratio: {sum(pca.explained_variance_ratio_):.2f}")
else:
    features_reduced = features
    print("Not enough samples for PCA. Using original features.")

n_clusters = 5
print(f"Clustering into {n_clusters} groups...")
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features_reduced)

for i in range(n_clusters):
    os.makedirs(os.path.join(cluster_path, f"cluster_{i}"), exist_ok=True)

print("Organizing images into clusters...")
for path, label in zip(img_paths, kmeans.labels_):
    try:
        shutil.copy(path, os.path.join(cluster_path, f"cluster_{label}"))
    except Exception as e:
        print(f"Failed to copy {path}: {str(e)}")

print("分类完成.")
